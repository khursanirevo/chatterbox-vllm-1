#!/usr/bin/env python3
"""
Async Streaming TTS example for Chatterbox vLLM using AsyncLLMEngine.

This example demonstrates how to stream audio chunks asynchronously using vLLM's
AsyncLLMEngine, which provides better performance for real-time TTS applications.
"""

from typing import AsyncGenerator, Optional
import asyncio
import torch
import torchaudio as ta
import numpy as np
from pathlib import Path

from chatterbox_vllm.tts import ChatterboxTTS
from chatterbox_vllm.models.t3 import SPEECH_TOKEN_OFFSET, T3Config
from chatterbox_vllm.models.t3.modules.cond_enc import T3Cond
from chatterbox_vllm.models.t3.modules.learned_pos_emb import LearnedPositionEmbeddings
from chatterbox_vllm.models.s3tokenizer import drop_invalid_tokens
from chatterbox_vllm.models.s3gen import S3Gen
from chatterbox_vllm.models.voice_encoder import VoiceEncoder
from chatterbox_vllm.text_utils import punc_norm, SUPPORTED_LANGUAGES

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from safetensors.torch import load_file


class ChatterboxTTSAsync:
    """Async wrapper for ChatterboxTTS using AsyncLLMEngine."""

    ENC_COND_LEN = 6 * 24000  # S3_SR
    DEC_COND_LEN = 10 * 24000  # S3GEN_SR

    def __init__(self, target_device: str, max_model_len: int,
                 t3_engine: AsyncLLMEngine, t3_config: T3Config, t3_cond_enc,
                 t3_speech_emb: torch.nn.Embedding, t3_speech_pos_emb: LearnedPositionEmbeddings,
                 s3gen: S3Gen, ve: VoiceEncoder, default_conds,
                 variant: str = "english"):
        self.target_device = target_device
        self.max_model_len = max_model_len
        self.t3_engine = t3_engine
        self.t3_config = t3_config
        self.t3_cond_enc = t3_cond_enc
        self.t3_speech_emb = t3_speech_emb
        self.t3_speech_pos_emb = t3_speech_pos_emb
        self.s3gen = s3gen
        self.ve = ve
        self.default_conds = default_conds
        self.variant = variant

    @property
    def sr(self) -> int:
        return 24000  # S3GEN_SR

    @classmethod
    async def from_pretrained(cls,
                              repo_id: str = "ResembleAI/chatterbox",
                              revision: str = "1b475dffa71fb191cb6d5901215eb6f55635a9b6",
                              target_device: str = "cuda",
                              max_model_len: int = 1000,
                              max_batch_size: int = 10,
                              variant: str = "english",
                              **kwargs) -> 'ChatterboxTTSAsync':
        from huggingface_hub import hf_hub_download

        # Download model files
        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            hf_hub_download(repo_id=repo_id, filename=fpath, revision=revision)

        # Ensure symlink
        local_path = Path(hf_hub_download(repo_id=repo_id, filename="t3_cfg.safetensors", revision=revision))
        t3_cfg_path = local_path.parent / "t3_cfg.safetensors"
        model_safetensors_path = Path.cwd() / "t3-model" / "model.safetensors"
        model_safetensors_path.unlink(missing_ok=True)
        model_safetensors_path.symlink_to(t3_cfg_path)

        return await cls.from_local(Path(local_path).parent, target_device=target_device,
                                    max_model_len=max_model_len, max_batch_size=max_batch_size,
                                    variant=variant, **kwargs)

    @classmethod
    async def from_local(cls, ckpt_dir: str | Path, target_device: str = "cuda",
                        max_model_len: int = 1000, max_batch_size: int = 10,
                        variant: str = "english", **kwargs) -> 'ChatterboxTTSAsync':
        from chatterbox_vllm.tts import Conditionals
        import librosa

        ckpt_dir = Path(ckpt_dir)
        t3_config = T3Config()

        # Load encoder components
        t3_weights = load_file(ckpt_dir / ("t3_cfg.safetensors" if variant == "english" else "t3_mtl23ls_v2.safetensors"))

        from chatterbox_vllm.models.t3.modules.cond_enc import T3CondEnc
        t3_enc = T3CondEnc(t3_config)
        t3_enc.load_state_dict({k.replace('cond_enc.', ''): v for k, v in t3_weights.items() if k.startswith('cond_enc.')})
        t3_enc = t3_enc.to(device=target_device).eval()

        t3_speech_emb = torch.nn.Embedding(t3_config.speech_tokens_dict_size, t3_config.n_channels)
        t3_speech_emb.load_state_dict({k.replace('speech_emb.', ''): v for k, v in t3_weights.items() if k.startswith('speech_emb.')})
        t3_speech_emb = t3_speech_emb.to(device=target_device).eval()

        t3_speech_pos_emb = LearnedPositionEmbeddings(t3_config.max_speech_tokens + 2 + 2, t3_config.n_channels)
        t3_speech_pos_emb.load_state_dict({k.replace('speech_pos_emb.', ''): v for k, v in t3_weights.items() if k.startswith('speech_pos_emb.')})
        t3_speech_pos_emb = t3_speech_pos_emb.to(device=target_device).eval()

        # Calculate vLLM memory
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        unused_gpu_memory = total_gpu_memory - torch.cuda.memory_allocated()
        vllm_memory_needed = (1.55*1024*1024*1024) + (max_batch_size * max_model_len * 1024 * 128)
        vllm_memory_percent = vllm_memory_needed / unused_gpu_memory

        print(f"Giving vLLM {vllm_memory_percent * 100:.2f}% of GPU memory ({vllm_memory_needed / 1024**2:.2f} MB)")

        # Create AsyncLLMEngine
        engine_args = AsyncEngineArgs(
            model="./t3-model" if variant == "english" else "./t3-model-multilingual",
            task="generate",
            tokenizer="EnTokenizer" if variant == "english" else "MtlTokenizer",
            tokenizer_mode="custom",
            gpu_memory_utilization=vllm_memory_percent,
            enforce_eager=True,
            max_model_len=max_model_len,
        )

        t3_engine = AsyncLLMEngine.from_engine_args(engine_args)

        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve = ve.to(device=target_device).eval()

        s3gen = S3Gen(use_fp16=False)
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
        s3gen = s3gen.to(device=target_device).eval()

        default_conds = Conditionals.load(ckpt_dir / "conds.pt")
        default_conds.to(device=target_device)

        return cls(
            target_device=target_device,
            max_model_len=max_model_len,
            t3_engine=t3_engine,
            t3_config=t3_config,
            t3_cond_enc=t3_enc,
            t3_speech_emb=t3_speech_emb,
            t3_speech_pos_emb=t3_speech_pos_emb,
            s3gen=s3gen,
            ve=ve,
            default_conds=default_conds,
            variant=variant,
        )

    async def stream_audio_chunks(
        self,
        prompt: str,
        audio_prompt_path: Optional[str] = None,
        language_id: str = 'en',
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        chunk_size_samples: int = 24000,
        request_id: str = "tts-request",
        **generation_kwargs
    ) -> AsyncGenerator[torch.Tensor, None]:
        """
        Async generator that streams audio chunks as tokens are generated.

        This uses vLLM's async streaming capabilities for lower latency.
        """
        import librosa
        from chatterbox_vllm.models.s3gen import S3GEN_SR

        # Get audio conditionals
        if audio_prompt_path is None:
            s3gen_ref_dict = self.default_conds.gen
            t3_cond_prompt_tokens = self.default_conds.t3.cond_prompt_speech_tokens
            ve_embed = self.default_conds.t3.speaker_emb
        else:
            s3gen_ref_wav, _sr = librosa.load(audio_prompt_path, sr=S3GEN_SR)
            ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=24000)
            s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
            s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR)

            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=self.t3_config.speech_cond_prompt_len)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens)

            ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=24000))
            ve_embed = ve_embed.mean(axis=0, keepdim=True)

        cond_prompt_speech_emb = self.t3_speech_emb(t3_cond_prompt_tokens)[0] + self.t3_speech_pos_emb(t3_cond_prompt_tokens)

        cond_emb = self.t3_cond_enc(T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            cond_prompt_speech_emb=cond_prompt_speech_emb,
            emotion_adv=0.5 * torch.ones(1, 1)
        ).to(device=self.target_device)).to(device="cpu")

        # Update exaggeration
        if exaggeration != 0.5:
            cond_emb[-1] = self.t3_cond_enc.emotion_adv_fc(
                (exaggeration * torch.ones(1, 1)).to(self.target_device)
            ).to('cpu')

        # Normalize and tokenize text
        normalized_prompt = "[START]" + punc_norm(prompt) + "[STOP]"
        if self.variant == "multilingual":
            normalized_prompt = f"<{language_id.lower()}>{normalized_prompt}"

        # Create sampling params
        sampling_params = SamplingParams(
            temperature=temperature,
            stop_token_ids=[self.t3_config.stop_speech_token + SPEECH_TOKEN_OFFSET],
            max_tokens=min(generation_kwargs.get('max_tokens', 1000), self.max_model_len),
            top_p=generation_kwargs.get('top_p', 1.0),
            repetition_penalty=generation_kwargs.get('repetition_penalty', 2.0),
        )

        # Generate using async engine
        from vllm import RequestOutput

        # Accumulate tokens and stream audio
        accumulated_tokens = []
        diffusion_steps = generation_kwargs.get('diffusion_steps', 10)

        result_generator = self.t3_engine.generate(
            prompt=None,  # We'll provide the prompt via multi_modal_data
            sampling_params=sampling_params,
            request_id=request_id,
            prompt_token_ids=None,  # Will be set by the engine
            multi_modal_data={
                "conditionals": [cond_emb],
                "prompt": normalized_prompt,
            },
        )

        async for request_output in result_generator:
            if request_output.outputs:
                output = request_output.outputs[0]
                new_tokens = output.token_ids[len(accumulated_tokens):]
                accumulated_tokens.extend(new_tokens)

                # Process new tokens into audio chunks
                if new_tokens:
                    speech_tokens = torch.tensor(
                        [token - SPEECH_TOKEN_OFFSET for token in accumulated_tokens],
                        device="cuda"
                    )
                    speech_tokens = drop_invalid_tokens(speech_tokens)
                    speech_tokens = speech_tokens[speech_tokens < 6561]

                    # Generate audio from accumulated tokens
                    with torch.inference_mode():
                        wav, _ = self.s3gen.inference(
                            speech_tokens=speech_tokens,
                            ref_dict=s3gen_ref_dict,
                            n_timesteps=diffusion_steps,
                        )

                        # Yield only the new portion as a chunk
                        total_samples = wav.shape[1]
                        samples_per_token = chunk_size_samples // max(len(accumulated_tokens), 1)
                        new_samples = min(len(new_tokens) * samples_per_token * 10, total_samples)

                        if new_samples > 0:
                            chunk_start = max(0, total_samples - new_samples)
                            chunk = wav[:, chunk_start:total_samples]
                            if chunk.shape[1] > 0:
                                yield chunk

    async def shutdown(self):
        """Clean up resources."""
        await self.t3_engine.shutdown()
        del self.t3_engine
        torch.cuda.empty_cache()


async def example_save_streamed_audio():
    """Example: Save streamed audio to files."""
    print("="*60)
    print("Example 1: Save streamed audio to file")
    print("="*60)

    model = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=3,
        max_model_len=1000,
    )

    prompts = [
        "This is a demonstration of async streaming text to speech.",
        "Audio chunks are generated asynchronously as tokens arrive.",
    ]

    for i, prompt in enumerate(prompts):
        print(f"\nGenerating audio for prompt {i+1}: {prompt}")

        chunks = []
        async for chunk in model.stream_audio_chunks(
            prompt=prompt,
            chunk_size_samples=24000,
            temperature=0.8,
        ):
            chunks.append(chunk.cpu())
            print(f"  Received chunk: {chunk.shape[1]} samples")

        if chunks:
            full_audio = torch.cat(chunks, dim=1)
            output_path = f"test-async-streaming-{i}.mp3"
            ta.save(output_path, full_audio, model.sr)
            print(f"  Saved to {output_path}")

    await model.shutdown()


async def example_realtime_simulation():
    """Example: Simulate real-time playback timing."""
    print("\n" + "="*60)
    print("Example 2: Real-time playback simulation")
    print("="*60)

    model = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=3,
        max_model_len=1000,
    )

    prompt = "This example demonstrates the async streaming capabilities with timing information for each audio chunk."
    print(f"\nGenerating: {prompt}")

    total_samples = 0
    chunk_count = 0

    async for chunk in model.stream_audio_chunks(
        prompt=prompt,
        chunk_size_samples=12000,  # 0.5 second chunks
        temperature=0.8,
    ):
        chunk_count += 1
        chunk_samples = chunk.shape[1]
        total_samples += chunk_samples
        duration = chunk_samples / model.sr
        total_duration = total_samples / model.sr

        print(f"  Chunk {chunk_count}: {chunk_samples} samples ({duration:.2f}s) | Total: {total_duration:.2f}s")

    await model.shutdown()


async def example_concurrent_requests():
    """Example: Handle multiple concurrent streaming requests."""
    print("\n" + "="*60)
    print("Example 3: Concurrent streaming requests")
    print("="*60)

    model = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=3,
        max_model_len=1000,
    )

    async def process_request(request_id: str, prompt: str):
        print(f"[{request_id}] Starting: {prompt[:50]}...")
        chunk_count = 0
        async for chunk in model.stream_audio_chunks(
            prompt=prompt,
            request_id=request_id,
            chunk_size_samples=24000,
            temperature=0.8,
        ):
            chunk_count += 1
            print(f"[{request_id}] Chunk {chunk_count}: {chunk.shape[1]} samples")
        print(f"[{request_id}] Complete! Total chunks: {chunk_count}")
        return chunk_count

    # Run multiple requests concurrently
    prompts = [
        ("req-1", "First request with some text to synthesize."),
        ("req-2", "Second request running concurrently."),
        ("req-3", "Third request showing parallel processing."),
    ]

    results = await asyncio.gather(*[
        process_request(req_id, prompt) for req_id, prompt in prompts
    ])

    print(f"\nAll requests complete! Total chunks: {sum(results)}")
    await model.shutdown()


async def main():
    """Run all examples."""
    await example_save_streamed_audio()
    await example_realtime_simulation()
    await example_concurrent_requests()
    print("\nAsync streaming examples complete!")


if __name__ == "__main__":
    asyncio.run(main())
