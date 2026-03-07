#!/usr/bin/env python3
"""
Async streaming TTS implementation using vLLM's AsyncLLMEngine.

This provides true streaming with <1s first chunk latency by using AsyncLLMEngine
to stream speech tokens incrementally, then processing them through S3Gen.

Usage:
    import asyncio
    from chatterbox_vllm.tts_async import AsyncChatterboxTTS

    async def main():
        model = await AsyncChatterboxTTS.from_pretrained()
        async for audio_chunk, metrics in model.generate_stream("Hello world"):
            # Process audio chunk (1, T) tensor
            pass
        await model.shutdown()

    asyncio.run(main())
"""

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple, Any, AsyncGenerator

import librosa
import torch
import torch.nn.functional as F
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs

from chatterbox_vllm.models.s3tokenizer import S3_SR, drop_invalid_tokens
from chatterbox_vllm.models.s3gen import S3GEN_SR, S3Gen
from chatterbox_vllm.models.voice_encoder import VoiceEncoder
from chatterbox_vllm.models.t3 import SPEECH_TOKEN_OFFSET
from chatterbox_vllm.models.t3.modules.cond_enc import T3Cond, T3CondEnc
from chatterbox_vllm.models.t3.modules.learned_pos_emb import LearnedPositionEmbeddings
from chatterbox_vllm.text_utils import punc_norm, SUPPORTED_LANGUAGES
from chatterbox_vllm.tts import Conditionals, StreamingMetrics
from chatterbox_vllm.s3gen_stream_pool import S3GenStreamPool

REPO_ID = "ResembleAI/chatterbox"


class AsyncChatterboxTTS:
    """
    Async streaming TTS using vLLM's AsyncLLMEngine.

    This class provides true streaming with <1s first chunk latency by:
    1. Using AsyncLLMEngine to stream speech tokens incrementally
    2. Processing token chunks through S3Gen as they arrive
    3. Yielding audio chunks in real-time
    """

    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        engine: AsyncLLMEngine,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        default_conds: Conditionals,
        max_model_len: int,
        device: str,
        variant: str = "english",
        s3gen_stream_pool: Optional[S3GenStreamPool] = None,
        default_chunk_size: int = 25,
    ):
        self.engine = engine
        self.s3gen = s3gen
        self.ve = ve
        self.default_conds = default_conds
        self.max_model_len = max_model_len
        self.device = device
        self.variant = variant
        self.s3gen_stream_pool = s3gen_stream_pool
        self.default_chunk_size = default_chunk_size

    @classmethod
    async def from_pretrained(
        cls,
        model_path: Optional[str] = None,
        audio_prompt_path: Optional[str] = None,
        variant: str = "english",
        max_model_len: int = 2000,
        gpu_memory_utilization: float = 0.90,
        enforce_eager: bool = True,
        s3gen_use_fp16: bool = False,
        enable_stream_pool: bool = True,
        num_s3gen_streams: int = 12,
        default_chunk_size: int = 25,
        **kwargs
    ) -> "AsyncChatterboxTTS":
        """
        Load the async streaming TTS model.

        Args:
            model_path: Path to local model or use default HuggingFace model
            audio_prompt_path: Path to reference audio for voice cloning
            variant: Model variant ("english" or "multilingual")
            max_model_len: Maximum sequence length
            gpu_memory_utilization: GPU memory utilization (0.0-1.0)
            enforce_eager: Disable CUDA graphs (useful for debugging)
            s3gen_use_fp16: Use FP16 for S3Gen (faster, slight quality loss)
            enable_stream_pool: Enable CUDA stream pool for concurrent S3Gen inference
            num_s3gen_streams: Number of CUDA streams in the pool
            default_chunk_size: Default speech tokens per audio chunk
            **kwargs: Additional arguments for AsyncEngineArgs
        """
        if model_path is None:
            model_path = "./t3-model"

        # Ensure model exists
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        device = "cuda:0"
        target_device = device

        # Find the checkpoint directory containing all weight files
        # The model_path typically has symlinks to HuggingFace cache
        # We need to find where the actual weight files (ve, s3gen, conds) are located
        ckpt_dir = model_path

        # Check if required files exist in model_path
        required_files = ["ve.safetensors", "s3gen.safetensors", "conds.pt"]
        missing_files = [f for f in required_files if not (ckpt_dir / f).exists()]

        if missing_files:
            # Files not in model_path, search in HuggingFace cache
            # The symlink in model_path points to the HF cache
            # Get the actual snapshot directory by reading the symlink
            model_safetensors = model_path / "model.safetensors"
            if model_safetensors.is_symlink():
                # Read symlink to find the snapshot directory
                link_target = os.readlink(str(model_safetensors))
                # link_target is like: ../../blobs/[hash] or an absolute path
                if link_target.startswith("../../"):
                    # This is the standard HF cache structure
                    # Go up from model_path to the cache root
                    # model_path = ./t3-model/model.safetensors
                    # link = ../../blobs/hash
                    # We need to find the snapshot directory
                    # The snapshot dir should be a sibling of the blobs dir
                    # Navigate: model_path -> ../ -> ../ -> models--... -> snapshots -> [hash]
                    cache_root = model_path.resolve().parent.parent.parent.parent
                    if "models--ResembleAI--chatterbox" in str(cache_root):
                        # We're in the right area, find snapshots
                        snapshots_dir = cache_root / "snapshots"
                        if snapshots_dir.exists():
                            snapshots = sorted(snapshots_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
                            for snapshot in snapshots:
                                if all((snapshot / f).exists() for f in required_files):
                                    ckpt_dir = snapshot
                                    print(f"Using checkpoint directory: {ckpt_dir}")
                                    break
                elif Path(link_target).exists():
                    # Absolute path or relative that exists, try to get parent
                    target_path = Path(link_target).resolve()
                    # The snapshot dir should be the parent of the target's parent if it's in blobs/
                    if "blobs" in str(target_path):
                        # Go up to snapshots directory
                        ckpt_dir = target_path.parent.parent.parent
                    else:
                        ckpt_dir = target_path.parent

            # Also try the shared cache location
            if not all((ckpt_dir / f).exists() for f in required_files):
                shared_cache = Path("/mnt/data/shared/hf/hub")
                if shared_cache.exists():
                    model_cache_dir = shared_cache / "models--ResembleAI--chatterbox"
                    if model_cache_dir.exists():
                        snapshots_dir = model_cache_dir / "snapshots"
                        if snapshots_dir.exists():
                            snapshots = sorted(snapshots_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
                            for snapshot in snapshots:
                                if all((snapshot / f).exists() for f in required_files):
                                    ckpt_dir = snapshot
                                    print(f"Using checkpoint directory: {ckpt_dir}")
                                    break

            # Final check
            if not all((ckpt_dir / f).exists() for f in required_files):
                raise FileNotFoundError(
                    f"Could not find all required weight files. Missing: {missing_files}\n"
                    f"Searched in: {ckpt_dir}\n"
                    f"Please ensure model files are downloaded."
                )

        # Load T3 components for conditional encoding
        print("Loading T3 conditional encoder components...")
        from safetensors.torch import load_file
        from chatterbox_vllm.models.t3.modules.t3_config import T3Config

        t3_config = T3Config()

        # Load T3 weights for conditional encoding
        # The model directory uses model.safetensors as a symlink to the actual weights
        t3_weights_path = model_path / "model.safetensors"
        if not t3_weights_path.exists():
            # Fallback: try direct filename (for multilingual variant)
            alt_name = "t3_mtl23ls_v2.safetensors" if variant == "multilingual" else "t3_cfg.safetensors"
            t3_weights_path = model_path / alt_name
            if not t3_weights_path.exists():
                raise FileNotFoundError(f"T3 weights not found at {model_path}/model.safetensors or {alt_name}")

        t3_weights = load_file(t3_weights_path)

        # Load T3 conditional encoder
        t3_cond_enc = T3CondEnc(t3_config)
        t3_cond_enc.load_state_dict({
            k.replace('cond_enc.', ''): v
            for k, v in t3_weights.items()
            if k.startswith('cond_enc.')
        })
        t3_cond_enc = t3_cond_enc.to(device=target_device).eval()

        # Load speech embedding
        t3_speech_emb = torch.nn.Embedding(
            t3_config.speech_tokens_dict_size,
            t3_config.n_channels
        )
        t3_speech_emb.load_state_dict({
            k.replace('speech_emb.', ''): v
            for k, v in t3_weights.items()
            if k.startswith('speech_emb.')
        })
        t3_speech_emb = t3_speech_emb.to(device=target_device).eval()

        # Load speech position embedding
        t3_speech_pos_emb = LearnedPositionEmbeddings(
            t3_config.max_speech_tokens + 2 + 2,
            t3_config.n_channels
        )
        t3_speech_pos_emb.load_state_dict({
            k.replace('speech_pos_emb.', ''): v
            for k, v in t3_weights.items()
            if k.startswith('speech_pos_emb.')
        })
        t3_speech_pos_emb = t3_speech_pos_emb.to(device=target_device).eval()

        # Initialize AsyncLLMEngine
        print("Initializing AsyncLLMEngine...")
        engine_args = AsyncEngineArgs(
            model=str(model_path),
            tokenizer="EnTokenizer" if variant == "english" else "MtlTokenizer",
            tokenizer_mode="custom",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            disable_log_stats=False,
            tensor_parallel_size=1,
            **kwargs,
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("AsyncLLMEngine ready!")

        # Load VoiceEncoder
        print("Loading VoiceEncoder...")
        ve = VoiceEncoder()
        ve_weights_path = ckpt_dir / "ve.safetensors"
        if not ve_weights_path.exists():
            raise FileNotFoundError(f"VoiceEncoder weights not found at {ve_weights_path}")
        ve.load_state_dict(load_file(ve_weights_path))
        ve = ve.to(device=target_device).eval()

        # Load S3Gen
        print("Loading S3Gen...")
        s3gen = S3Gen(use_fp16=s3gen_use_fp16)
        s3gen_weights_path = ckpt_dir / "s3gen.safetensors"
        if not s3gen_weights_path.exists():
            raise FileNotFoundError(f"S3Gen weights not found at {s3gen_weights_path}")
        s3gen.load_state_dict(load_file(s3gen_weights_path), strict=False)
        s3gen = s3gen.to(device=target_device).eval()

        # Load default conditionals
        print("Loading default conditionals...")
        conds_path = ckpt_dir / "conds.pt"
        if not conds_path.exists():
            raise FileNotFoundError(f"Default conditionals not found at {conds_path}")
        default_conds = Conditionals.load(conds_path)
        default_conds.to(device=target_device)

        # Create stream pool if enabled
        s3gen_stream_pool = None
        if enable_stream_pool:
            print(f"Creating S3Gen stream pool with {num_s3gen_streams} streams...")
            s3gen_stream_pool = S3GenStreamPool(
                s3gen_model=s3gen,
                num_streams=num_s3gen_streams,
                device=target_device,
            )
            await s3gen_stream_pool.initialize()
            print("✓ Stream pool initialized!")

        # Create instance
        instance = cls(
            engine=engine,
            s3gen=s3gen,
            ve=ve,
            default_conds=default_conds,
            max_model_len=max_model_len,
            device=target_device,
            variant=variant,
            s3gen_stream_pool=s3gen_stream_pool,
            default_chunk_size=default_chunk_size,
        )

        # Store additional components needed for conditioning
        instance.t3_config = t3_config
        instance.t3_cond_enc = t3_cond_enc
        instance.t3_speech_emb = t3_speech_emb
        instance.t3_speech_pos_emb = t3_speech_pos_emb

        print("✓ Model loaded successfully!")

        # Warmup
        print("\nWarming up model (for steady-state performance)...")
        warmup_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "This is a warmup test.",
            "Hello world, warmup complete.",
        ]
        for i, text in enumerate(warmup_texts, 1):
            async for _, _ in instance.generate_stream(
                text=text,
                max_tokens=200,
                chunk_size=25,
                print_metrics=False,
            ):
                break  # Only need first chunk
            print(f"  Warmup {i}/3 complete")

        print("✓ Model warmed up and ready!\n")

        return instance

    def get_supported_languages(self) -> dict[str, str]:
        """Return dictionary of supported language codes and names."""
        if self.variant == "multilingual":
            return SUPPORTED_LANGUAGES.copy()
        else:
            return {"en": "English"}

    def get_audio_conditionals(self, wav_fpath: Optional[str] = None) -> Tuple[dict[str, Any], torch.Tensor]:
        """
        Get audio conditionals for T3 and S3Gen.

        Args:
            wav_fpath: Optional path to reference audio for voice cloning

        Returns:
            Tuple of (s3gen_ref_dict, t3_cond_emb)
        """
        if wav_fpath is None:
            s3gen_ref_dict = self.default_conds.gen
            t3_cond_prompt_tokens = self.default_conds.t3.cond_prompt_speech_tokens
            ve_embed = self.default_conds.t3.speaker_emb
        else:
            # Load reference wav
            s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
            ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

            s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
            s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR)

            # Speech cond prompt tokens
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward(
                [ref_16k_wav[:self.ENC_COND_LEN]],
                max_len=self.t3_config.speech_cond_prompt_len
            )
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens)

            # Voice-encoder speaker embedding
            ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
            ve_embed = ve_embed.mean(axis=0, keepdim=True)

        cond_prompt_speech_emb = self.t3_speech_emb(t3_cond_prompt_tokens)[0] + self.t3_speech_pos_emb(t3_cond_prompt_tokens)

        cond_emb = self.t3_cond_enc(T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            cond_prompt_speech_emb=cond_prompt_speech_emb,
            emotion_adv=0.5 * torch.ones(1, 1)
        ).to(device=self.device)).to(device="cpu")  # Conditionals need to be given to VLLM in CPU

        return s3gen_ref_dict, cond_emb

    def update_exaggeration(self, cond_emb: torch.Tensor, exaggeration: float) -> torch.Tensor:
        """Update exaggeration in conditional embedding."""
        if exaggeration == 0.5:
            return cond_emb

        new_cond_emb = cond_emb.clone()
        new_cond_emb[-1] = self.t3_cond_enc.emotion_adv_fc(
            (exaggeration * torch.ones(1, 1)).to(self.device)
        ).to('cpu')
        return new_cond_emb

    async def generate_stream(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        language_id: int = 0,
        exaggeration: float = 0.0,
        temperature: float = 0.8,
        max_tokens: int = 500,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        chunk_size: Optional[int] = None,
        context_window: int = 50,
        fade_duration: float = 0.02,
        diffusion_steps: int = 10,
        print_metrics: bool = False,
    ) -> AsyncGenerator[Tuple[torch.Tensor, StreamingMetrics], None]:
        """
        Generate audio asynchronously with true streaming.

        This method uses AsyncLLMEngine to stream speech tokens incrementally,
        achieving <1s first chunk latency.

        Args:
            text: Input text to synthesize
            audio_prompt_path: Optional reference audio for voice cloning
            language_id: Language code (multilingual variant only)
            exaggeration: Emotion exaggeration (0.0 to 1.0)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty
            chunk_size: Speech tokens per audio chunk (uses default_chunk_size if None)
            context_window: Context tokens for continuity between chunks
            fade_duration: Fade-in duration in seconds between chunks
            diffusion_steps: S3Gen diffusion steps
            print_metrics: Whether to print progress metrics

        Yields:
            (audio_chunk, metrics) tuples where:
            - audio_chunk: (1, T) tensor of audio samples at 24kHz
            - metrics: StreamingMetrics with current performance data
        """
        start_time = time.time()
        metrics = StreamingMetrics()

        # Use default chunk size if not specified
        if chunk_size is None:
            chunk_size = self.default_chunk_size

        # Track setup timing
        setup_start = time.time()

        # Get conditionals
        s3gen_ref, cond_emb = self.get_audio_conditionals(audio_prompt_path)
        cond_emb = self.update_exaggeration(cond_emb, exaggeration)
        conditionals_time = time.time() - setup_start

        # Validate language
        if self.variant == "multilingual":
            if language_id not in self.get_supported_languages():
                raise ValueError(f"Unsupported language '{language_id}'")
            lang_code = SUPPORTED_LANGUAGES[language_id]
        else:
            lang_code = None

        # Normalize and prepare text
        text_prep_start = time.time()
        text_normalized = "[START]" + punc_norm(text) + "[STOP]"
        if lang_code:
            text_normalized = f"<|{lang_code}|> {text_normalized}"
        text_prep_time = time.time() - text_prep_start

        # Setup sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Track all tokens generated so far
        all_tokens = []
        last_processed_count = 0  # Track how many tokens we've processed
        first_token_time = None
        t3_start_time = time.time()
        last_yield_time = t3_start_time

        # Initialize granular timing variables
        token_conversion_time = 0.0
        context_prep_time = 0.0
        chunk_prep_time = 0.0

        request_id = f"tts-request-{time.time()}"

        if print_metrics:
            print(f"[DEBUG] Starting generation at {t3_start_time:.3f}")

        # Stream tokens from AsyncLLMEngine
        # NOTE: Pass prompt as string to enable incremental token streaming
        # Passing multi_modal_data causes vLLM to batch all tokens at once
        async for request_output in self.engine.generate(
            prompt=text_normalized,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            current_time = time.time()
            time_since_last_yield = current_time - last_yield_time
            last_yield_time = current_time

            # Track first token time
            if first_token_time is None and request_output.outputs:
                first_token_time = current_time
                metrics.t3_first_token_time = first_token_time - t3_start_time
                if print_metrics:
                    print(f"[DEBUG] First token received at {first_token_time:.3f} ({metrics.t3_first_token_time*1000:.2f}ms)")

            # Collect NEW tokens only
            if request_output.outputs:
                output = request_output.outputs[0]
                current_tokens = list(output.token_ids)
                new_token_count = len(current_tokens) - last_processed_count

                if print_metrics and new_token_count > 0:
                    print(f"[DEBUG] Yield: +{new_token_count} tokens (total: {len(current_tokens)}) at {time_since_last_yield*1000:.2f}ms")

                # Only process if we have new tokens
                if new_token_count > 0:
                    all_tokens = current_tokens

                    # Process chunk if we have enough NEW tokens
                    # Process as many complete chunks as possible from new tokens
                    tokens_to_process = new_token_count
                    offset = 0

                    while tokens_to_process >= chunk_size or (tokens_to_process > 0 and request_output.finished):
                        # Calculate chunk size (use chunk_size, or remaining tokens if finished)
                        current_chunk_size = min(chunk_size, tokens_to_process) if request_output.finished else chunk_size

                        if tokens_to_process < current_chunk_size:
                            break

                        # Extract NEW chunk (tokens we haven't processed yet)
                        chunk_prep_start = time.time()
                        chunk_start_idx = last_processed_count + offset
                        chunk_end_idx = chunk_start_idx + current_chunk_size
                        token_chunk = all_tokens[chunk_start_idx:chunk_end_idx]

                        # Convert to speech tokens (subtract offset)
                        token_conversion_start = time.time()
                        token_chunk_tensor = torch.tensor(
                            [token - SPEECH_TOKEN_OFFSET for token in token_chunk],
                            device="cuda"
                        )
                        token_chunk_tensor = drop_invalid_tokens(token_chunk_tensor.unsqueeze(0))
                        token_chunk_tensor = token_chunk_tensor[token_chunk_tensor < 6561]
                        token_conversion_time = time.time() - token_conversion_start

                        if token_chunk_tensor.numel() == 0:
                            offset += current_chunk_size
                            tokens_to_process -= current_chunk_size
                            continue

                        # Get context for continuity (convert all PREVIOUS tokens to speech tokens)
                        context_prep_start = time.time()
                        if chunk_start_idx > 0:
                            context_tokens_list = [
                                t - SPEECH_TOKEN_OFFSET for t in all_tokens[:chunk_start_idx]
                            ]
                            context_tokens_tensor = torch.tensor(context_tokens_list, device="cuda")
                            context_tokens_tensor = drop_invalid_tokens(context_tokens_tensor)
                            context_tokens_tensor = context_tokens_tensor[context_tokens_tensor < 6561]
                        else:
                            context_tokens_tensor = None
                        context_prep_time = time.time() - context_prep_start
                        chunk_prep_time = time.time() - chunk_prep_start

                        # Process tokens to audio
                        s3gen_start = time.time()
                        # Use stream pool if available, otherwise fall back to async method
                        if self.s3gen_stream_pool:
                            audio_chunk = await self.s3gen_stream_pool.process_async(
                                token_chunk=token_chunk_tensor,
                                context_tokens=context_tokens_tensor,
                                s3gen_ref=s3gen_ref,
                                context_window=context_window,
                                fade_duration=fade_duration,
                                diffusion_steps=diffusion_steps,
                            )
                        else:
                            audio_chunk = await self._process_token_chunk_async(
                                token_chunk=token_chunk_tensor,
                                context_tokens=context_tokens_tensor,
                                s3gen_ref=s3gen_ref,
                                context_window=context_window,
                                fade_duration=fade_duration,
                                diffusion_steps=diffusion_steps,
                            )
                        s3gen_time = time.time() - s3gen_start

                        # Update metrics
                        if metrics.chunk_count == 0:
                            metrics.latency_to_first_chunk = current_time - start_time
                            if first_token_time is not None:
                                metrics.s3gen_first_chunk_time = current_time - first_token_time
                                metrics.t3_token_generation_time = first_token_time - t3_start_time
                            metrics.first_s3gen_inference_time = s3gen_time
                            # Store granular first-chunk timing
                            metrics.conditionals_prep_ms = conditionals_time * 1000
                            metrics.text_prep_ms = text_prep_time * 1000
                            metrics.token_conversion_ms = token_conversion_time * 1000
                            metrics.context_prep_ms = context_prep_time * 1000
                            metrics.chunk_prep_overhead_ms = max(0, (chunk_prep_time - token_conversion_time - context_prep_time) * 1000)
                            if print_metrics:
                                print(f"[DEBUG] First chunk breakdown:")
                                print(f"  Conditionals: {conditionals_time*1000:.2f}ms")
                                print(f"  Text prep: {text_prep_time*1000:.2f}ms")
                                print(f"  Token conversion: {token_conversion_time*1000:.2f}ms")
                                print(f"  Context prep: {context_prep_time*1000:.2f}ms")
                                print(f"  Chunk prep overhead: {max(0, (chunk_prep_time - token_conversion_time - context_prep_time))*1000:.2f}ms")
                                print(f"  S3Gen inference: {s3gen_time*1000:.2f}ms")

                        metrics.chunk_count += 1
                        metrics.last_chunk_time = current_time - start_time

                        if audio_chunk is not None:
                            metrics.total_audio_duration += audio_chunk.shape[-1] / S3GEN_SR
                            yield audio_chunk, metrics

                        # Update processed count
                        offset += current_chunk_size
                        tokens_to_process -= current_chunk_size

                    # Update the count of tokens we've processed
                    last_processed_count += offset

            # Check if generation is complete
            if request_output.finished:
                break

        # Process any remaining tokens that haven't been processed
        remaining_count = len(all_tokens) - last_processed_count
        if remaining_count > 0:
            # Convert remaining tokens to speech tokens
            remaining_tokens = [
                t - SPEECH_TOKEN_OFFSET for t in all_tokens[last_processed_count:]
            ]
            remaining_tokens_tensor = torch.tensor(remaining_tokens, device="cuda")
            remaining_tokens_tensor = drop_invalid_tokens(remaining_tokens_tensor)
            remaining_tokens_tensor = remaining_tokens_tensor[remaining_tokens_tensor < 6561]

            if remaining_tokens_tensor.numel() > 0:
                # Get context for continuity
                if last_processed_count > 0:
                    context_tokens_list = [
                        t - SPEECH_TOKEN_OFFSET for t in all_tokens[:last_processed_count]
                    ]
                    context_tokens_tensor = torch.tensor(context_tokens_list, device="cuda")
                    context_tokens_tensor = drop_invalid_tokens(context_tokens_tensor)
                    context_tokens_tensor = context_tokens_tensor[context_tokens_tensor < 6561]
                else:
                    context_tokens_tensor = None

                # Use stream pool if available, otherwise fall back to async method
                if self.s3gen_stream_pool:
                    audio_chunk = await self.s3gen_stream_pool.process_async(
                        token_chunk=remaining_tokens_tensor.unsqueeze(0),
                        context_tokens=context_tokens_tensor,
                        s3gen_ref=s3gen_ref,
                        context_window=context_window,
                        fade_duration=fade_duration,
                        diffusion_steps=diffusion_steps,
                    )
                else:
                    audio_chunk = await self._process_token_chunk_async(
                        token_chunk=remaining_tokens_tensor.unsqueeze(0),
                        context_tokens=context_tokens_tensor,
                        s3gen_ref=s3gen_ref,
                        context_window=context_window,
                        fade_duration=fade_duration,
                        diffusion_steps=diffusion_steps,
                    )
                if audio_chunk is not None:
                    metrics.chunk_count += 1
                    metrics.total_audio_duration += audio_chunk.shape[-1] / S3GEN_SR
                    yield audio_chunk, metrics

        # Final metrics
        metrics.total_generation_time = time.time() - start_time
        metrics.t3_token_generation_time = first_token_time - t3_start_time if first_token_time else 0
        if metrics.total_audio_duration > 0:
            metrics.rtf = metrics.total_generation_time / metrics.total_audio_duration

        if print_metrics:
            print(f"\n[PROFILE] Async Streaming Complete:")
            print(f"  - First chunk latency: {metrics.latency_to_first_chunk*1000:.2f}ms")
            print(f"  - Total time: {metrics.total_generation_time:.2f}s")
            print(f"  - Audio duration: {metrics.total_audio_duration:.2f}s")
            print(f"  - RTF: {metrics.rtf:.3f}")
            print(f"  - Chunks: {metrics.chunk_count}")

    async def _process_token_chunk_async(
        self,
        token_chunk: torch.Tensor,
        context_tokens: Optional[torch.Tensor],
        s3gen_ref: dict[str, Any],
        context_window: int = 50,
        fade_duration: float = 0.02,
        diffusion_steps: int = 10,
    ) -> Optional[torch.Tensor]:
        """
        Process a chunk of speech tokens to audio asynchronously.

        This runs S3Gen inference in a thread pool to avoid blocking the event loop.

        Args:
            token_chunk: New tokens to process (1, T_new)
            context_tokens: Optional context tokens for continuity
            s3gen_ref: S3Gen reference dictionary
            context_window: Context tokens to include for continuity
            fade_duration: Fade-in duration in seconds
            diffusion_steps: S3Gen diffusion steps

        Returns:
            Audio chunk tensor or None if no valid tokens
        """
        loop = asyncio.get_event_loop()

        def _process_sync():
            """Synchronous processing to run in thread pool."""
            # Use local variables to avoid scope issues
            ctx_tokens = context_tokens  # Capture from outer scope

            # Build tokens with context window
            if ctx_tokens is not None and len(ctx_tokens) > 0:
                # Ensure ctx_tokens is 1D for slicing
                if ctx_tokens.dim() > 1:
                    ctx_tokens = ctx_tokens.squeeze(0)
                ctx_tokens_window = (
                    ctx_tokens[-context_window:]
                    if len(ctx_tokens) > context_window
                    else ctx_tokens
                )
                # Ensure token_chunk is 1D for concatenation
                token_chunk_1d = token_chunk.squeeze(0) if token_chunk.dim() > 1 else token_chunk
                tokens_to_process = torch.cat([ctx_tokens_window, token_chunk_1d], dim=-1).unsqueeze(0)
                context_length = len(ctx_tokens_window)
            else:
                tokens_to_process = token_chunk
                context_length = 0

            # Clean tokens
            clean_tokens = drop_invalid_tokens(tokens_to_process)
            if len(clean_tokens) == 0:
                return None

            # Run S3Gen inference
            wav, _ = self.s3gen.inference(
                speech_tokens=clean_tokens,
                ref_dict=s3gen_ref,
                n_timesteps=diffusion_steps,
            )

            # Crop context if present
            if context_length > 0:
                samples_per_token = wav.shape[-1] / len(clean_tokens)
                skip_samples = int(context_length * samples_per_token)
                audio_chunk = wav[:, skip_samples:]
            else:
                audio_chunk = wav

            if audio_chunk.shape[-1] == 0:
                return None

            # Apply fade-in (clone to avoid inplace update in inference_mode)
            fade_samples = int(fade_duration * S3GEN_SR)
            if fade_samples > 0 and fade_samples < audio_chunk.shape[-1]:
                fade_in = torch.linspace(0.0, 1.0, fade_samples, device=audio_chunk.device)
                audio_chunk = audio_chunk.clone()
                audio_chunk[:, :fade_samples] *= fade_in

            return audio_chunk

        # Run in thread pool to avoid blocking
        return await loop.run_in_executor(None, _process_sync)

    async def shutdown(self):
        """Cleanup resources."""
        import gc
        # Shutdown stream pool if present
        if self.s3gen_stream_pool is not None:
            print("Shutting down stream pool...")
            await self.s3gen_stream_pool.shutdown()
        # Delete the engine to trigger cleanup
        if hasattr(self, 'engine'):
            del self.engine
        # Clear CUDA cache
        torch.cuda.empty_cache()
        # Force garbage collection
        gc.collect()


# TODO: Refactor ChatterboxTTS to extract common loading logic
# that can be shared between sync and async versions
