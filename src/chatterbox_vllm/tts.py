from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple, Any, Generator
import time

from vllm import LLM, SamplingParams
from functools import lru_cache

import librosa
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from chatterbox_vllm.models.t3.modules.t3_config import T3Config

from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.voice_encoder import VoiceEncoder
from .models.t3 import SPEECH_TOKEN_OFFSET
from .models.t3.modules.cond_enc import T3Cond, T3CondEnc
from .models.t3.modules.learned_pos_emb import LearnedPositionEmbeddings
from .text_utils import punc_norm, SUPPORTED_LANGUAGES

REPO_ID = "ResembleAI/chatterbox"


@dataclass
class StreamingMetrics:
    """Metrics for streaming generation"""
    latency_to_first_chunk: float = 0.0
    rtf: float = 0.0  # Real-time factor
    total_generation_time: float = 0.0
    total_audio_duration: float = 0.0
    chunk_count: int = 0


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    @classmethod
    def load(cls, fpath):
        kwargs = torch.load(fpath, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(self, target_device: str, max_model_len: int,
                 t3: LLM, t3_config: T3Config, t3_cond_enc: T3CondEnc, 
                 t3_speech_emb: torch.nn.Embedding, t3_speech_pos_emb: LearnedPositionEmbeddings,
                 s3gen: S3Gen, ve: VoiceEncoder, default_conds: Conditionals,
                 variant: str = "english"):
        self.target_device = target_device
        self.max_model_len = max_model_len
        self.t3 = t3
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
        """Sample rate of synthesized audio"""
        return S3GEN_SR

    @classmethod
    def from_local(cls, ckpt_dir: str | Path, target_device: str = "cuda", 
                   max_model_len: int = 1000, compile: bool = False,
                   max_batch_size: int = 10,
                   variant: str = "english",

                   # Original Chatterbox defaults this to False. I don't see a substantial performance difference when running with FP16.
                   s3gen_use_fp16: bool = False,
                   **kwargs) -> 'ChatterboxTTS':
        ckpt_dir = Path(ckpt_dir)

        t3_config = T3Config()

        # Load *just* the necessary weights to perform inference with T3CondEnc
        t3_weights = load_file(ckpt_dir / ("t3_cfg.safetensors" if variant == "english" else "t3_mtl23ls_v2.safetensors"))

        t3_enc = T3CondEnc(t3_config)
        t3_enc.load_state_dict({ k.replace('cond_enc.', ''):v for k,v in t3_weights.items() if k.startswith('cond_enc.') })
        t3_enc = t3_enc.to(device=target_device).eval()

        t3_speech_emb = torch.nn.Embedding(t3_config.speech_tokens_dict_size, t3_config.n_channels)
        t3_speech_emb.load_state_dict({ k.replace('speech_emb.', ''):v for k,v in t3_weights.items() if k.startswith('speech_emb.') })
        t3_speech_emb = t3_speech_emb.to(device=target_device).eval()

        t3_speech_pos_emb = LearnedPositionEmbeddings(t3_config.max_speech_tokens + 2 + 2, t3_config.n_channels)
        t3_speech_pos_emb.load_state_dict({ k.replace('speech_pos_emb.', ''):v for k,v in t3_weights.items() if k.startswith('speech_pos_emb.') })
        t3_speech_pos_emb = t3_speech_pos_emb.to(device=target_device).eval()

        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        unused_gpu_memory = total_gpu_memory - torch.cuda.memory_allocated()
        
        # Heuristic: rough calculation for what percentage of GPU memory to give to vLLM.
        # Tune this until the 'Maximum concurrency for ___ tokens per request: ___x' is just over 1.
        # This rough heuristic gives 1.55GB for the model weights plus 128KB per token.
        vllm_memory_needed = (1.55*1024*1024*1024) + (max_batch_size * max_model_len * 1024 * 128)
        vllm_memory_percent = vllm_memory_needed / unused_gpu_memory

        print(f"Giving vLLM {vllm_memory_percent * 100:.2f}% of GPU memory ({vllm_memory_needed / 1024**2:.2f} MB)")

        base_vllm_kwargs = {
            "model": "./t3-model" if variant == "english" else "./t3-model-multilingual",
            "task": "generate",
            "tokenizer": "EnTokenizer" if variant == "english" else "MtlTokenizer",
            "tokenizer_mode": "custom",
            "gpu_memory_utilization": vllm_memory_percent,
            "enforce_eager": not compile,
            "max_model_len": max_model_len,
        }

        t3 = LLM(**{**base_vllm_kwargs, **kwargs})

        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve = ve.to(device=target_device).eval()

        s3gen = S3Gen(use_fp16=s3gen_use_fp16)
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
        s3gen = s3gen.to(device=target_device).eval()

        default_conds = Conditionals.load(ckpt_dir / "conds.pt")
        default_conds.to(device=target_device)

        return cls(
            target_device=target_device, max_model_len=max_model_len,
            t3=t3, t3_config=t3_config, t3_cond_enc=t3_enc, t3_speech_emb=t3_speech_emb, t3_speech_pos_emb=t3_speech_pos_emb,
            s3gen=s3gen, ve=ve, default_conds=default_conds,
            variant=variant,
        )

    @classmethod
    def from_pretrained(cls,
                        repo_id: str = REPO_ID,
                        revision: str = "1b475dffa71fb191cb6d5901215eb6f55635a9b6",
                        *args, **kwargs) -> 'ChatterboxTTS':
        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=repo_id, filename=fpath, revision=revision)

        # Ensure the symlink in './t3-model/model.safetensors' points to t3_cfg_path
        t3_cfg_path = Path(local_path).parent / "t3_cfg.safetensors"
        model_safetensors_path = Path.cwd() / "t3-model" / "model.safetensors"
        model_safetensors_path.unlink(missing_ok=True)
        model_safetensors_path.symlink_to(t3_cfg_path)

        return cls.from_local(Path(local_path).parent, variant="english", *args, **kwargs)

    @classmethod
    def from_pretrained_multilingual(cls,
                                    repo_id: str = REPO_ID,
                                    revision: str = "05e904af2b5c7f8e482687a9d7336c5c824467d9",
                                    *args, **kwargs) -> 'ChatterboxTTS':
        for fpath in ["ve.safetensors", "t3_mtl23ls_v2.safetensors", "s3gen.safetensors", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"]:
            local_path = hf_hub_download(repo_id=repo_id, filename=fpath, revision=revision)

        # Ensure the symlink in './t3-model-multilingual/model.safetensors' points to t3_cfg_path
        t3_cfg_path = Path(local_path).parent / "t3_mtl23ls_v2.safetensors"
        model_safetensors_path = Path.cwd() / "t3-model-multilingual" / "model.safetensors"
        model_safetensors_path.unlink(missing_ok=True)
        model_safetensors_path.symlink_to(t3_cfg_path)

        return cls.from_local(Path(local_path).parent, variant="multilingual", *args, **kwargs)
    
    def get_supported_languages(self) -> dict[str, str]:
        """Return dictionary of supported language codes and names."""
        if self.variant == "multilingual":
            return SUPPORTED_LANGUAGES.copy()
        else:
            return { "en": "English" }

    @lru_cache(maxsize=10)
    def get_audio_conditionals(self, wav_fpath: Optional[str] = None) -> Tuple[dict[str, Any], torch.Tensor]:
        if wav_fpath is None:
            s3gen_ref_dict = self.default_conds.gen
            t3_cond_prompt_tokens = self.default_conds.t3.cond_prompt_speech_tokens
            ve_embed = self.default_conds.t3.speaker_emb
        else:
            ## Load reference wav
            s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
            ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

            s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
            s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR)

            # Speech cond prompt tokens
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=self.t3_config.speech_cond_prompt_len)
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
        ).to(device=self.target_device)).to(device="cpu")  # Conditionals need to be given to VLLM in CPU

        return s3gen_ref_dict, cond_emb

    def update_exaggeration(self, cond_emb: torch.Tensor, exaggeration: float) -> torch.Tensor:
        if exaggeration == 0.5:
            return cond_emb

        new_cond_emb = cond_emb.clone()
        new_cond_emb[-1] = self.t3_cond_enc.emotion_adv_fc(
            (exaggeration * torch.ones(1, 1)).to(self.target_device)
        ).to('cpu')
        return new_cond_emb

    def _process_token_chunk(
        self,
        token_chunk: torch.Tensor,
        all_tokens_so_far: torch.Tensor,
        s3gen_ref: dict[str, Any],
        context_window: int = 50,
        fade_duration: float = 0.02,
    ) -> Optional[torch.Tensor]:
        """
        Process a chunk of speech tokens through S3Gen with context window.

        Args:
            token_chunk: New tokens to process (1, T_new)
            all_tokens_so_far: All tokens generated so far (1, T_total)
            s3gen_ref: S3Gen reference dictionary
            context_window: Context tokens to include for continuity
            fade_duration: Fade-in duration in seconds

        Returns:
            Audio chunk tensor or None if no valid tokens
        """
        # Build tokens with context window
        if len(all_tokens_so_far) > 0:
            # Ensure all_tokens_so_far is 1D for slicing, then reshape
            if all_tokens_so_far.dim() > 1:
                all_tokens_so_far = all_tokens_so_far.squeeze(0)
            context_tokens = (
                all_tokens_so_far[-context_window:]
                if len(all_tokens_so_far) > context_window
                else all_tokens_so_far
            )
            # Ensure token_chunk is 1D for concatenation
            token_chunk_1d = token_chunk.squeeze(0) if token_chunk.dim() > 1 else token_chunk
            tokens_to_process = torch.cat([context_tokens, token_chunk_1d], dim=-1).unsqueeze(0)
            context_length = len(context_tokens)
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
            n_timesteps=10,  # Default diffusion steps
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

        # Apply fade-in
        fade_samples = int(fade_duration * S3GEN_SR)
        if fade_samples > 0 and fade_samples < audio_chunk.shape[-1]:
            fade_in = torch.linspace(0.0, 1.0, fade_samples, device=audio_chunk.device)
            audio_chunk[:, :fade_samples] *= fade_in

        return audio_chunk

    def generate(
        self,
        prompts: Union[str, list[str]],
        audio_prompt_path: Optional[str] = None,
        language_id: Optional[str] = 'en',
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        max_tokens=1000, # Capped at max_model_len

        # From original Chatterbox HF generation args
        top_p=0.8,
        repetition_penalty=2.0,

        # Supports anything in https://docs.vllm.ai/en/v0.9.2/api/vllm/index.html?h=samplingparams#vllm.SamplingParams
        *args, **kwargs,
    ) -> list[any]:
        s3gen_ref, cond_emb = self.get_audio_conditionals(audio_prompt_path)

        return self.generate_with_conds(
            prompts=prompts,
            s3gen_ref=s3gen_ref,
            cond_emb=cond_emb,
            temperature=temperature,
            language_id=language_id,
            exaggeration=exaggeration,
            max_tokens=max_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            *args, **kwargs
        )

    def generate_with_conds(
        self,
        prompts: Union[str, list[str]],
        s3gen_ref: dict[str, Any],
        cond_emb: torch.Tensor,
        language_id: Optional[str] = 'en',
        temperature: float = 0.8,
        exaggeration: float = 0.5,
        max_tokens=1000, # Capped at max_model_len

        # Number of diffusion steps to use for S3Gen
        # The original Chatterbox uses 10. 5 is often enough for good quality audio, though some quality loss can be detected.
        # This can be as low as 2 or 3 for faster generation, though the audio quality will degrade substantially.
        diffusion_steps: int = 10,

        # From original Chatterbox HF generation args
        top_p=1.0,
        min_p=0.05,
        repetition_penalty=2.0,

        # Supports anything in https://docs.vllm.ai/en/v0.9.2/api/vllm/index.html?h=samplingparams#vllm.SamplingParams
        *args, **kwargs,
    ) -> list[any]:
        if isinstance(prompts, str):
            prompts = [prompts]

        # Validate language_id
        if language_id and language_id.lower() not in self.get_supported_languages():
            supported_langs = ", ".join(self.get_supported_languages().keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )

        cond_emb = self.update_exaggeration(cond_emb, exaggeration)

        # Norm and tokenize text
        prompts = ["[START]" + punc_norm(p) + "[STOP]" for p in prompts]

        # For multilingual, prepend the language token
        if self.variant == "multilingual":
            # Use angle brackets to avoid conflicts with other start/stop tokens.
            # This will be parsed and replaced in the tokenizer.
            prompts = [f"<{language_id.lower()}>{p}" for p in prompts]

        with torch.inference_mode():
            start_time = time.time()
            batch_results = self.t3.generate(
                [
                    {
                        "prompt": text,
                        "multi_modal_data": {
                            "conditionals": [cond_emb],
                        },
                    }
                    for text in prompts
                ],
                sampling_params=SamplingParams(
                    temperature=temperature,

                    stop_token_ids=[self.t3_config.stop_speech_token + SPEECH_TOKEN_OFFSET],
                    max_tokens=min(max_tokens, self.max_model_len),
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,

                    *args, **kwargs,
                )
            )
            t3_gen_time = time.time() - start_time
            print(f"[T3] Speech Token Generation time: {t3_gen_time:.2f}s")

            # run torch gc
            torch.cuda.empty_cache()

            start_time = time.time()
            results = []
            for i, batch_result in enumerate(batch_results):
                for output in batch_result.outputs:
                    if i % 5 == 0:
                        print(f"[S3] Processing prompt {i} of {len(batch_results)}")

                    # Run gc every 10 prompts
                    if i % 10 == 0:
                        torch.cuda.empty_cache()

                    speech_tokens = torch.tensor([token - SPEECH_TOKEN_OFFSET for token in output.token_ids], device="cuda")
                    speech_tokens = drop_invalid_tokens(speech_tokens)
                    speech_tokens = speech_tokens[speech_tokens < 6561]

                    wav, _ = self.s3gen.inference(
                        speech_tokens=speech_tokens,
                        ref_dict=s3gen_ref,
                        n_timesteps=diffusion_steps,
                    )
                    results.append(wav.cpu())
            s3gen_gen_time = time.time() - start_time
            print(f"[S3Gen] Wavform Generation time: {s3gen_gen_time:.2f}s")

            return results

    def generate_stream(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        language_id: Optional[str] = 'en',
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        max_tokens: int = 1000,
        chunk_size: int = 25,  # Speech tokens per chunk
        context_window: int = 50,
        fade_duration: float = 0.02,
        print_metrics: bool = True,
        # S3Gen parameters
        diffusion_steps: int = 10,
        # Sampling parameters
        top_p: float = 1.0,
        repetition_penalty: float = 2.0,
        *args, **kwargs,
    ) -> Generator[Tuple[torch.Tensor, StreamingMetrics], None, None]:
        """
        Generate streaming audio from text, yielding chunks as they're synthesized.

        This method generates all speech tokens using vLLM (fast), then streams
        them through S3Gen incrementally for real-time audio playback.

        Args:
            text: Input text to synthesize
            audio_prompt_path: Optional path to reference audio for voice cloning
            language_id: Language code (multilingual variant only)
            exaggeration: Emotion exaggeration factor (0.0 to 1.0)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            chunk_size: Speech tokens per audio chunk
            context_window: Context tokens for audio continuity
            fade_duration: Fade-in duration between chunks (seconds)
            print_metrics: Whether to print timing metrics
            diffusion_steps: S3Gen diffusion steps (lower = faster, lower quality)
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty for sampling
            *args, **kwargs: Additional vLLM SamplingParams arguments

        Yields:
            Tuple of (audio_chunk, metrics) where:
            - audio_chunk: torch.Tensor of shape (1, T) audio samples
            - metrics: StreamingMetrics with timing information

        Example:
            >>> for audio_chunk, metrics in model.generate_stream("Hello world"):
            ...     # Play or save audio_chunk
            ...     print(f"Chunk {metrics.chunk_count}: {audio_chunk.shape}")
        """
        start_time = time.time()
        metrics = StreamingMetrics()

        # Get conditionals
        s3gen_ref, cond_emb = self.get_audio_conditionals(audio_prompt_path)
        cond_emb = self.update_exaggeration(cond_emb, exaggeration)

        # Validate language
        if language_id and language_id.lower() not in self.get_supported_languages():
            raise ValueError(f"Unsupported language '{language_id}'")

        # Normalize and tokenize text
        text_normalized = "[START]" + punc_norm(text) + "[STOP]"
        if self.variant == "multilingual":
            text_normalized = f"<{language_id.lower()}>{text_normalized}"

        with torch.inference_mode():
            # === Stage 1: Generate all speech tokens using vLLM ===
            t3_start = time.time()
            batch_results = self.t3.generate(
                [{
                    "prompt": text_normalized,
                    "multi_modal_data": {"conditionals": [cond_emb]},
                }],
                sampling_params=SamplingParams(
                    temperature=temperature,
                    stop_token_ids=[self.t3_config.stop_speech_token + SPEECH_TOKEN_OFFSET],
                    max_tokens=min(max_tokens, self.max_model_len),
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    *args, **kwargs,
                )
            )
            t3_gen_time = time.time() - t3_start
            if print_metrics:
                print(f"[T3] Speech token generation: {t3_gen_time:.2f}s")

            # Extract tokens from result
            speech_tokens = None
            for batch_result in batch_results:
                for output in batch_result.outputs:
                    tokens = torch.tensor(
                        [token - SPEECH_TOKEN_OFFSET for token in output.token_ids],
                        device="cuda"
                    )
                    speech_tokens = drop_invalid_tokens(tokens)
                    speech_tokens = speech_tokens[speech_tokens < 6561]
                    break
                break

            if speech_tokens is None or len(speech_tokens) == 0:
                return

            # === Stage 2: Stream tokens through S3Gen ===
            all_tokens_processed = torch.tensor([], device="cuda", dtype=torch.long)

            for i in range(0, len(speech_tokens), chunk_size):
                # Get chunk
                chunk = speech_tokens[i:i + chunk_size].unsqueeze(0)

                # Process with S3Gen
                audio_chunk = self._process_token_chunk(
                    token_chunk=chunk,
                    all_tokens_so_far=all_tokens_processed,
                    s3gen_ref=s3gen_ref,
                    context_window=context_window,
                    fade_duration=fade_duration,
                )

                # Update metrics
                if metrics.chunk_count == 0:
                    metrics.latency_to_first_chunk = time.time() - start_time
                    if print_metrics:
                        print(f"Latency to first chunk: {metrics.latency_to_first_chunk:.3f}s")

                metrics.chunk_count += 1

                # Yield if we got audio
                if audio_chunk is not None:
                    metrics.total_audio_duration += audio_chunk.shape[-1] / S3GEN_SR
                    yield audio_chunk.cpu(), metrics

                # Update processed tokens
                all_tokens_processed = torch.cat([all_tokens_processed, chunk.squeeze(0)], dim=0)

            # Final metrics
            metrics.total_generation_time = time.time() - start_time
            if metrics.total_audio_duration > 0:
                metrics.rtf = metrics.total_generation_time / metrics.total_audio_duration

            if print_metrics:
                print(f"[S3Gen] Streaming complete: {metrics.chunk_count} chunks")
                print(f"Total time: {metrics.total_generation_time:.2f}s, "
                      f"Audio: {metrics.total_audio_duration:.2f}s, "
                      f"RTF: {metrics.rtf:.3f}")

    def shutdown(self):
        del self.t3
        torch.cuda.empty_cache()
