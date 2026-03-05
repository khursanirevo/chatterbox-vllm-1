"""
Async ChatterboxTTS using AsyncLLMEngine for continuous batching.

This module provides an async version of ChatterboxTTS that uses vLLM's AsyncLLMEngine,
enabling continuous batching for better throughput and lower latency when handling
multiple concurrent TTS requests.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple, Any, AsyncGenerator
import time
import asyncio

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

import librosa
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from chatterbox_vllm.models.t3.modules.t3_config import T3Config

# Import models/t3 to trigger tokenizer registration before AsyncLLMEngine
import chatterbox_vllm.models.t3  # This registers custom tokenizers with vLLM

from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.voice_encoder import VoiceEncoder
from .models.t3 import SPEECH_TOKEN_OFFSET
from .models.t3.modules.cond_enc import T3Cond, T3CondEnc
from .models.t3.modules.learned_pos_emb import LearnedPositionEmbeddings
from .text_utils import punc_norm, SUPPORTED_LANGUAGES
from .profiling import TTFAProfiler, TTFAMetrics
from .adaptive_config import (
    classify_request,
    classify_request_by_chars,
    get_profile,
    get_priority_for_category,
    is_adaptive_mode_enabled,
    get_default_profile,
)

REPO_ID = "ResembleAI/chatterbox"


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
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


class ChatterboxTTSAsync:
    """
    Async ChatterboxTTS using AsyncLLMEngine for continuous batching.

    This allows multiple TTS requests to be processed concurrently with dynamic batching,
    significantly improving throughput and reducing latency compared to static batching.
    """
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(self, target_device: str, max_model_len: int,
                 t3_engine: AsyncLLMEngine, t3_config: T3Config, t3_cond_enc: T3CondEnc,
                 t3_speech_emb: torch.nn.Embedding, t3_speech_pos_emb: LearnedPositionEmbeddings,
                 s3gen: S3Gen, ve: VoiceEncoder, default_conds: Conditionals,
                 variant: str = "english",
                 enable_ttfa_tracking: bool = False):
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

        # TTFA Profiling
        self.enable_ttfa_tracking = enable_ttfa_tracking
        self.ttfa_profiler = TTFAProfiler() if enable_ttfa_tracking else None

    @property
    def sr(self) -> int:
        """Sample rate of synthesized audio"""
        return S3GEN_SR

    @classmethod
    async def from_local(cls, ckpt_dir: str | Path, target_device: str = "cuda",
                        max_model_len: int = 1000, compile: bool = False,
                        max_batch_size: int = 10,
                        variant: str = "english",
                        s3gen_use_fp16: bool = False,
                        s3gen_compile_model: bool = False,
                        enable_ttfa_tracking: bool = False,
                        **kwargs) -> 'ChatterboxTTSAsync':
        """
        Create ChatterboxTTSAsync from local checkpoint directory.

        Args:
            ckpt_dir: Path to checkpoint directory
            target_device: Device to load models on
            max_model_len: Maximum sequence length for T3 model
            compile: Whether to use CUDA graphs (not recommended currently)
            max_batch_size: Maximum batch size for continuous batching
            variant: Model variant ("english" or "multilingual")
            s3gen_use_fp16: Whether to use FP16 for S3Gen
            s3gen_compile_model: Whether to compile S3Gen with torch.compile() (30-40% speedup)
            enable_ttfa_tracking: Enable TTFA profiling and metrics collection
            **kwargs: Additional arguments for AsyncEngineArgs

        Returns:
            ChatterboxTTSAsync instance with AsyncLLMEngine
        """
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
        vllm_memory_needed = (1.55*1024*1024*1024) + (max_batch_size * max_model_len * 1024 * 128)
        vllm_memory_percent = vllm_memory_needed / unused_gpu_memory

        print(f"Giving vLLM {vllm_memory_percent * 100:.2f}% of GPU memory ({vllm_memory_needed / 1024**2:.2f} MB)")

        # CRITICAL: Set up sitecustomize for spawned worker processes
        # AsyncLLMEngine spawns worker processes via 'spawn' method when CUDA is
        # initialized. These workers are fresh Python interpreters that haven't
        # imported chatterbox_vllm.models.t3, causing tokenizer registration to fail.
        # This sets up PYTHONPATH to include sitecustomize.py which registers tokenizers.
        from chatterbox_vllm.vllm_worker_patch import apply_worker_patch
        apply_worker_patch()

        # Set multiprocessing method to 'fork' (will be overridden to 'spawn' by CUDA,
        # but the sitecustomize.py above ensures tokenizers are registered in spawned workers)
        import os
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"

        # Create AsyncEngineArgs for AsyncLLMEngine
        engine_args = AsyncEngineArgs(
            model="./t3-model" if variant == "english" else "./t3-model-multilingual",
            task="generate",
            tokenizer="EnTokenizer" if variant == "english" else "MtlTokenizer",
            tokenizer_mode="custom",
            gpu_memory_utilization=vllm_memory_percent,
            enforce_eager=not compile,
            max_model_len=max_model_len,
            **kwargs
        )

        # Create AsyncLLMEngine (supports continuous batching)
        t3_engine = AsyncLLMEngine.from_engine_args(engine_args)

        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve = ve.to(device=target_device).eval()

        s3gen = S3Gen(use_fp16=s3gen_use_fp16, compile_model=s3gen_compile_model)
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
        s3gen = s3gen.to(device=target_device).eval()

        default_conds = Conditionals.load(ckpt_dir / "conds.pt")
        default_conds.to(device=target_device)

        return cls(
            target_device=target_device, max_model_len=max_model_len,
            t3_engine=t3_engine, t3_config=t3_config, t3_cond_enc=t3_enc,
            t3_speech_emb=t3_speech_emb, t3_speech_pos_emb=t3_speech_pos_emb,
            s3gen=s3gen, ve=ve, default_conds=default_conds,
            variant=variant,
            enable_ttfa_tracking=enable_ttfa_tracking,
        )

    @classmethod
    async def from_pretrained(cls,
                             repo_id: str = REPO_ID,
                             revision: str = "1b475dffa71fb191cb6d5901215eb6f55635a9b6",
                             *args, **kwargs) -> 'ChatterboxTTSAsync':
        """
        Create ChatterboxTTSAsync from pretrained model on HuggingFace.

        Args:
            repo_id: HuggingFace repository ID
            revision: Git revision to checkout
            *args: Positional args for from_local
            **kwargs: Keyword args for from_local

        Returns:
            ChatterboxTTSAsync instance
        """
        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=repo_id, filename=fpath, revision=revision)

        # Ensure the symlink in './t3-model/model.safetensors' points to t3_cfg_path
        t3_cfg_path = Path(local_path).parent / "t3_cfg.safetensors"
        model_safetensors_path = Path.cwd() / "t3-model" / "model.safetensors"
        model_safetensors_path.unlink(missing_ok=True)
        model_safetensors_path.symlink_to(t3_cfg_path)

        return await cls.from_local(Path(local_path).parent, variant="english", *args, **kwargs)

    @classmethod
    async def from_pretrained_multilingual(cls,
                                         repo_id: str = REPO_ID,
                                         revision: str = "05e904af2b5c7f8e482687a9d7336c5c824467d9",
                                         *args, **kwargs) -> 'ChatterboxTTSAsync':
        """Create multilingual ChatterboxTTSAsync from pretrained model."""
        for fpath in ["ve.safetensors", "t3_mtl23ls_v2.safetensors", "s3gen.safetensors", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"]:
            local_path = hf_hub_download(repo_id=repo_id, filename=fpath, revision=revision)

        # Ensure the symlink in './t3-model-multilingual/model.safetensors' points to t3_cfg_path
        t3_cfg_path = Path(local_path).parent / "t3_mtl23ls_v2.safetensors"
        model_safetensors_path = Path.cwd() / "t3-model-multilingual" / "model.safetensors"
        model_safetensors_path.unlink(missing_ok=True)
        model_safetensors_path.symlink_to(t3_cfg_path)

        return await cls.from_local(Path(local_path).parent, variant="multilingual", *args, **kwargs)

    def get_supported_languages(self) -> dict[str, str]:
        """Return dictionary of supported language codes and names."""
        if self.variant == "multilingual":
            return SUPPORTED_LANGUAGES.copy()
        else:
            return { "en": "English" }

    def get_audio_conditionals(self, wav_fpath: Optional[str] = None) -> Tuple[dict[str, Any], torch.Tensor]:
        """
        Get audio conditionals for generation.

        Note: This is a synchronous method. In production, you may want to cache
        the results or make this async as well.
        """
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
        """Update the emotion exaggeration in the conditioning embedding."""
        if exaggeration == 0.5:
            return cond_emb

        new_cond_emb = cond_emb.clone()
        new_cond_emb[-1] = self.t3_cond_enc.emotion_adv_fc(
            (exaggeration * torch.ones(1, 1)).to(self.target_device)
        ).to('cpu')
        return new_cond_emb

    async def generate(
        self,
        prompts: Union[str, list[str]],
        audio_prompt_path: Optional[str] = None,
        language_id: Optional[str] = 'en',
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        max_tokens=1000,
        top_p=0.8,
        repetition_penalty=2.0,
        *args, **kwargs,
    ) -> list[any]:
        """
        Generate audio from text prompts using continuous batching.

        This async method leverages AsyncLLMEngine's continuous batching to efficiently
        handle multiple concurrent requests.

        Args:
            prompts: Text prompt(s) to synthesize
            audio_prompt_path: Optional reference audio for voice cloning
            language_id: Language code for multilingual models
            exaggeration: Emotion exaggeration (0.5 = neutral)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty
            **kwargs: Additional sampling parameters

        Returns:
            List of generated audio tensors
        """
        s3gen_ref, cond_emb = self.get_audio_conditionals(audio_prompt_path)

        return await self.generate_with_conds(
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

    async def generate_with_conds(
        self,
        prompts: Union[str, list[str]],
        s3gen_ref: dict[str, Any],
        cond_emb: torch.Tensor,
        language_id: Optional[str] = 'en',
        temperature: float = 0.8,
        exaggeration: float = 0.5,
        max_tokens=1000,
        diffusion_steps: int = 5,
        top_p=1.0,
        min_p=0.05,
        repetition_penalty=2.0,
        *args, **kwargs,
    ) -> list[any]:
        """
        Generate audio with pre-computed conditioning.

        This method uses continuous batching via AsyncLLMEngine to process
        multiple requests efficiently.

        TTFA Tracking:
            If enable_ttfa_tracking=True, measures and records timing for:
            - Queue time (arrival to start)
            - Tokenizer time
            - T3 first token time
            - S3 first chunk time
            - Total TTFA
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        # TTFA Tracking: Record queue start time
        queue_start = time.time() if self.enable_ttfa_tracking else None
        tokenizer_start = None

        # Validate language_id
        if language_id and language_id.lower() not in self.get_supported_languages():
            supported_langs = ", ".join(self.get_supported_languages().keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )

        cond_emb = self.update_exaggeration(cond_emb, exaggeration)

        # TTFA Tracking: Record tokenizer start
        if self.enable_ttfa_tracking:
            tokenizer_start = time.time()

        # Norm and tokenize text
        prompts = ["[START]" + punc_norm(p) + "[STOP]" for p in prompts]

        # For multilingual, prepend the language token
        if self.variant == "multilingual":
            prompts = [f"<{language_id.lower()}>{p}" for p in prompts]

        # TTFA Tracking: Classify requests and count tokens
        token_counts = []
        categories = []
        if self.enable_ttfa_tracking:
            tokenizer = self.t3_engine.engine.tokenizer
            for p in prompts:
                tokens = tokenizer.encode(p)
                token_counts.append(len(tokens))
                # Fast classification based on tokens
                if len(tokens) < 20:
                    categories.append("short")
                elif len(tokens) < 50:
                    categories.append("medium")
                else:
                    categories.append("long")

        # Create unique request IDs for continuous batching
        request_ids = [f"tts_req_{time.time()}_{i}" for i in range(len(prompts))]

        # Create sampling params
        sampling_params = SamplingParams(
            temperature=temperature,
            stop_token_ids=[self.t3_config.stop_speech_token + SPEECH_TOKEN_OFFSET],
            max_tokens=min(max_tokens, self.max_model_len),
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Submit all requests to AsyncLLMEngine (continuous batching)
        generators = []
        for req_id, prompt_text in zip(request_ids, prompts):
            # The engine will handle these concurrently with continuous batching
            # Note: AsyncLLMEngine expects multi_modal_data in the prompt dict (TextPrompt format)
            gen = self.t3_engine.generate(
                prompt={
                    "prompt": prompt_text,
                    "multi_modal_data": {
                        "conditionals": [cond_emb],
                    },
                },
                sampling_params=sampling_params,
                request_id=req_id,
            )
            generators.append(gen)

        # Collect results as they complete (continuous batching benefit)
        start_time = time.time()

        # TTFA Tracking: Record T3 start and first token times
        t3_start_times = {req_id: None for req_id in request_ids}
        t3_first_token_times = {req_id: None for req_id in request_ids}

        batch_results = []
        for i, generator in enumerate(generators):
            req_id = request_ids[i]
            if t3_start_times[req_id] is None:
                t3_start_times[req_id] = time.time()

            async for request_output in generator:
                # TTFA Tracking: Record first token time
                if self.enable_ttfa_tracking and t3_first_token_times[req_id] is None and request_output.outputs:
                    t3_first_token_times[req_id] = time.time()

                # Get the final output when request is complete
                if request_output.finished:
                    batch_results.append(request_output)

        t3_gen_time = time.time() - start_time
        print(f"[T3] Speech Token Generation time (continuous batching): {t3_gen_time:.2f}s")

        # Run gc
        torch.cuda.empty_cache()

        # Process speech tokens to audio
        start_time = time.time()

        # TTFA Tracking: S3 generation times
        s3_start_times = {}
        s3_first_chunk_times = {}

        results = []

        for i, request_output in enumerate(batch_results):
            req_id = request_ids[i] if i < len(request_ids) else f"unknown_{i}"

            # TTFA Tracking: Record S3 start
            if self.enable_ttfa_tracking:
                s3_start_times[req_id] = time.time()

            for output in request_output.outputs:
                if i % 5 == 0:
                    print(f"[S3] Processing prompt {i} of {len(batch_results)}")

                # Run gc every 10 prompts
                if i % 10 == 0:
                    torch.cuda.empty_cache()

                speech_tokens = torch.tensor([token - SPEECH_TOKEN_OFFSET for token in output.token_ids], device="cuda")
                speech_tokens = drop_invalid_tokens(speech_tokens)
                speech_tokens = speech_tokens[speech_tokens < 6561]

                # TTFA Tracking: Record S3 first chunk time
                if self.enable_ttfa_tracking and req_id not in s3_first_chunk_times:
                    s3_first_chunk_times[req_id] = time.time()

                wav, _ = self.s3gen.inference(
                    speech_tokens=speech_tokens,
                    ref_dict=s3gen_ref,
                    n_timesteps=diffusion_steps,
                )
                results.append(wav.cpu())

        s3gen_gen_time = time.time() - start_time
        print(f"[S3Gen] Waveform Generation time: {s3gen_gen_time:.2f}s")

        # TTFA Tracking: Record metrics
        if self.enable_ttfa_tracking:
            total_end = time.time()
            for i, req_id in enumerate(request_ids):
                if req_id in t3_start_times and req_id in t3_first_token_times and req_id in s3_start_times:
                    metrics = TTFAMetrics.create(
                        request_id=req_id,
                        input_length_tokens=token_counts[i] if i < len(token_counts) else 0,
                        input_length_chars=len(prompts[i]) if i < len(prompts) else 0,
                        queue_start=queue_start,
                        tokenizer_start=tokenizer_start,
                        t3_start=t3_start_times[req_id],
                        t3_first_token_time=t3_first_token_times[req_id],
                        s3_start=s3_start_times[req_id],
                        s3_first_chunk_time=s3_first_chunk_times.get(req_id, s3_start_times[req_id]),
                        total_end=total_end,
                        category=categories[i] if i < len(categories) else "unknown",
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                    )
                    self.ttfa_profiler.add_metrics(metrics)

        return results

    async def shutdown(self):
        """Shutdown the AsyncLLMEngine and clean up resources."""
        # Note: AsyncLLMEngine doesn't have an explicit shutdown method in some versions
        # The engine will be cleaned up when the object is destroyed
        del self.t3_engine
        torch.cuda.empty_cache()

    def get_ttfa_statistics(self, category: Optional[str] = None) -> dict:
        """
        Get TTFA statistics from the profiler.

        Args:
            category: Optional category filter ("short", "medium", "long")

        Returns:
            Dictionary with P50, P95, P99 statistics
        """
        if not self.enable_ttfa_tracking or self.ttfa_profiler is None:
            return {}
        return self.ttfa_profiler.get_statistics(category)

    def print_ttfa_summary(self):
        """Print a summary of TTFA statistics."""
        if not self.enable_ttfa_tracking or self.ttfa_profiler is None:
            print("TTFA tracking is not enabled.")
            return
        self.ttfa_profiler.print_summary()

    def save_ttfa_metrics(self, filename: str = "ttfa_metrics.csv"):
        """
        Save TTFA metrics to CSV file.

        Args:
            filename: Output filename
        """
        if not self.enable_ttfa_tracking or self.ttfa_profiler is None:
            print("TTFA tracking is not enabled.")
            return
        self.ttfa_profiler.save_csv(filename)
