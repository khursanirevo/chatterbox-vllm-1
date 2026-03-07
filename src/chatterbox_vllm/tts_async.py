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
    ):
        self.engine = engine
        self.s3gen = s3gen
        self.ve = ve
        self.default_conds = default_conds
        self.max_model_len = max_model_len
        self.device = device
        self.variant = variant

    @classmethod
    async def from_pretrained(
        cls,
        model_path: Optional[str] = None,
        audio_prompt_path: Optional[str] = None,
        variant: str = "english",
        max_model_len: int = 2000,
        gpu_memory_utilization: float = 0.90,
        enforce_eager: bool = True,
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
            **kwargs: Additional arguments for AsyncEngineArgs
        """
        if model_path is None:
            model_path = "./t3-model"

        # Ensure model exists
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        # Initialize AsyncLLMEngine
        print("Initializing AsyncLLMEngine...")
        engine_args = AsyncEngineArgs(
            model=str(model_path),
            tokenizer="EnTokenizer",
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

        # Get device from engine
        device = "cuda:0"  # Simplified for single GPU

        # Load S3Gen and other components (reuse existing code from tts.py)
        # This would need to be extracted from the synchronous ChatterboxTTS class
        # For now, we'll need to load these components separately

        # TODO: Load S3Gen, VoiceEncoder, and default conditionals
        # This requires refactoring the synchronous class to extract common loading logic

        raise NotImplementedError(
            "AsyncChatterboxTTS.from_pretrained() requires extracting common loading logic "
            "from ChatterboxTTS. See TODO in tts_async.py."
        )

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
        chunk_size: int = 25,
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
            chunk_size: Speech tokens per audio chunk
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

        # Prepare text
        if self.variant == "multilingual":
            text = f"<|{SUPPORTED_LANGUAGES[language_id]}|> {text}"

        # Create prompt
        prompt = f"[START]{text}[STOP]"

        # Setup sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Collect all tokens
        all_tokens = []
        first_token_time = None
        t3_start_time = time.time()

        request_id = f"tts-request-{time.time()}"

        # Stream tokens from AsyncLLMEngine
        async for request_output in self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            current_time = time.time()

            # Track first token time
            if first_token_time is None and request_output.outputs:
                first_token_time = current_time
                metrics.t3_first_token_time = first_token_time - t3_start_time

            # Collect tokens
            if request_output.outputs:
                output = request_output.outputs[0]
                all_tokens = list(output.token_ids)

                # Process chunk if we have enough tokens
                if len(all_tokens) >= chunk_size:
                    # Extract current chunk
                    current_chunk_start = max(0, len(all_tokens) - chunk_size)
                    token_chunk = all_tokens[current_chunk_start:]

                    # Get context for continuity
                    context_start = max(0, len(all_tokens) - chunk_size - context_window)
                    context_tokens = all_tokens[context_start:current_chunk_start]

                    # Process tokens to audio
                    audio_chunk = await self._process_token_chunk_async(
                        token_chunk=torch.tensor([token_chunk]),
                        context_tokens=torch.tensor([context_tokens]) if context_tokens else None,
                        conditionals=self.default_conds,
                        fade_duration=fade_duration,
                        diffusion_steps=diffusion_steps,
                    )

                    # Update metrics
                    if metrics.chunk_count == 0:
                        metrics.latency_to_first_chunk = current_time - start_time
                        metrics.s3gen_first_chunk_time = current_time - first_token_time

                    metrics.chunk_count += 1
                    metrics.last_chunk_time = current_time - start_time

                    if audio_chunk is not None:
                        yield audio_chunk, metrics

            # Check if generation is complete
            if request_output.finished:
                break

        # Process remaining tokens
        if all_tokens:
            # ... process final chunk
            pass

        # Final metrics
        metrics.total_generation_time = time.time() - start_time
        metrics.t3_token_generation_time = first_token_time - t3_start_time if first_token_time else 0

    async def _process_token_chunk_async(
        self,
        token_chunk: torch.Tensor,
        context_tokens: Optional[torch.Tensor],
        conditionals: Conditionals,
        fade_duration: float,
        diffusion_steps: int,
    ) -> Optional[torch.Tensor]:
        """
        Process a chunk of speech tokens to audio asynchronously.

        This is the async version of _process_token_chunk from tts.py.
        """
        # TODO: Implement async S3Gen processing
        # This requires either:
        # 1. Making S3Gen async-compatible
        # 2. Running S3Gen in a thread pool executor
        # 3. Using synchronous calls (will block the event loop)

        # For now, we'll use synchronous S3Gen calls
        # This is not ideal but will work for initial implementation

        raise NotImplementedError(
            "Async S3Gen processing not yet implemented. "
            "This requires either async-compatible S3Gen or thread pool execution."
        )

    async def shutdown(self):
        """Cleanup resources."""
        # TODO: Properly shutdown the AsyncLLMEngine
        del self.engine


# TODO: Refactor ChatterboxTTS to extract common loading logic
# that can be shared between sync and async versions
