"""CUDA stream pool for concurrent S3Gen inference."""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Any, Union, List

import torch

logger = logging.getLogger(__name__)


@dataclass
class StreamPoolMetrics:
    """Metrics for stream pool performance monitoring."""
    total_requests: int = 0
    active_streams: int = 0
    queue_depth: int = 0
    avg_queue_wait_ms: float = 0.0
    stream_utilization: List[float] = field(default_factory=list)


class S3GenStreamPool:
    """Manages a pool of CUDA streams for concurrent S3Gen inference.

    Multiple S3Gen operations can run concurrently on the GPU by using
    different CUDA streams. This pool manages stream allocation and
    provides fair distribution via asyncio.Queue.

    Args:
        s3gen_model: The S3Token2Wav model for inference
        num_streams: Number of CUDA streams in the pool (must be positive)
        device: CUDA device to use

    Raises:
        ValueError: If num_streams is not positive
        RuntimeError: If CUDA is not available

    Example:
        >>> pool = S3GenStreamPool(s3gen_model, num_streams=4)
        >>> await pool.initialize()
        >>> # Use pool...
        >>> await pool.shutdown()
    """

    def __init__(
        self,
        s3gen_model: Union["S3Token2Wav", Any],
        num_streams: int = 12,
        device: str = "cuda",
    ):
        if num_streams <= 0:
            raise ValueError(f"num_streams must be positive, got {num_streams}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        self.s3gen = s3gen_model
        self.device = device
        self.num_streams = num_streams

        # Create CUDA streams
        self.streams: List[torch.cuda.Stream] = [torch.cuda.Stream() for _ in range(num_streams)]

        # Will be initialized in async context
        self.stream_queue: Optional[asyncio.Queue[torch.cuda.Stream]] = None

        # Metrics tracking
        self.metrics = StreamPoolMetrics()

    async def initialize(self) -> None:
        """Initialize the stream queue (must be called in async context).

        Raises:
            RuntimeError: If CUDA stream creation fails
        """
        try:
            self.stream_queue = asyncio.Queue()
            for stream in self.streams:
                await self.stream_queue.put(stream)
            logger.info(f"S3GenStreamPool initialized with {self.num_streams} streams")
        except Exception as e:
            logger.error(f"Failed to initialize S3GenStreamPool: {e}")
            raise RuntimeError(f"S3GenStreamPool initialization failed: {e}") from e

    async def shutdown(self, timeout: float = 30.0) -> None:
        """Gracefully shutdown - wait for active requests to complete.

        Args:
            timeout: Maximum seconds to wait for active streams (default: 30.0)

        Raises:
            asyncio.TimeoutError: If timeout expires before all streams complete
        """
        start_time = time.time()
        while self.metrics.active_streams > 0:
            if time.time() - start_time > timeout:
                raise asyncio.TimeoutError(f"Shutdown timeout: {self.metrics.active_streams} streams still active")
            await asyncio.sleep(0.1)

        # Clear queue
        if self.stream_queue:
            while not self.stream_queue.empty():
                self.stream_queue.get_nowait()

        logger.info(f"S3GenStreamPool shutdown. Total requests: {self.metrics.total_requests}")

    def _build_token_context(
        self,
        token_chunk: torch.Tensor,
        context_tokens: Optional[torch.Tensor],
        context_window: int,
    ) -> torch.Tensor:
        """Build tokens with context window for continuity.

        Args:
            token_chunk: New tokens to process (1, T_new)
            context_tokens: Optional context tokens for continuity
            context_window: How many context tokens to include

        Returns:
            Tensor with shape (1, T_context + T_new)
        """
        if context_tokens is not None and len(context_tokens) > 0:
            # Ensure context_tokens is 1D for slicing
            if context_tokens.dim() > 1:
                ctx_tokens = context_tokens.squeeze(0)
            else:
                ctx_tokens = context_tokens

            # Take last N tokens from context (or all if less than window)
            ctx_window = (
                ctx_tokens[-context_window:]
                if len(ctx_tokens) > context_window
                else ctx_tokens
            )

            # Ensure token_chunk is 1D for concatenation
            chunk_1d = (
                token_chunk.squeeze(0)
                if token_chunk.dim() > 1
                else token_chunk
            )

            # Concatenate context + new chunk
            tokens_to_process = torch.cat([ctx_window, chunk_1d], dim=-1).unsqueeze(0)
            return tokens_to_process
        else:
            # No context, return as-is
            return token_chunk

    async def process_async(
        self,
        token_chunk: torch.Tensor,
        context_tokens: Optional[torch.Tensor],
        s3gen_ref: dict[str, Any],
        context_window: int = 50,
        fade_duration: float = 0.02,
        diffusion_steps: int = 10,
    ) -> Optional[torch.Tensor]:
        """Process tokens through S3Gen on an available CUDA stream.

        Args:
            token_chunk: New tokens to process (1, T_new)
            context_tokens: Optional context tokens for continuity
            s3gen_ref: S3Gen reference dictionary
            context_window: Context tokens to include for continuity
            fade_duration: Fade-in duration in seconds
            diffusion_steps: S3Gen diffusion steps

        Returns:
            Audio chunk tensor or None if processing failed
        """
        if self.stream_queue is None:
            raise RuntimeError("Stream pool not initialized. Call initialize() first.")

        # Step 1: Get stream from queue (fair FIFO)
        queue_start = time.time()
        stream = await self.stream_queue.get()
        queue_wait_ms = (time.time() - queue_start) * 1000

        # Track metrics
        self.metrics.active_streams += 1
        self.metrics.queue_depth = self.stream_queue.qsize()

        try:
            # Step 2: Run inference in thread pool (non-blocking)
            loop = asyncio.get_event_loop()

            def _inference_on_stream():
                """Synchronous function to run in thread pool."""
                with torch.cuda.stream(stream):
                    # Build tokens with context
                    tokens_to_process = self._build_token_context(
                        token_chunk, context_tokens, context_window
                    )

                    # Run S3Gen inference (concurrent with other streams)
                    audio = self.s3gen.inference(
                        speech_tokens=tokens_to_process,
                        ref_dict=s3gen_ref,
                        finalize=False,
                        n_timesteps=diffusion_steps,
                    )

                    return audio

            # Submit to thread pool
            audio_chunk = await loop.run_in_executor(
                None,  # Use default executor
                _inference_on_stream,
            )

            return audio_chunk

        finally:
            # Step 3: Always return stream to pool (even on error)
            self.metrics.active_streams -= 1
            await self.stream_queue.put(stream)

            # Update metrics
            self.metrics.total_requests += 1
            if self.metrics.total_requests > 0:
                self.metrics.avg_queue_wait_ms = (
                    (self.metrics.avg_queue_wait_ms * (self.metrics.total_requests - 1) + queue_wait_ms)
                    / self.metrics.total_requests
                )

    async def __aenter__(self) -> "S3GenStreamPool":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.shutdown()
