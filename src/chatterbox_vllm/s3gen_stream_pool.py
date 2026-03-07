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

    async def __aenter__(self) -> "S3GenStreamPool":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.shutdown()
