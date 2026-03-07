"""CUDA stream pool for concurrent S3Gen inference."""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class StreamPoolMetrics:
    """Metrics for stream pool performance monitoring."""
    total_requests: int = 0
    active_streams: int = 0
    queue_depth: int = 0
    avg_queue_wait_ms: float = 0.0
    stream_utilization: list = field(default_factory=list)


class S3GenStreamPool:
    """Manages a pool of CUDA streams for concurrent S3Gen inference.

    Multiple S3Gen operations can run concurrently on the GPU by using
    different CUDA streams. This pool manages stream allocation and
    provides fair distribution via asyncio.Queue.

    Args:
        s3gen_model: The S3Token2Wav model for inference
        num_streams: Number of CUDA streams in the pool
        device: CUDA device to use
    """

    def __init__(
        self,
        s3gen_model: Any,
        num_streams: int = 12,
        device: str = "cuda",
    ):
        self.s3gen = s3gen_model
        self.device = device
        self.num_streams = num_streams

        # Create CUDA streams
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]

        # Will be initialized in async context
        self.stream_queue: Optional[asyncio.Queue] = None

        # Metrics tracking
        self.metrics = StreamPoolMetrics()

    async def initialize(self) -> None:
        """Initialize the stream queue (must be called in async context)."""
        self.stream_queue = asyncio.Queue()
        for stream in self.streams:
            await self.stream_queue.put(stream)

        logger.info(f"S3GenStreamPool initialized with {self.num_streams} streams")

    async def shutdown(self) -> None:
        """Gracefully shutdown - wait for active requests to complete."""
        # Wait for all active streams to return
        while self.metrics.active_streams > 0:
            await asyncio.sleep(0.1)

        # Clear queue
        if self.stream_queue:
            while not self.stream_queue.empty():
                self.stream_queue.get_nowait()

        logger.info(f"S3GenStreamPool shutdown. Total requests: {self.metrics.total_requests}")
