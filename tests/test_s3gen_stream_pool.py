import pytest
import torch
import asyncio
from unittest.mock import Mock, AsyncMock
from chatterbox_vllm.s3gen_stream_pool import S3GenStreamPool

@pytest.fixture
def mock_s3gen():
    """Create a mock S3Gen model."""
    s3gen = Mock()
    s3gen.inference = Mock(return_value=torch.randn(1, 24000))  # 1 second audio
    return s3gen

@pytest.mark.asyncio
async def test_stream_pool_initialization(mock_s3gen):
    """Test that stream pool initializes correctly."""
    pool = S3GenStreamPool(mock_s3gen, num_streams=4, device="cuda")
    await pool.initialize()

    assert pool.num_streams == 4
    assert pool.stream_queue.qsize() == 4
    assert len(pool.streams) == 4
    assert all(isinstance(s, torch.cuda.Stream) for s in pool.streams)

    await pool.shutdown()

@pytest.mark.asyncio
async def test_metrics_initialization(mock_s3gen):
    """Test that metrics are properly initialized."""
    pool = S3GenStreamPool(mock_s3gen, num_streams=4, device="cuda")

    assert pool.metrics.total_requests == 0
    assert pool.metrics.active_streams == 0
    assert pool.metrics.queue_depth == 0
    assert pool.metrics.avg_queue_wait_ms == 0.0
    assert pool.metrics.stream_utilization == []
