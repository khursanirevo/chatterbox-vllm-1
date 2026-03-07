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

@pytest.mark.asyncio
async def test_build_token_context(mock_s3gen):
    """Test building token context with context window."""
    pool = S3GenStreamPool(mock_s3gen, num_streams=4, device="cuda")
    await pool.initialize()

    # Create test data
    token_chunk = torch.tensor([[1, 2, 3, 4, 5]])
    context_tokens = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    context_window = 5

    result = pool._build_token_context(token_chunk, context_tokens, context_window)

    # Should take last 5 from context, plus new chunk
    expected = torch.tensor([[15, 16, 17, 18, 19, 1, 2, 3, 4, 5]])
    assert torch.equal(result, expected)

    await pool.shutdown()

@pytest.mark.asyncio
async def test_build_token_context_no_context(mock_s3gen):
    """Test building token context with no context tokens."""
    pool = S3GenStreamPool(mock_s3gen, num_streams=4, device="cuda")
    await pool.initialize()

    token_chunk = torch.tensor([[1, 2, 3]])
    result = pool._build_token_context(token_chunk, None, 5)

    assert torch.equal(result, token_chunk)

    await pool.shutdown()

@pytest.mark.asyncio
async def test_build_token_context_context_larger_than_window(mock_s3gen):
    """Test when context_tokens is smaller than context_window."""
    pool = S3GenStreamPool(mock_s3gen, num_streams=4, device="cuda")
    await pool.initialize()

    token_chunk = torch.tensor([[1, 2, 3]])
    context_tokens = torch.tensor([10, 11])  # Only 2 tokens, window is 5
    result = pool._build_token_context(token_chunk, context_tokens, 5)

    # Should use all available context
    expected = torch.tensor([[10, 11, 1, 2, 3]])
    assert torch.equal(result, expected)

    await pool.shutdown()
