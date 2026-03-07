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

@pytest.mark.asyncio
async def test_process_async_single_request(mock_s3gen):
    """Test processing a single request through stream pool."""
    pool = S3GenStreamPool(mock_s3gen, num_streams=4, device="cuda")
    await pool.initialize()

    token_chunk = torch.tensor([[1, 2, 3, 4, 5]])
    s3gen_ref = {"embedding": torch.randn(1, 80)}

    result = await pool.process_async(
        token_chunk=token_chunk,
        context_tokens=None,
        s3gen_ref=s3gen_ref,
        context_window=5,
        fade_duration=0.02,
        diffusion_steps=10,
    )

    assert result is not None
    assert result.shape[0] == 1  # Batch dimension
    assert pool.metrics.total_requests == 1
    assert pool.metrics.active_streams == 0  # Released back to pool

    await pool.shutdown()

@pytest.mark.asyncio
async def test_process_async_with_context(mock_s3gen):
    """Test processing with context tokens."""
    pool = S3GenStreamPool(mock_s3gen, num_streams=4, device="cuda")
    await pool.initialize()

    token_chunk = torch.tensor([[1, 2, 3]])
    context_tokens = torch.tensor([10, 11, 12, 13, 14])
    s3gen_ref = {"embedding": torch.randn(1, 80)}

    # Mock should be called with concatenated tokens
    mock_s3gen.inference.return_value = torch.randn(1, 24000)

    result = await pool.process_async(
        token_chunk=token_chunk,
        context_tokens=context_tokens,
        s3gen_ref=s3gen_ref,
        context_window=5,
    )

    assert result is not None
    # Verify inference was called
    mock_s3gen.inference.assert_called_once()

    await pool.shutdown()

@pytest.mark.asyncio
async def test_concurrent_requests(mock_s3gen):
    """Test that multiple requests can run concurrently."""
    pool = S3GenStreamPool(mock_s3gen, num_streams=4, device="cuda")
    await pool.initialize()

    num_requests = 8

    # Create requests
    async def make_request(i):
        token_chunk = torch.tensor([[i, i+1, i+2, i+3, i+4]])
        s3gen_ref = {"embedding": torch.randn(1, 80)}
        return await pool.process_async(
            token_chunk=token_chunk,
            context_tokens=None,
            s3gen_ref=s3gen_ref,
        )

    # Launch all requests concurrently
    results = await asyncio.gather(*[make_request(i) for i in range(num_requests)])

    # Verify all completed
    assert len(results) == num_requests
    assert all(r is not None for r in results)
    assert pool.metrics.total_requests == num_requests

    # All streams should be back in pool
    assert pool.stream_queue.qsize() == pool.num_streams

    await pool.shutdown()

@pytest.mark.asyncio
async def test_stream_reuse(mock_s3gen):
    """Test that streams are reused properly."""
    pool = S3GenStreamPool(mock_s3gen, num_streams=2, device="cuda")
    await pool.initialize()

    # Process more requests than streams
    for i in range(6):
        token_chunk = torch.tensor([[i, i+1]])
        s3gen_ref = {"embedding": torch.randn(1, 80)}
        result = await pool.process_async(
            token_chunk=token_chunk,
            context_tokens=None,
            s3gen_ref=s3gen_ref,
        )
        assert result is not None

    # All streams should be back in queue
    assert pool.stream_queue.qsize() == pool.num_streams
    assert pool.metrics.total_requests == 6

    await pool.shutdown()
