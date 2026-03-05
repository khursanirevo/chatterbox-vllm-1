"""
TTFA (Time To First Audio) profiling infrastructure for ChatterboxTTS.

This module provides utilities for tracking and analyzing TTFA metrics across
different stages of the TTS pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import time
import csv
from pathlib import Path
import statistics
import logging

logger = logging.getLogger(__name__)


@dataclass
class TTFAMetrics:
    """
    TTFA metrics for a single TTS request.

    Tracks timing for each stage of the TTS pipeline:
    - Queue time: Time from request arrival to start of processing
    - Tokenizer time: Time to tokenize input text
    - T3 first token: Time to first token from T3 model
    - S3 first chunk: Time to first audio chunk from S3 model
    - TTFA: Total time to first audio (queue + t3_first_token + s3_first_chunk)
    """
    request_id: str
    input_length_tokens: int
    input_length_chars: int
    queue_time_ms: float
    tokenizer_time_ms: float
    t3_first_token_ms: float
    s3_first_chunk_ms: float
    ttfa_ms: float
    total_time_ms: float
    category: str = "unknown"  # short, medium, long

    # Additional metadata
    temperature: float = 0.8
    max_tokens: int = 1000
    top_p: float = 1.0
    repetition_penalty: float = 2.0

    @classmethod
    def create(
        cls,
        request_id: str,
        input_length_tokens: int,
        input_length_chars: int,
        queue_start: float,
        tokenizer_start: float,
        t3_start: float,
        t3_first_token_time: float,
        s3_start: float,
        s3_first_chunk_time: float,
        total_end: float,
        category: str = "unknown",
        **kwargs
    ) -> 'TTFAMetrics':
        """Create TTFAMetrics from timestamp measurements."""
        queue_time_ms = (tokenizer_start - queue_start) * 1000
        tokenizer_time_ms = (t3_start - tokenizer_start) * 1000
        t3_first_token_ms = (t3_first_token_time - t3_start) * 1000
        s3_first_chunk_ms = (s3_first_chunk_time - s3_start) * 1000
        ttfa_ms = queue_time_ms + t3_first_token_ms + s3_first_chunk_ms
        total_time_ms = (total_end - queue_start) * 1000

        return cls(
            request_id=request_id,
            input_length_tokens=input_length_tokens,
            input_length_chars=input_length_chars,
            queue_time_ms=queue_time_ms,
            tokenizer_time_ms=tokenizer_time_ms,
            t3_first_token_ms=t3_first_token_ms,
            s3_first_chunk_ms=s3_first_chunk_ms,
            ttfa_ms=ttfa_ms,
            total_time_ms=total_time_ms,
            category=category,
            **kwargs
        )


class TTFAProfiler:
    """
    Context manager and aggregator for TTFA profiling.

    Usage:
        profiler = TTFAProfiler()

        with profiler.profile_request("req_1", text) as ctx:
            # Your TTS generation code here
            ctx.record_tokenizer_start()
            tokens = tokenize(text)
            ctx.record_t3_start()
            t3_output = t3_model.generate(tokens)
            ctx.record_t3_first_token()
            s3_output = s3_model.generate(t3_output)
            ctx.record_s3_first_chunk()

        # Get statistics
        stats = profiler.get_statistics()
        print(f"P50 TTFA: {stats['ttfa_ms']['p50']:.2f}ms")
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize TTFAProfiler.

        Args:
            output_dir: Optional directory to save CSV reports
        """
        self.metrics: List[TTFAMetrics] = []
        self.output_dir = output_dir or Path("./ttfa_profiles")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Current request context
        self._current_request: Optional['_RequestContext'] = None

    def profile_request(self, request_id: str, text: str, category: str = "unknown"):
        """
        Create a context manager for profiling a single request.

        Args:
            request_id: Unique identifier for this request
            text: Input text (for length tracking)
            category: Request category (short/medium/long)

        Returns:
            RequestContext for timing measurements
        """
        self._current_request = _RequestContext(
            profiler=self,
            request_id=request_id,
            text=text,
            category=category
        )
        return self._current_request

    def add_metrics(self, metrics: TTFAMetrics):
        """Add completed metrics to the profiler."""
        self.metrics.append(metrics)
        logger.info(
            f"[TTFA] {metrics.request_id} | "
            f"Category: {metrics.category} | "
            f"Input: {metrics.input_length_tokens} tokens | "
            f"TTFA: {metrics.ttfa_ms:.2f}ms | "
            f"Queue: {metrics.queue_time_ms:.2f}ms | "
            f"T3: {metrics.t3_first_token_ms:.2f}ms | "
            f"S3: {metrics.s3_first_chunk_ms:.2f}ms"
        )

    def get_statistics(self, category: Optional[str] = None) -> Dict:
        """
        Get aggregated statistics for all recorded metrics.

        Args:
            category: Optional category filter (short/medium/long)

        Returns:
            Dictionary with P50, P95, P99 statistics for each metric
        """
        filtered = self.metrics if category is None else [
            m for m in self.metrics if m.category == category
        ]

        if not filtered:
            return {}

        def percentile(data, p):
            return statistics.quantiles(data, n=100)[int(p) - 1] if len(data) >= 100 else \
                   statistics.quantiles(data, n=10)[int((p - 1) // 10)]

        return {
            "count": len(filtered),
            "ttfa_ms": {
                "p50": percentile([m.ttfa_ms for m in filtered], 50),
                "p95": percentile([m.ttfa_ms for m in filtered], 95),
                "p99": percentile([m.ttfa_ms for m in filtered], 99),
                "mean": statistics.mean([m.ttfa_ms for m in filtered]),
                "min": min([m.ttfa_ms for m in filtered]),
                "max": max([m.ttfa_ms for m in filtered]),
            },
            "queue_time_ms": {
                "p50": percentile([m.queue_time_ms for m in filtered], 50),
                "p95": percentile([m.queue_time_ms for m in filtered], 95),
                "mean": statistics.mean([m.queue_time_ms for m in filtered]),
            },
            "t3_first_token_ms": {
                "p50": percentile([m.t3_first_token_ms for m in filtered], 50),
                "p95": percentile([m.t3_first_token_ms for m in filtered], 95),
                "mean": statistics.mean([m.t3_first_token_ms for m in filtered]),
            },
            "s3_first_chunk_ms": {
                "p50": percentile([m.s3_first_chunk_ms for m in filtered], 50),
                "p95": percentile([m.s3_first_chunk_ms for m in filtered], 95),
                "mean": statistics.mean([m.s3_first_chunk_ms for m in filtered]),
            },
            "input_length_tokens": {
                "p50": percentile([m.input_length_tokens for m in filtered], 50),
                "p95": percentile([m.input_length_tokens for m in filtered], 95),
                "mean": statistics.mean([m.input_length_tokens for m in filtered]),
            }
        }

    def save_csv(self, filename: str = "ttfa_metrics.csv"):
        """
        Save all metrics to a CSV file.

        Args:
            filename: Output filename
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "request_id", "category", "input_length_tokens", "input_length_chars",
                "queue_time_ms", "tokenizer_time_ms", "t3_first_token_ms",
                "s3_first_chunk_ms", "ttfa_ms", "total_time_ms",
                "temperature", "max_tokens", "top_p", "repetition_penalty"
            ])

            for m in self.metrics:
                writer.writerow([
                    m.request_id, m.category, m.input_length_tokens, m.input_length_chars,
                    f"{m.queue_time_ms:.2f}", f"{m.tokenizer_time_ms:.2f}",
                    f"{m.t3_first_token_ms:.2f}", f"{m.s3_first_chunk_ms:.2f}",
                    f"{m.ttfa_ms:.2f}", f"{m.total_time_ms:.2f}",
                    m.temperature, m.max_tokens, m.top_p, m.repetition_penalty
                ])

        logger.info(f"Saved {len(self.metrics)} metrics to {output_path}")

    def print_summary(self):
        """Print a summary of TTFA statistics by category."""
        print("\n" + "="*80)
        print("TTFA PROFILING SUMMARY")
        print("="*80)

        for category in ["short", "medium", "long", "unknown"]:
            stats = self.get_statistics(category)
            if stats:
                print(f"\n{category.upper()} Requests (n={stats['count']}):")
                print(f"  Input Length (tokens): P50={stats['input_length_tokens']['p50']:.1f}, "
                      f"P95={stats['input_length_tokens']['p95']:.1f}")
                print(f"  TTFA: P50={stats['ttfa_ms']['p50']:.2f}ms, "
                      f"P95={stats['ttfa_ms']['p95']:.2f}ms, "
                      f"P99={stats['ttfa_ms']['p99']:.2f}ms")
                print(f"  Queue Time: P50={stats['queue_time_ms']['p50']:.2f}ms, "
                      f"P95={stats['queue_time_ms']['p95']:.2f}ms")
                print(f"  T3 First Token: P50={stats['t3_first_token_ms']['p50']:.2f}ms, "
                      f"P95={stats['t3_first_token_ms']['p95']:.2f}ms")
                print(f"  S3 First Chunk: P50={stats['s3_first_chunk_ms']['p50']:.2f}ms, "
                      f"P95={stats['s3_first_chunk_ms']['p95']:.2f}ms")

        # Overall stats
        overall = self.get_statistics()
        if overall:
            print(f"\nOVERALL (n={overall['count']}):")
            print(f"  TTFA: P50={overall['ttfa_ms']['p50']:.2f}ms, "
                  f"P95={overall['ttfa_ms']['p95']:.2f}ms, "
                  f"P99={overall['ttfa_ms']['p99']:.2f}ms")

        print("="*80 + "\n")


class _RequestContext:
    """Internal context manager for tracking request timings."""

    def __init__(self, profiler: TTFAProfiler, request_id: str, text: str, category: str):
        self.profiler = profiler
        self.request_id = request_id
        self.text = text
        self.category = category
        self.input_length_chars = len(text)
        self.input_length_tokens = 0  # Will be set during tokenization

        # Timestamps
        self.queue_start = time.time()
        self.tokenizer_start = None
        self.t3_start = None
        self.t3_first_token_time = None
        self.s3_start = None
        self.s3_first_chunk_time = None
        self.total_end = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Auto-finalize if not already done
        if self.total_end is None:
            self.finalize()

    def record_tokenizer_start(self):
        """Record start of tokenization."""
        self.tokenizer_start = time.time()

    def record_tokenizer_end(self, token_count: int):
        """Record end of tokenization and token count."""
        self.input_length_tokens = token_count

    def record_t3_start(self):
        """Record start of T3 generation."""
        self.t3_start = time.time()

    def record_t3_first_token(self):
        """Record when first token is generated from T3."""
        if self.t3_first_token_time is None:
            self.t3_first_token_time = time.time()

    def record_s3_start(self):
        """Record start of S3 generation."""
        self.s3_start = time.time()

    def record_s3_first_chunk(self):
        """Record when first audio chunk is generated from S3."""
        if self.s3_first_chunk_time is None:
            self.s3_first_chunk_time = time.time()

    def finalize(self):
        """Finalize metrics and add to profiler."""
        if self.total_end is not None:
            return  # Already finalized

        self.total_end = time.time()

        # Validate all timestamps are set
        if None in [self.tokenizer_start, self.t3_start, self.t3_first_token_time,
                    self.s3_start, self.s3_first_chunk_time]:
            logger.warning(f"Missing timestamps for request {self.request_id}, skipping metrics")
            return

        metrics = TTFAMetrics.create(
            request_id=self.request_id,
            input_length_tokens=self.input_length_tokens,
            input_length_chars=self.input_length_chars,
            queue_start=self.queue_start,
            tokenizer_start=self.tokenizer_start,
            t3_start=self.t3_start,
            t3_first_token_time=self.t3_first_token_time,
            s3_start=self.s3_start,
            s3_first_chunk_time=self.s3_first_chunk_time,
            total_end=self.total_end,
            category=self.category
        )

        self.profiler.add_metrics(metrics)


# Convenience decorator for profiling functions
def profile_ttfa(profiler: TTFAProfiler, category: str = "unknown"):
    """
    Decorator to automatically profile a TTS generation function.

    Usage:
        profiler = TTFAProfiler()

        @profile_ttfa(profiler, category="short")
        async def generate_tts(text: str):
            # Your TTS generation code
            return audio

        # The decorator will automatically track timing
        audio = await generate_tts("Hello world")
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            request_id = f"req_{time.time()}"
            text = kwargs.get('prompts', args[0] if args else "")
            if isinstance(text, list):
                text = text[0] if text else ""

            with profiler.profile_request(request_id, text, category) as ctx:
                result = await func(*args, **kwargs)

            return result

        return async_wrapper
    return decorator
