# vLLM Tuning for TTFA Optimization - Implementation Summary

## Overview

This implementation optimizes vLLM generation parameters to achieve:
- **TTFA < 1s** for short requests (primary latency metric)
- **Maximize concurrent requests** for throughput
- Support mixed workload (interactive real-time TTS + API service)

## Implementation Completed

### 1. TTFA Profiling Infrastructure ✅

**File**: `src/chatterbox_vllm/profiling.py`

Components:
- `TTFAMetrics` dataclass: Captures timing for each pipeline stage
- `TTFAProfiler` class: Context manager for tracking and aggregating metrics
- Statistics calculation: P50, P95, P99 for each metric
- CSV export: Detailed metrics export for analysis
- Summary reporting: Print formatted summaries by category

Stages tracked:
- Queue time (arrival to start)
- Tokenizer time
- T3 first token time
- S3 first chunk time
- Total TTFA

### 2. Dedicated Profiling Script ✅

**File**: `profile_tts_stages.py`

Features:
- Isolate each stage independently (tokenization, T3, S3)
- Test with fixed inputs: 10, 50, 100, 200 tokens
- Cold vs warm cache performance comparison
- CSV output with timing breakdowns
- Estimated TTFA breakdown by percentage

Usage:
```bash
uv run python profile_tts_stages.py
```

### 3. Inline TTFA Tracking ✅

**Modified**: `src/chatterbox_vllm/tts_async.py`

Changes:
- Added `enable_ttfa_tracking` parameter to `ChatterboxTTSAsync`
- `ttfa_profiler` attribute for metrics collection
- Timing measurements in `generate_with_conds()`:
  - Queue time tracking
  - Tokenizer time tracking
  - T3 first token time tracking
  - S3 first chunk time tracking
- New methods:
  - `get_ttfa_statistics(category)`: Get statistics by category
  - `print_ttfa_summary()`: Print formatted summary
  - `save_ttfa_metrics(filename)`: Export to CSV

Usage:
```python
model = await ChatterboxTTSAsync.from_pretrained(
    enable_ttfa_tracking=True
)
# ... generate requests ...
model.print_ttfa_summary()
model.save_ttfa_metrics("metrics.csv")
```

### 4. Adaptive Configuration System ✅

**File**: `src/chatterbox_vllm/adaptive_config.py`

Profile definitions:

| Category | Max Model Len | Max Seqs | Batch Tokens | GPU Util | Target TTFA |
|----------|--------------|----------|--------------|----------|-------------|
| Short    | 256          | 16       | 4096         | 0.02     | < 1s        |
| Medium   | 512          | 8        | 6144         | 0.03     | < 2s        |
| Long     | 1000         | 4        | 8192         | 0.04     | < 4s        |

Functions:
- `classify_request(text, tokenizer)`: Categorize by token count
- `classify_request_by_chars(text)`: Fast heuristic classification
- `get_profile(category)`: Get profile parameters
- `get_priority_for_category(category)`: Get scheduling priority
- Feature flags: `is_adaptive_mode_enabled()`, `enable_adaptive_mode()`, `disable_adaptive_mode()`

### 5. Request Routing ✅

**Modified**: `src/chatterbox_vllm/tts_async.py`, `src/chatterbox_vllm/tts_streaming.py`

Changes:
- Import adaptive configuration functions
- Classify requests in `generate_with_conds()`
- Track category per request for metrics

Note: Full multi-engine routing (separate engine per category) is the next step
for full implementation. Current version supports classification and tracking.

### 6. Testing Suite ✅

#### test_adaptive_tts.py

Tests:
- Single category: `--profile short|medium|long`
- Mixed workload: `--mixed` (70% short, 20% medium, 10% long)
- All tests: `--all`

Usage:
```bash
# Test single category
uv run python test_adaptive_tts.py --profile short --num-requests 100

# Test mixed workload
uv run python test_adaptive_tts.py --mixed --num-requests 200

# Run all tests
uv run python test_adaptive_tts.py --all --enable-ttfa
```

#### test_concurrent_tts.py

Tests:
- Single concurrency stress test: `--single-concurrency N`
- Progressive stress test: `--progressive` (1 to max-concurrent)
- Duration-based test: `--duration SECONDS`

Usage:
```bash
# Stress test at 20 concurrent
uv run python test_concurrent_tts.py --single-concurrency 20

# Progressive test 1-50 concurrent
uv run python test_concurrent_tts.py --progressive --max-concurrent 50

# Duration-based test
uv run python test_concurrent_tts.py --max-concurrent 30 --duration 60
```

## Testing Phases

### Phase 1: Baseline Profiling
```bash
# Measure current performance
uv run python example-tts-streaming-ttfa.py
uv run python profile_tts_stages.py
```

### Phase 2: Single-Category Testing
```bash
uv run python test_adaptive_tts.py --profile short --num-requests 100
uv run python test_adaptive_tts.py --profile medium --num-requests 50
uv run python test_adaptive_tts.py --profile long --num-requests 20
```

### Phase 3: Mixed Workload Testing
```bash
uv run python test_adaptive_tts.py --mixed --num-requests 200
```

### Phase 4: Concurrent Load Testing
```bash
uv run python test_concurrent_tts.py --progressive --max-concurrent 50
```

## Success Criteria

| Criteria | Target | Status |
|----------|--------|--------|
| Short TTFA P95 | < 1.0s | 🟡 To be verified |
| Medium TTFA P95 | < 2.0s | 🟡 To be verified |
| Long TTFA P95 | < 4.0s | 🟡 To be verified |
| Concurrent requests | ≥ 20 | 🟡 To be verified |
| No starvation | Fair scheduling | 🟡 To be verified |
| System stability | No OOM | 🟡 To be verified |

## Next Steps

### Immediate (Testing & Validation)
1. Run Phase 1 profiling to establish baseline
2. Run Phase 2 single-category tests
3. Run Phase 3 mixed workload tests
4. Run Phase 4 concurrent stress tests

### Future Enhancements
1. **Multi-engine architecture**: Implement separate AsyncLLMEngine instances per category
2. **Request router**: Load balancer to route requests to appropriate engine
3. **Dynamic scaling**: Auto-adjust based on load patterns
4. **Cache optimization**: Implement prefix caching for common phrases
5. **Priority queue**: Ensure short requests are prioritized under load

## Rollback Plan

- Feature flag: `CHATTERBOX_ADAPTIVE_MODE` environment variable
- Disable: Set `CHATTERBOX_ADAPTIVE_MODE=false`
- Revert: Use original `ChatterboxTTSAsync` without `enable_ttfa_tracking=True`

## Files Created

1. `src/chatterbox_vllm/profiling.py` - TTFA tracking utilities
2. `src/chatterbox_vllm/adaptive_config.py` - Configuration profiles
3. `profile_tts_stages.py` - Stage profiling script
4. `test_adaptive_tts.py` - Adaptive testing suite
5. `test_concurrent_tts.py` - Concurrency stress test

## Files Modified

1. `src/chatterbox_vllm/tts_async.py` - Added TTFA tracking
2. `src/chatterbox_vllm/tts_streaming.py` - Added adaptive config support

## Key Metrics to Monitor

- **TTFA P50/P95/P99**: By category (short/medium/long)
- **Throughput**: Requests per second
- **Concurrency**: Max concurrent without degradation
- **Queue time**: Time waiting for processing
- **Stage breakdown**: Tokenization vs T3 vs S3 time

## Environment Variables

```bash
# Enable/disable adaptive mode
export CHATTERBOX_ADAPTIVE_MODE=true

# Default profile when adaptive mode disabled
export CHATTERBOX_DEFAULT_PROFILE=medium

# Enable TTFA tracking
export CHATTERBOX_ENABLE_TTFA_TRACKING=true
```
