# S3Gen Optimization - Final Results

## Quality vs Speed Trade-off

After testing, **n_timesteps < 5 produces unacceptable audio quality**.

The optimal production setting is **n_timesteps=5**.

## Performance with n_timesteps=5

| Metric | Value |
|--------|-------|
| **Short Requests Mean** | 0.548s |
| **Short Requests P95** | 0.631s |
| **Speedup** | 1.73x (42% faster) |
| **Quality** | ✅ Acceptable |

## Success Criteria

| Category | Target | Actual (n_timesteps=5) | Status |
|----------|--------|----------------------|--------|
| Short TTFA P95 | < 1.0s | **0.631s** | ✅ PASS |
| Medium TTFA P95 | < 2.0s | **1.148s** | ✅ PASS |
| Long TTFA P95 | < 4.0s | **2.196s** | ✅ PASS |

## Implementation

Update the default `n_timesteps` parameter in:
- `src/chatterbox_vllm/tts_async.py`
- `src/chatterbox_vllm/tts_streaming.py`

From:
```python
n_timesteps=10
```

To:
```python
n_timesteps=5  # Optimal: balance of speed (1.73x) and quality
```

## Bottleneck After Optimization

| Component | % of TTFA | Notes |
|-----------|-----------|-------|
| S3Gen | ~60% | Still the main bottleneck |
| T3 Generation | ~35% | Now more significant |
| Tokenization | < 1% | Negligible |

## Future Optimizations

1. **Fix FP16 dtype mismatch** - Additional 20-30% speedup potential
2. **torch.compile()** - 30-40% speedup
3. **TensorRT optimization** - 2-3x speedup (code already has support)
4. **Reduce CFG rate** - 2x speedup (with quality tradeoff)

## Recommendation

**Use n_timesteps=5 for production** - This gives the best balance:
- 42% faster than baseline
- P95 under 1s for short requests
- Acceptable audio quality
- No degradation for medium/long requests
