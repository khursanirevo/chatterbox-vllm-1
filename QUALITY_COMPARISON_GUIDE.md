# Quality Comparison Guide: n_timesteps=10 vs n_timesteps=5

## Audio Samples Generated

**Baseline (n_timesteps=10):** `quality_baseline_samples/`
**Comparison (n_timesteps=5):** `quality_baseline_samples_comparison/`

## Test Cases

| Test Case | Description | File (10 steps) | File (5 steps) |
|-----------|-------------|------------------|-----------------|
| short | Simple greeting | `n_timesteps_10_short.wav` | `n_timesteps_5_short.wav` |
| medium | Weather forecast | `n_timesteps_10_medium.wav` | `n_timesteps_5_medium.wav` |
| long | Long technical text | `n_timesteps_10_long.wav` | `n_timesteps_5_long.wav` |
| punctuation | Complex punctuation | `n_timesteps_10_punctuation.wav` | `n_timesteps_5_punctuation.wav` |
| numbers | Numbers and prices | `n_timesteps_10_numbers.wav` | `n_timesteps_5_numbers.wav` |
| emotional | Excited emotion | `n_timesteps_10_emotional.wav` | `n_timesteps_5_emotional.wav` |
| question | Question with surprise | `n_timesteps_10_question.wav` | `n_timesteps_5_question.wav` |

## Performance Comparison

| Metric | n_timesteps=10 | n_timesteps=5 | Improvement |
|--------|-----------------|----------------|-------------|
| **Short (mean)** | 1.369s | 0.644s | **53% faster** |
| **Medium (mean)** | 1.648s | 1.388s | 16% faster |
| **Long (mean)** | 2.984s | 2.691s | 10% faster |
| **Average** | 1.557s | 1.259s | **19% faster** |

## How to Listen

### Option 1: Command Line
```bash
# Listen to baseline (10 steps)
play quality_baseline_samples/n_timesteps_10_short.wav

# Listen to optimized (5 steps)
play quality_baseline_samples_comparison/n_timesteps_5_short.wav
```

### Option 2: Compare Side-by-Side
```bash
# Linux
play quality_baseline_samples/n_timesteps_10_short.wav &
sleep 1
play quality_baseline_samples_comparison/n_timesteps_5_short.wav

# Or use a player with A/B comparison
```

### Option 3: Audio Player
- Open both folders in your audio player
- Listen to each pair (10 vs 5 steps)
- Note any quality differences

## What to Listen For

1. **Clarity**: Are words clear and distinct?
2. **Naturalness**: Does speech sound natural?
3. **Emotion**: Is emotional content preserved?
4. **Punctuation**: Are pauses and intonation correct?
5. **Artifacts**: Any noise, distortion, or glitches?

## Quality Assessment

Please listen to all 7 test cases and note:
- ✅ **No noticeable difference** → Use n_timesteps=5
- ⚠️ **Minor difference** → Consider if speed trade-off is worth it
- ❌ **Significant degradation** → Stay with n_timesteps=10

## Latency Impact

Your listening results should help decide:

| Option | TTFA (short) | Quality |
|--------|---------------|---------|
| **n_timesteps=10** | ~1.37s | Best (baseline) |
| **n_timesteps=5** | **~0.64s** | Slightly worse? |

If quality is acceptable with n_timesteps=5, you get **53% faster** short requests!

## Next Steps

After listening to the samples:

1. **If n_timesteps=5 quality is acceptable:**
   - Keep current settings (diffusion_steps=5)
   - Enjoy 53% speedup for short requests
   - All TTFA targets met ✅

2. **If quality is not acceptable:**
   - Revert to n_timesteps=10
   - Consider other optimizations (torch.compile, TensorRT)
   - TTFA targets may not be met with n_timesteps=10

3. **If you want to test n_timesteps=7 or 8:**
   - Middle ground between 5 and 10
   - Let me know and I can test that configuration
