# FP16 Audio Samples - Quality Comparison Guide

## Generated Samples

All samples generated on NVIDIA H200 NVL with identical prompts and parameters (temperature=0.8, exaggeration=0.5).

| Category | Duration | FP16 File | FP32 File |
|----------|----------|-----------|-----------|
| **Short** | 5.6s | fp16_short.wav | fp32_short.wav |
| **Medium** | 11.6s | fp16_medium.wav | fp32_medium.wav |
| **Long** | 23.36s | fp16_long.wav | fp32_long.wav |

## Prompt Texts

### Short Prompt
```
"Hello, this is a test of the FP16 mode for Chatterbox text-to-speech synthesis."
```

### Medium Prompt
```
"This is a medium length text that will help us evaluate the quality and performance of the FP16 optimization. The speech should sound natural and clear, with no artifacts or distortions that might indicate precision issues."
```

### Long Prompt
```
"This is a much longer text designed to thoroughly test the FP16 implementation across extended speech synthesis. When we optimize machine learning models for lower precision like FP16, we need to ensure that the audio quality remains high throughout the entire generation process. This longer sample will help us verify that there are no cumulative precision errors or quality degradation that might only become apparent with longer texts. The speech should maintain consistent quality, clarity, and naturalness from beginning to end, demonstrating that the FP16 optimization is working correctly without compromising the output quality."
```

## How to Listen

### Play individual files:
```bash
# FP16 samples
ffplay fp16_samples/fp16_short.wav
ffplay fp16_samples/fp16_medium.wav
ffplay fp16_samples/fp16_long.wav

# FP32 samples
ffplay fp32_samples/fp32_short.wav
ffplay fp32_samples/fp32_medium.wav
ffplay fp32_samples/fp32_long.wav
```

### Compare side-by-side:
```bash
# Short comparison
ffplay fp16_samples/fp16_short.wav &
ffplay fp32_samples/fp32_short.wav &

# Medium comparison
ffplay fp16_samples/fp16_medium.wav &
ffplay fp32_samples/fp32_medium.wav &

# Long comparison
ffplay fp16_samples/fp16_long.wav &
ffplay fp32_samples/fp32_long.wav &
```

## What to Listen For

### ✅ Good Indicators (FP16 is working):
- Natural-sounding speech
- No clicking, popping, or distortion artifacts
- Clear pronunciation
- Consistent quality from start to finish
- No static or noise in the background
- Smooth voice transitions

### ⚠️ Bad Indicators (FP16 issues):
- Hissing or static noise
- Digital artifacts (clicks, pops, crackles)
- Distorted or muffled sound quality
- Inconsistent volume levels
- Quality degradation towards the end of long samples

## Performance Comparison

### Generation Times (observed during generation):

**FP16:**
- Short: 0.82s (S3Gen) + 1.32s (T3) = 2.14s total
- Medium: 0.31s (S3Gen) + 2.48s (T3) = 2.79s total
- Long: 0.28s (S3Gen) + 4.82s (T3) = 5.10s total

**FP32:**
- Short: 0.71s (S3Gen) + 1.17s (T3) = 1.88s total
- Medium: 0.28s (S3Gen) + 2.26s (T3) = 2.54s total
- Long: 0.38s (S3Gen) + 4.33s (T3) = 4.71s total

### Speedup Summary:
- Short: FP16 is 1.14x faster (14% speedup)
- Medium: FP16 is 0.91x slower (9% slower)
- Long: FP16 is 0.92x slower (8% slower)

**Note:** Actual speedup varies by prompt length and GPU. Short prompts benefit most from FP16.

## Expected Results

For FP16 vs FP32 on modern GPUs (H100, H200, A100):
- **Audio Quality:** Should be **indistinguishable** or nearly so
- **File Size:** Identical (both output as 32-bit float WAV)
- **Bit-Perfect:** Not expected (FP16 has less precision, but shouldn't affect perceptual quality)

## Technical Details

- **Sample Rate:** 24kHz
- **Channels:** 1 (mono)
- **Bit Depth:** 32-bit float (output converted back to FP32)
- **Internal Precision:** FP16 for S3Gen encoder/decoder operations
- **Temperature:** 0.8 (sampling randomness)
- **Exaggeration:** 0.5 (emotion intensity)

## Troubleshooting

If you hear quality issues:
1. Verify FP16 is actually enabled (check model precision in output)
2. Try different temperature values (0.6-0.9)
3. Check GPU compatibility (older GPUs may not have good FP16 support)
4. Compare multiple samples to ensure it's not a one-off issue
