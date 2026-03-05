# Poisson Load Profiling Summary

**Timestamp:** 2026-03-05T19:06:16.279885
**GPU:** NVIDIA H200 NVL
**Rate:** 2.0 requests/second
**Total Requests:** 50
**Test Duration:** 11.77s

## Overall Statistics

| Metric | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) |
|--------|-----------|-------------|----------|----------|
| Queue Time (wait before processing) | 0.06 | 0.04 | 0.10 | 0.35 |
| Processing Time (T3 + S3Gen) | 6440.74 | 6302.18 | 10909.98 | 11317.04 |
| Total Latency (queue + processing) | 6440.79 | 6302.22 | 10910.02 | 11317.08 |

**Throughput:** 4.25 requests/second
**Target Rate:** 2.0 requests/second
**Achieved:** 212.4% of target

## Breakdown by Text Length

| Category | Count | Queue (ms) | Processing (ms) | Total (ms) |
|----------|-------|-----------|----------------|----------|
| Short (≤20) | 17 | 0.05 | 4606.13 | 4606.18 |
| Medium (21-50) | 19 | 0.04 | 5363.02 | 5363.06 |
| Very Long (100+) | 14 | 0.09 | 10131.07 | 10131.16 |
