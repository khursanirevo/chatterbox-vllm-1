#!/usr/bin/env python3
"""Profile streaming TTS to show detailed timing breakdown"""

import os
import torch
import torchaudio as ta
from chatterbox_vllm.tts import ChatterboxTTS

# Set GPU before any CUDA operations
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    print("="*70)
    print("STREAMING TTS PROFILING")
    print("="*70)

    print("\nLoading model...")
    model = ChatterboxTTS.from_pretrained(
        max_batch_size=3,
        max_model_len=2000,
        gpu_memory_utilization=0.90,
    )
    print("Model loaded!\n")

    # Test with medium text
    text = (
        "This is a profiling test for the streaming TTS implementation. "
        "We will measure exactly how much time each stage takes, from text "
        "tokenization through T3 speech token generation to the first S3Gen "
        "audio chunk. This helps identify bottlenecks and optimize the pipeline."
    )

    print(f"Text: {text}\n")
    print("="*70)
    print("GENERATING WITH DETAILED PROFILING")
    print("="*70)

    audio_chunks = []
    for audio_chunk, metrics in model.generate_stream(
        text=text,
        max_tokens=1500,
        chunk_size=25,
        context_window=50,
        print_metrics=True,
    ):
        audio_chunks.append(audio_chunk)

        # Print per-chunk progress
        if metrics.chunk_count == 1:
            print(f"\n[PROGRESS] Received chunk 1: shape={audio_chunk.shape}, "
                  f"duration={audio_chunk.shape[-1]/model.sr:.3f}s")
        elif metrics.chunk_count % 5 == 0:
            print(f"[PROGRESS] Received chunk {metrics.chunk_count}: "
                  f"last_chunk={metrics.last_chunk_time*1000:.1f}ms, "
                  f"avg_chunk={metrics.avg_chunk_time*1000:.1f}ms")

    # Save result
    if audio_chunks:
        full_audio = torch.cat(audio_chunks, dim=-1)
        output_path = "test-profiling.wav"
        ta.save(output_path, full_audio, model.sr)

        print(f"\n{'='*70}")
        print(f"SAVED: {output_path}")
        print(f"{'='*70}")
        print(f"Duration: {full_audio.shape[-1]/model.sr:.2f}s")
        print(f"Chunks: {len(audio_chunks)}")
        print(f"File size: {os.path.getsize(output_path)/1024:.1f}KB")

    model.shutdown()
    print("\nDone!")
