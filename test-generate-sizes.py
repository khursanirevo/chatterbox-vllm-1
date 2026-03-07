#!/usr/bin/env python3
"""Generate streaming audio for short, medium, and long texts"""

import os
import torch
import torchaudio as ta
from chatterbox_vllm.tts import ChatterboxTTS

# Set GPU before any CUDA operations
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Define test texts
TEST_CASES = {
    "short": {
        "text": "Hello, this is a short test.",
        "output": "test-short.wav",
        "max_tokens": 500  # Short texts need fewer tokens
    },
    "medium": {
        "text": (
            "This is a medium-length text to test the streaming capabilities. "
            "It contains multiple sentences and should take a few seconds to generate. "
            "The streaming feature allows audio chunks to be produced incrementally."
        ),
        "output": "test-medium.wav",
        "max_tokens": 1000  # Medium texts
    },
    "long": {
        "text": (
            "This is a longer text designed to thoroughly test the streaming TTS implementation. "
            "When we have substantially more content, it allows us to observe how the system handles "
            "multiple audio chunks over an extended period. The streaming approach is particularly "
            "useful for real-time applications where users don't want to wait for the entire generation "
            "to complete before hearing the first audio. With this implementation, we use a two-stage "
            "process: first, vLLM rapidly generates all the speech tokens, and then we stream those "
            "tokens through the S3Gen model in chunks. This provides an excellent balance between "
            "the batch processing efficiency of vLLM and the real-time playback capabilities needed "
            "for interactive applications. The result is an RTF of approximately 0.7, meaning the "
            "audio generates faster than real-time playback speed."
        ),
        "output": "test-long.wav",
        "max_tokens": 2000  # Long texts need more tokens
    }
}

if __name__ == "__main__":
    print("Loading model...")
    model = ChatterboxTTS.from_pretrained(
        max_batch_size=3,
        max_model_len=2000,  # Increased to 2000 for longer texts
        gpu_memory_utilization=0.90,
    )
    print("Model loaded!\n")

    results = {}

    for size, config in TEST_CASES.items():
        text = config["text"]
        output_path = config["output"]
        max_tokens = config["max_tokens"]

        print(f"\n{'='*60}")
        print(f"Generating {size.upper()} audio (max_tokens={max_tokens})")
        print(f"{'='*60}")
        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}\n")

        audio_chunks = []
        for audio_chunk, metrics in model.generate_stream(
            text=text,
            max_tokens=max_tokens,
            chunk_size=25,
            context_window=50,
            print_metrics=True,
        ):
            audio_chunks.append(audio_chunk)

        # Combine and save
        if audio_chunks:
            full_audio = torch.cat(audio_chunks, dim=-1)
            ta.save(output_path, full_audio, model.sr)
            duration = full_audio.shape[-1] / model.sr

            results[size] = {
                "text": text,
                "output_path": output_path,
                "duration": duration,
                "chunks": len(audio_chunks),
                "rtf": metrics.rtf,
                "latency": metrics.latency_to_first_chunk
            }

            print(f"\n✓ Saved to: {output_path}")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Chunks: {len(audio_chunks)}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY - Text to Audio Mapping")
    print(f"{'='*60}\n")

    for size in ["short", "medium", "long"]:
        if size in results:
            r = results[size]
            print(f"[{size.upper()}]")
            print(f"  Text: {r['text']}")
            print(f"  Audio: {r['output_path']}")
            print(f"  Duration: {r['duration']:.2f}s | Chunks: {r['chunks']} | RTF: {r['rtf']:.3f} | Latency: {r['latency']:.3f}s")
            print()

    model.shutdown()
    print("Done!")
