#!/usr/bin/env python3
"""Example of streaming TTS with vLLM backend"""

import os
import torchaudio as ta
from chatterbox_vllm.tts import ChatterboxTTS

# Set GPU before any CUDA operations
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    # Load model with explicit GPU memory settings
    model = ChatterboxTTS.from_pretrained(
        max_batch_size=3,
        max_model_len=1000,
        gpu_memory_utilization=0.90,  # Use 90% of GPU 0's memory
    )

    # Test text
    text = (
        "This is a streaming demo of the Chatterbox TTS model running on vLLM. "
        "Audio chunks will be generated incrementally as speech tokens are produced."
    )

    # Generate streaming audio
    print(f"Generating streaming audio for: {text}\n")

    audio_chunks = []
    for audio_chunk, metrics in model.generate_stream(
        text=text,
        chunk_size=25,  # tokens per chunk
        context_window=50,
        print_metrics=True,
    ):
        audio_chunks.append(audio_chunk)
        print(f"Received chunk {metrics.chunk_count}: shape={audio_chunk.shape}, "
              f"duration={audio_chunk.shape[-1] / model.sr:.3f}s")

    # Combine and save
    if audio_chunks:
        import torch
        full_audio = torch.cat(audio_chunks, dim=-1)
        output_path = "test-streaming-vllm.wav"
        ta.save(output_path, full_audio, model.sr)
        print(f"\nSaved streaming audio to {output_path}")
        print(f"Total chunks: {len(audio_chunks)}")
        print(f"Final duration: {full_audio.shape[-1] / model.sr:.2f}s")

    model.shutdown()
