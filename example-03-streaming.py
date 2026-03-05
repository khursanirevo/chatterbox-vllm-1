#!/usr/bin/env python3
"""
Token-level streaming demo for ChatterboxTTSStreaming.
"""
import asyncio
import torch
import torchaudio as ta
from chatterbox_vllm import ChatterboxTTSStreaming

async def main():
    model = await ChatterboxTTSStreaming.from_pretrained()

    prompt = "This is a demonstration of token-level streaming for faster time to first audio."

    print(f"Generating: {prompt}")
    print("Audio chunks will appear as they're generated...\n")

    chunks = []
    async for chunk in model.stream_audio_tokens(prompt):
        print(f"  Received chunk: {chunk.shape} samples")
        chunks.append(chunk.cpu())

    # Combine and save
    full_audio = torch.cat(chunks, dim=1)
    output_path = "demo-streaming.mp3"
    ta.save(output_path, full_audio, model.sr)
    print(f"\nSaved: {output_path}")

    await model.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
