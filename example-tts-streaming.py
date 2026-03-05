#!/usr/bin/env python3
"""
Streaming TTS example for Chatterbox vLLM.

This example demonstrates how to stream audio chunks as they are generated,
which is useful for real-time TTS applications where you want to start playing
audio before the entire generation is complete.
"""

from typing import Generator, Optional
import torch
import torchaudio as ta
import numpy as np
from pathlib import Path
from chatterbox_vllm.tts import ChatterboxTTS


def stream_audio_chunks(
    model: ChatterboxTTS,
    prompt: str,
    audio_prompt_path: Optional[str] = None,
    language_id: str = 'en',
    exaggeration: float = 0.5,
    temperature: float = 0.8,
    chunk_size_samples: int = 24000,  # 1 second at 24kHz
    **generation_kwargs
) -> Generator[torch.Tensor, None, None]:
    """
    Stream audio chunks during generation.

    Args:
        model: ChatterboxTTS instance
        prompt: Text to synthesize
        audio_prompt_path: Optional reference audio path
        language_id: Language code (e.g., 'en', 'fr', 'es')
        exaggeration: Emotion exaggeration (0.5 is neutral)
        temperature: Sampling temperature
        chunk_size_samples: Number of samples per chunk (default: 24000 = 1s at 24kHz)
        **generation_kwargs: Additional arguments for model.generate_with_conds

    Yields:
        Audio chunks as torch.Tensor of shape [1, chunk_size]
    """
    # Get audio conditionals
    s3gen_ref, cond_emb = model.get_audio_conditionals(audio_prompt_path)
    cond_emb = model.update_exaggeration(cond_emb, exaggeration)

    # Import for text normalization
    from chatterbox_vllm.text_utils import punc_norm

    # Normalize and tokenize text
    from chatterbox_vllm.models.t3 import SPEECH_TOKEN_OFFSET
    normalized_prompt = "[START]" + punc_norm(prompt) + "[STOP]"

    if model.variant == "multilingual":
        normalized_prompt = f"<{language_id.lower()}>{normalized_prompt}"

    # Generate speech tokens using vLLM
    from vllm import SamplingParams
    batch_results = model.t3.generate(
        [
            {
                "prompt": normalized_prompt,
                "multi_modal_data": {
                    "conditionals": [cond_emb],
                },
            }
        ],
        sampling_params=SamplingParams(
            temperature=temperature,
            stop_token_ids=[model.t3_config.stop_speech_token + SPEECH_TOKEN_OFFSET],
            max_tokens=min(generation_kwargs.get('max_tokens', 1000), model.max_model_len),
            top_p=generation_kwargs.get('top_p', 1.0),
            repetition_penalty=generation_kwargs.get('repetition_penalty', 2.0),
        )
    )

    # Process and stream audio chunks
    from chatterbox_vllm.models.s3tokenizer import drop_invalid_tokens

    for batch_result in batch_results:
        for output in batch_result.outputs:
            speech_tokens = torch.tensor(
                [token - SPEECH_TOKEN_OFFSET for token in output.token_ids],
                device="cuda"
            )
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens[speech_tokens < 6561]

            # Generate audio from speech tokens
            diffusion_steps = generation_kwargs.get('diffusion_steps', 10)
            wav, _ = model.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=s3gen_ref,
                n_timesteps=diffusion_steps,
            )

            # Stream in chunks
            total_samples = wav.shape[1]
            for start_idx in range(0, total_samples, chunk_size_samples):
                end_idx = min(start_idx + chunk_size_samples, total_samples)
                chunk = wav[:, start_idx:end_idx]
                yield chunk


def save_streamed_audio(stream: Generator[torch.Tensor, None, None], output_path: str, model_sr: int):
    """
    Concatenate streamed chunks and save to file.

    Args:
        stream: Generator yielding audio chunks
        output_path: Path to save the output audio
        model_sr: Sample rate of the model
    """
    chunks = list(stream)
    if chunks:
        full_audio = torch.cat(chunks, dim=1)
        ta.save(output_path, full_audio, model_sr)
        print(f"Saved streamed audio to {output_path}")
    else:
        print("No audio chunks received!")


def simulate_realtime_playback(stream: Generator[torch.Tensor, None, None], model_sr: int):
    """
    Simulate real-time audio playback by processing chunks as they arrive.

    This is a demonstration of how you might handle streaming audio in a real application.

    Args:
        stream: Generator yielding audio chunks
        model_sr: Sample rate of the model
    """
    import time
    chunk_duration = 0
    total_samples = 0

    for chunk in stream:
        chunk_samples = chunk.shape[1]
        total_samples += chunk_samples
        chunk_duration = chunk_samples / model_sr

        print(f"Received chunk: {chunk_samples} samples ({chunk_duration:.2f}s)")
        print(f"Total audio so far: {total_samples} samples ({total_samples/model_sr:.2f}s)")

        # In a real application, you would play the chunk here
        # For now, just simulate the timing
        # time.sleep(chunk_duration * 0.5)  # Simulate faster-than-realtime processing


if __name__ == "__main__":
    print("Initializing Chatterbox TTS streaming model...")
    model = ChatterboxTTS.from_pretrained(
        max_batch_size=3,
        max_model_len=1000,
    )

    # Test prompts
    prompts = [
        "This is a demonstration of streaming text to speech using the Chatterbox model.",
        "Audio chunks are generated and yielded in real-time, allowing for immediate playback.",
        "The streaming approach reduces latency and improves user experience in interactive applications.",
    ]

    audio_prompt_path = None  # Or use "docs/audio-sample-01.mp3" for voice cloning

    print("\n" + "="*60)
    print("Example 1: Save streamed audio to file")
    print("="*60)

    for i, prompt in enumerate(prompts):
        print(f"\nGenerating audio for prompt {i+1}: {prompt}")

        # Create streaming generator
        stream = stream_audio_chunks(
            model=model,
            prompt=prompt,
            audio_prompt_path=audio_prompt_path,
            chunk_size_samples=24000,  # 1 second chunks
            temperature=0.8,
            exaggeration=0.5,
        )

        # Save streamed audio
        output_path = f"test-streaming-{i}.mp3"
        save_streamed_audio(stream, output_path, model.sr)

    print("\n" + "="*60)
    print("Example 2: Simulate real-time playback")
    print("="*60)

    prompt = "This is a longer example to demonstrate the streaming capabilities. Notice how chunks are processed progressively, simulating real-time audio generation."
    print(f"\nGenerating audio with real-time simulation: {prompt}")

    stream = stream_audio_chunks(
        model=model,
        prompt=prompt,
        audio_prompt_path=audio_prompt_path,
        chunk_size_samples=12000,  # 0.5 second chunks for more frequent updates
        temperature=0.8,
        exaggeration=0.5,
    )

    simulate_realtime_playback(stream, model.sr)

    print("\n" + "="*60)
    print("Example 3: Stream to numpy array for web/audio APIs")
    print("="*60)

    prompt = "This example shows how to collect chunks into a format suitable for web audio APIs."
    print(f"\nGenerating: {prompt}")

    stream = stream_audio_chunks(
        model=model,
        prompt=prompt,
        audio_prompt_path=audio_prompt_path,
        chunk_size_samples=24000,
        temperature=0.8,
        exaggeration=0.5,
    )

    # Collect chunks and convert to numpy
    chunks = []
    for chunk in stream:
        chunks.append(chunk.cpu().numpy())

    if chunks:
        full_audio = np.concatenate(chunks, axis=1).squeeze(0)
        print(f"Collected audio: {full_audio.shape} samples, dtype={full_audio.dtype}")
        print(f"Duration: {len(full_audio)/model.sr:.2f} seconds")

        # Save for verification
        ta.save("test-streaming-numpy.mp3", torch.from_numpy(full_audio).unsqueeze(0), model.sr)

    model.shutdown()
    print("\nStreaming examples complete!")
