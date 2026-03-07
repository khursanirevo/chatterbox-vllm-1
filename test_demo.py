#!/usr/bin/env python3
"""
Chatterbox vLLM Demo Script

Unified test script for TTS and Voice Conversion streaming.
Supports both streaming and non-streaming generation with real-time audio playback.
"""

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import torch
import torchaudio as ta

# Try to import audio playback library
try:
    import sounddevice as sd
    import numpy as np
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


class ContinuousAudioPlayer:
    """Continuous audio player that prevents chunk cutoffs"""
    def __init__(self, sample_rate, buffer_size=8192):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.audio_buffer = np.array([], dtype=np.float32)
        self.stream = None
        self.playing = False
        self.lock = threading.Lock()

    def start(self):
        if not AUDIO_AVAILABLE:
            return

        def audio_callback(outdata, frames, time_info, status):
            with self.lock:
                if len(self.audio_buffer) >= frames:
                    outdata[:, 0] = self.audio_buffer[:frames]
                    self.audio_buffer = self.audio_buffer[frames:]
                else:
                    available = len(self.audio_buffer)
                    outdata[:available, 0] = self.audio_buffer
                    outdata[available:, 0] = 0
                    self.audio_buffer = np.array([], dtype=np.float32)

        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=audio_callback,
            blocksize=self.buffer_size
        )
        self.stream.start()
        self.playing = True

    def add_audio(self, audio_chunk):
        if not AUDIO_AVAILABLE or not self.playing:
            return
        audio_np = audio_chunk.squeeze().numpy().astype(np.float32)
        with self.lock:
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_np])

    def stop(self):
        if self.stream and self.playing:
            while len(self.audio_buffer) > 0:
                time.sleep(0.1)
            self.stream.stop()
            self.stream.close()
            self.playing = False


def play_audio_chunk(audio_chunk, sample_rate):
    """Play audio chunk using sounddevice with proper sequencing"""
    if not AUDIO_AVAILABLE:
        return
    try:
        audio_np = audio_chunk.squeeze().numpy()
        sd.play(audio_np, sample_rate)
        sd.wait()
    except Exception as e:
        print(f"Error playing audio: {e}")


def audio_player_worker(audio_queue, sample_rate):
    """Worker thread that plays audio chunks from queue"""
    while True:
        try:
            audio_chunk = audio_queue.get(timeout=1.0)
            if audio_chunk is None:
                break
            play_audio_chunk(audio_chunk, sample_rate)
            audio_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Audio player error: {e}")


def run_demo(args):
    """Run the TTS/VC demo with specified arguments"""
    # Import ChatterboxTTS
    try:
        from chatterbox.tts import ChatterboxTTS
    except ImportError:
        print("Error: Could not import ChatterboxTTS. Make sure you're running from the correct directory.")
        sys.exit(1)

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    if not AUDIO_AVAILABLE:
        print("Note: sounddevice not available. Install with: pip install sounddevice")

    # Load model
    print("Loading model...")
    model = ChatterboxTTS.from_pretrained(device=device)

    # Validate VC mode
    if args.mode == "vc" and not args.audio_prompt:
        print("Error: Voice conversion mode requires --audio_prompt")
        sys.exit(1)

    if args.mode == "vc" and not Path(args.audio_prompt).exists():
        print(f"Error: Audio prompt file not found: {args.audio_prompt}")
        sys.exit(1)

    # Prepare generation kwargs
    gen_kwargs = {
        "text": args.text,
        "chunk_size": args.chunk_size,
        "temperature": args.temperature,
        "cfg_weight": args.cfg_weight,
        "print_metrics": True,
    }

    if args.mode == "vc":
        gen_kwargs["audio_prompt_path"] = args.audio_prompt
        print(f"Mode: Voice Conversion with reference: {args.audio_prompt}")
    else:
        gen_kwargs["exaggeration"] = args.exaggeration
        print("Mode: TTS")

    # Test 1: Non-streaming generation
    if not args.skip_non_streaming:
        print("\n" + "=" * 60)
        print("TEST 1: Non-streaming generation")
        print("=" * 60)
        try:
            kwargs = gen_kwargs.copy()
            kwargs.pop("chunk_size", None)
            kwargs.pop("print_metrics", None)
            wav = model.generate(**kwargs)
            output_path = Path(args.output) or Path("test-non-streaming.wav")
            ta.save(output_path, wav, model.sr)
            print(f"Saved to: {output_path}")
            print(f"Duration: {wav.shape[-1] / model.sr:.2f}s")
        except Exception as e:
            print(f"Error: {e}")

    # Test 2: Streaming generation
    print("\n" + "=" * 60)
    print("TEST 2: Streaming generation")
    print("=" * 60)

    streamed_chunks = []
    chunk_count = 0
    audio_queue = None
    audio_thread = None

    # Setup audio playback
    if AUDIO_AVAILABLE and args.play_audio:
        audio_queue = queue.Queue()
        audio_thread = threading.Thread(
            target=audio_player_worker,
            args=(audio_queue, model.sr)
        )
        audio_thread.daemon = True
        audio_thread.start()
        print("Real-time audio playback enabled!")

    try:
        start_time = time.time()
        first_chunk_time = None

        for audio_chunk, metrics in model.generate_stream(**gen_kwargs):
            chunk_count += 1
            streamed_chunks.append(audio_chunk)

            if first_chunk_time is None:
                first_chunk_time = time.time()
                time_to_first = (first_chunk_time - start_time) * 1000
                print(f"⚡ First audio chunk: {time_to_first:.1f}ms")

            if AUDIO_AVAILABLE and audio_queue:
                audio_queue.put(audio_chunk.clone())

            chunk_duration = audio_chunk.shape[-1] / model.sr
            print(f"Chunk {chunk_count}: {chunk_duration:.3f}s, shape: {audio_chunk.shape}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during streaming: {e}")

    # Stop audio thread
    if audio_queue:
        audio_queue.join()
        audio_queue.put(None)

    # Save streaming output
    if streamed_chunks:
        full_audio = torch.cat(streamed_chunks, dim=-1)
        output_path = Path(args.output) or Path("test-streaming.wav")
        ta.save(output_path, full_audio, model.sr)

        print("\n" + "=" * 60)
        print("STREAMING COMPLETE")
        print("=" * 60)
        print(f"Output: {output_path}")
        print(f"Chunks: {len(streamed_chunks)}")
        print(f"Duration: {full_audio.shape[-1] / model.sr:.2f}s")
        if first_chunk_time:
            print(f"Time to first chunk: {(first_chunk_time - start_time) * 1000:.1f}ms")
    else:
        print("No audio chunks generated!")


def main():
    parser = argparse.ArgumentParser(
        description="Chatterbox vLLM Demo - TTS and Voice Conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # TTS with streaming
  python test_demo.py "Hello world, this is a test."

  # Voice conversion with reference audio
  python test_demo.py --mode vc --audio_prompt reference.wav "This is voice conversion."

  # TTS with custom settings
  python test_demo.py --temperature 0.9 --chunk-size 30 "Test with higher temperature."

  # Skip non-streaming test
  python test_demo.py --skip-non-streaming "Streaming only test."
        """
    )

    # Text input
    parser.add_argument(
        "text",
        nargs="?",
        default="Hello world, this is a test of the streaming TTS system.",
        help="Text to synthesize"
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["tts", "vc"],
        default="tts",
        help="Generation mode: tts (default) or vc (voice conversion)"
    )

    # Audio prompt for VC
    parser.add_argument(
        "--audio-prompt",
        help="Path to reference audio file (required for VC mode)"
    )

    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=25,
        help="Tokens per chunk (default: 25)"
    )

    parser.add_argument(
        "--cfg-weight",
        type=float,
        default=0.5,
        help="Classifier-free guidance weight (default: 0.5)"
    )

    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        help="Prosody exaggeration (TTS mode only, default: 0.5)"
    )

    # Output options
    parser.add_argument(
        "--output", "-o",
        help="Output WAV file path (default: test-streaming.wav)"
    )

    parser.add_argument(
        "--skip-non-streaming",
        action="store_true",
        help="Skip non-streaming generation test"
    )

    parser.add_argument(
        "--no-play-audio",
        action="store_true",
        help="Disable real-time audio playback"
    )

    parser.add_argument(
        "--play-audio",
        action="store_true",
        default=True,
        help=argparse.SUPPRESS  # Hidden option to enable by default
    )

    args = parser.parse_args()

    # Handle the flag logic
    if hasattr(args, 'no_play_audio') and args.no_play_audio:
        args.play_audio = False

    run_demo(args)


if __name__ == "__main__":
    main()
