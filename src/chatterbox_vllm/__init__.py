#!/usr/bin/env python3
"""
Chatterbox vLLM - Text-to-Speech using vLLM for continuous batching.

This package provides both synchronous and asynchronous TTS implementations:
- ChatterboxTTS: Synchronous implementation using vLLM.LLM
- ChatterboxTTSAsync: Asynchronous implementation using vLLM.AsyncLLMEngine with continuous batching
"""

from chatterbox_vllm.tts import ChatterboxTTS
from chatterbox_vllm.tts_async import ChatterboxTTSAsync
from chatterbox_vllm.tts_streaming import ChatterboxTTSStreaming

__all__ = ["ChatterboxTTS", "ChatterboxTTSAsync", "ChatterboxTTSStreaming"]
