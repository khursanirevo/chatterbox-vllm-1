#!/usr/bin/env python3
"""
Chatterbox vLLM - Text-to-Speech using vLLM for continuous batching.

This package provides both synchronous and asynchronous TTS implementations:
- ChatterboxTTS: Synchronous implementation using vLLM.LLM
- ChatterboxTTSAsync: Asynchronous implementation using vLLM.AsyncLLMEngine (experimental)
- ChatterboxTTSStreaming: Token-level streaming for improved TTFA (experimental)
- ChatterboxTTSAsyncWrapper: Async wrapper around ChatterboxTTS using asyncio.to_thread() (recommended)

Note: The AsyncLLMEngine-based implementations (ChatterboxTTSAsync, ChatterboxTTSStreaming)
work with CUDA via spawn-based multiprocessing. Custom tokenizer registration is handled
via sitecustomize.py which is imported by spawned worker processes.
"""

# Register custom tokenizers with vLLM
from vllm.transformers_utils.tokenizer_base import TokenizerRegistry

TokenizerRegistry.register("EnTokenizer", "chatterbox_vllm.models.t3.entokenizer", "EnTokenizer")
TokenizerRegistry.register("MtlTokenizer", "chatterbox_vllm.models.t3.mtltokenizer", "MTLTokenizer")

from chatterbox_vllm.tts import ChatterboxTTS
from chatterbox_vllm.vllm_worker_patch import apply_worker_patch

# Optional: Import wrapper if available
try:
    from chatterbox_vllm.tts_async_wrapper import ChatterboxTTSAsyncWrapper
    _has_wrapper = True
except (ImportError, RuntimeError):
    _has_wrapper = False
    ChatterboxTTSAsyncWrapper = None

# Optional: Import experimental AsyncLLMEngine-based implementations
# These require VLLM_WORKER_MULTIPROC_METHOD=fork which is incompatible with CUDA
try:
    from chatterbox_vllm.tts_async import ChatterboxTTSAsync
    from chatterbox_vllm.tts_streaming import ChatterboxTTSStreaming
    _has_async_llm = True
except (ImportError, RuntimeError):
    _has_async_llm = False
    ChatterboxTTSAsync = None
    ChatterboxTTSStreaming = None

__all__ = [
    "ChatterboxTTS",
    "apply_worker_patch",
]

if _has_wrapper:
    __all__.append("ChatterboxTTSAsyncWrapper")

if _has_async_llm:
    __all__.extend(["ChatterboxTTSAsync", "ChatterboxTTSStreaming"])
