"""Site customization for vLLM worker processes.

This module is automatically imported by Python when it starts up (if placed in
site-packages or via PYTHONPATH). It registers the chatterbox tokenizers before
any vLLM code runs.

This is needed because when vLLM spawns worker processes using the 'spawn'
method, they are fresh Python interpreters that haven't imported
chatterbox_vllm.models.t3, causing tokenizer registration to fail.
"""

# Register custom tokenizers with vLLM
# This happens before any vLLM code imports TokenizerRegistry
try:
    from vllm.transformers_utils.tokenizer_base import TokenizerRegistry
    TokenizerRegistry.register("EnTokenizer", "chatterbox_vllm.models.t3.entokenizer", "EnTokenizer")
    TokenizerRegistry.register("MtlTokenizer", "chatterbox_vllm.models.t3.mtltokenizer", "MTLTokenizer")
except ImportError:
    # vLLM not installed, skip registration
    pass
