"""Monkey-patch vLLM worker process to register custom tokenizers.

When AsyncLLMEngine spawns worker processes using multiprocessing, they are
fresh Python interpreters that haven't imported chatterbox_vllm.models.t3.
This causes tokenizer registration to fail with "Tokenizer EnTokenizer not found."

This module sets up a sitecustomize.py module that will be imported by spawned
worker processes, ensuring tokenizers are registered before vLLM code runs.
"""

import sys
import os


def apply_worker_patch():
    """Set up PYTHONPATH to include sitecustomize for worker processes.

    This adds the src/chatterbox_vllm directory to PYTHONPATH so that spawned
    worker processes will automatically import sitecustomize.py when they start,
    which registers the custom tokenizers before vLLM code runs.

    Call this BEFORE creating AsyncLLMEngine.
    """
    # Get the path to chatterbox_vllm directory
    import chatterbox_vllm
    chatterbox_path = os.path.dirname(chatterbox_vllm.__file__)

    # Add to PYTHONPATH environment variable
    # Spawned processes will inherit this and import sitecustomize.py from there
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if chatterbox_path not in current_pythonpath:
        if current_pythonpath:
            os.environ['PYTHONPATH'] = f"{chatterbox_path}:{current_pythonpath}"
        else:
            os.environ['PYTHONPATH'] = chatterbox_path

    # Also add to sys.path so sitecustomize is importable in current process
    if chatterbox_path not in sys.path:
        sys.path.insert(0, chatterbox_path)
