"""
CUDA MPS Worker for S3Gen parallel inference.

This module provides worker functions that can be pickled for multiprocessing.
Each worker process loads its own S3Gen model instance and processes requests
independently, enabling true parallelism with CUDA MPS.
"""
import os
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
from safetensors.torch import load_file


# Global state for worker processes (persists across tasks)
_model = None
_device = None
_use_fp16 = None
_compile_model = None
_worker_id = None


def _init_worker(ckpt_dir: str, use_fp16: bool, compile_model: bool, device: str):
    """
    Initialize worker process by loading S3Gen model.

    This function is called once per worker when the multiprocessing pool is created.
    It loads the S3Gen model into global state for this process.

    Args:
        ckpt_dir: Path to checkpoint directory containing s3gen.safetensors
        use_fp16: Whether to use FP16 for S3Gen
        compile_model: Whether to compile S3Gen with torch.compile()
        device: Device string (e.g., "cuda:0")
    """
    global _model, _device, _use_fp16, _compile_model, _worker_id

    try:
        import time
        _device = device
        _use_fp16 = use_fp16
        _compile_model = compile_model
        _worker_id = os.getpid()

        print(f"[S3Gen Worker {_worker_id}] Starting initialization on {device}")
        t0 = time.time()

        # Import here to avoid circular imports
        print(f"[S3Gen Worker {_worker_id}] Importing S3Gen...")
        t1 = time.time()
        from chatterbox_vllm.models.s3gen import S3Gen
        print(f"[S3Gen Worker {_worker_id}] Import done in {time.time()-t1:.1f}s")

        # Load S3Gen model
        ckpt_path = Path(ckpt_dir) / "s3gen.safetensors"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"S3Gen checkpoint not found: {ckpt_path}")

        print(f"[S3Gen Worker {_worker_id}] Creating model...")
        t2 = time.time()
        _model = S3Gen(
            use_fp16=use_fp16,
            compile_model=compile_model,
        )
        print(f"[S3Gen Worker {_worker_id}] Model created in {time.time()-t2:.1f}s")

        print(f"[S3Gen Worker {_worker_id}] Loading weights from {ckpt_path}...")
        t3 = time.time()
        state_dict = load_file(ckpt_path)
        _model.load_state_dict(state_dict, strict=False)
        print(f"[S3Gen Worker {_worker_id}] Weights loaded in {time.time()-t3:.1f}s")

        print(f"[S3Gen Worker {_worker_id}] Moving to {device}...")
        t4 = time.time()
        _model = _model.to(device=device).eval()
        torch.cuda.synchronize()
        print(f"[S3Gen Worker {_worker_id}] Model on {device} in {time.time()-t4:.1f}s")

        total = time.time() - t0
        print(f"[S3Gen Worker {_worker_id}] ✓ Initialized in {total:.1f}s")

    except Exception as e:
        print(f"[S3Gen Worker] ERROR in worker {os.getpid()}: {e}")
        print(traceback.format_exc())
        raise


def _run_s3gen_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run S3Gen inference on a single task.

    This is the main worker function that processes individual inference requests.
    It must be at module level to be picklable for multiprocessing.

    Args:
        task: Dictionary containing:
            - speech_tokens: numpy array [1, T] of speech tokens
            - ref_dict: Dictionary with reference embeddings (numpy arrays)
            - n_timesteps: Number of diffusion steps
            - index: Task index for ordering results

    Returns:
        Dictionary containing:
            - wav: Generated audio as numpy array [1, T_audio]
            - index: Task index (same as input)
            - error: Error string if inference failed (optional)
    """
    global _model, _device, _worker_id

    try:
        if _model is None:
            return {
                'index': task['index'],
                'error': 'Model not initialized',
                'wav': None,
            }

        # Extract task data
        speech_tokens_np = task['speech_tokens']
        ref_dict_np = task['ref_dict']
        n_timesteps = task['n_timesteps']
        index = task['index']

        # Convert numpy arrays to torch tensors
        speech_tokens = torch.from_numpy(speech_tokens_np).to(device=_device)

        # Convert ref_dict values to tensors and move to device
        ref_dict = {}
        for k, v in ref_dict_np.items():
            if isinstance(v, np.ndarray):
                ref_dict[k] = torch.from_numpy(v).to(device=_device)
            else:
                ref_dict[k] = v

        # Run inference
        with torch.inference_mode():
            wav, _ = _model.inference(
                speech_tokens=speech_tokens,
                ref_dict=ref_dict,
                n_timesteps=n_timesteps,
            )

        # Convert result to numpy for pickling
        wav_np = wav.detach().cpu().numpy()

        return {
            'wav': wav_np,
            'index': index,
            'error': None,
        }

    except Exception as e:
        print(f"[S3Gen Worker] ERROR in task {task.get('index', '?')}: {e}")
        print(traceback.format_exc())
        return {
            'index': task.get('index', -1),
            'error': str(e),
            'wav': None,
        }


def _get_worker_status() -> Dict[str, Any]:
    """
    Get the status of the current worker process.

    Returns:
        Dictionary with worker status information
    """
    global _model, _device, _worker_id, _use_fp16, _compile_model

    return {
        'worker_id': _worker_id,
        'device': _device,
        'model_loaded': _model is not None,
        'use_fp16': _use_fp16,
        'compile_model': _compile_model,
    }
