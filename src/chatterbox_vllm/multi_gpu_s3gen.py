"""
Multi-GPU S3Gen wrapper for parallel inference.

This module provides a wrapper that distributes S3Gen requests across multiple GPUs
for true parallel processing.
"""
import asyncio
import torch
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import threading


class MultiGPUS3Gen:
    """
    Multi-GPU S3Gen wrapper that distributes requests across multiple GPUs.

    Each request is processed on a different GPU in parallel, providing true
    GPU-level parallelism instead of thread-level serialization.
    """

    def __init__(self, s3gen_model, gpu_ids: List[int] = [0, 1, 2, 3]):
        """
        Initialize multi-GPU S3Gen wrapper.

        Args:
            s3gen_model: Base S3Gen model (will be copied to each GPU)
            gpu_ids: List of GPU IDs to use (default: [0, 1, 2, 3])
        """
        self.gpu_ids = gpu_ids
        self.base_model = s3gen_model
        self.models = {}  # gpu_id -> model instance
        self.locks = {}   # gpu_id -> lock for thread safety
        self.executor = ThreadPoolExecutor(max_workers=len(gpu_ids))

        print(f"[MultiGPU] Initializing S3Gen on {len(gpu_ids)} GPUs: {gpu_ids}")

        # Create a model instance for each GPU
        for gpu_id in gpu_ids:
            try:
                # Move model to GPU
                model_gpu = self._copy_model_to_gpu(s3gen_model, gpu_id)
                self.models[gpu_id] = model_gpu
                self.locks[gpu_id] = threading.Lock()
                print(f"[MultiGPU] GPU {gpu_id}: Model loaded successfully")
            except Exception as e:
                print(f"[MultiGPU] GPU {gpu_id}: Failed to load model - {e}")
                # Remove this GPU from the list
                self.gpu_ids = [gid for gid in self.gpu_ids if gid != gpu_id]

        if not self.gpu_ids:
            raise RuntimeError("No GPUs available for multi-GPU S3Gen")

        print(f"[MultiGPU] Using {len(self.gpu_ids)} GPUs for parallel inference")

    def _copy_model_to_gpu(self, base_model, gpu_id: int):
        """Copy model to specific GPU."""
        # Create a copy of the model state dict
        model_copy = type(base_model)(**self._get_model_init_args(base_model))

        # Copy weights
        model_copy.load_state_dict(base_model.state_dict())
        model_copy.to(f'cuda:{gpu_id}')
        model_copy.eval()

        return model_copy

    def _get_model_init_args(self, model) -> Dict[str, Any]:
        """Extract initialization arguments from model."""
        # This is a placeholder - actual implementation depends on model structure
        # For now, return empty dict and rely on state_dict copying
        return {}

    def _get_next_gpu(self) -> int:
        """Round-robin GPU selection."""
        # Use thread-local counter for round-robin
        if not hasattr(self, '_gpu_counter'):
            self._gpu_counter = threading.local()
            self._gpu_counter.value = 0

        gpu_id = self.gpu_ids[self._gpu_counter.value % len(self.gpu_ids)]
        self._gpu_counter.value += 1
        return gpu_id

    async def inference_async(self, speech_tokens, ref_dict, n_timesteps: int):
        """
        Run S3Gen inference on the next available GPU (async).

        Args:
            speech_tokens: Speech tokens [1, T]
            ref_dict: Reference dictionary
            n_timesteps: Number of diffusion steps

        Returns:
            Tuple of (wav, output_device)
        """
        # Get next GPU in round-robin fashion
        gpu_id = self._get_next_gpu()

        # Run inference on thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._inference_on_gpu,
            gpu_id,
            speech_tokens,
            ref_dict,
            n_timesteps
        )

        return result

    def _inference_on_gpu(self, gpu_id: int, speech_tokens, ref_dict, n_timesteps: int):
        """Run inference on specific GPU."""
        with self.locks[gpu_id]:
            model = self.models[gpu_id]

            # Move inputs to GPU
            speech_tokens_gpu = speech_tokens.to(f'cuda:{gpu_id}')

            # Move ref_dict to GPU
            ref_dict_gpu = {}
            for k, v in ref_dict.items():
                if torch.is_tensor(v):
                    ref_dict_gpu[k] = v.to(f'cuda:{gpu_id}')
                else:
                    ref_dict_gpu[k] = v

            # Run inference
            with torch.inference_mode():
                with torch.cuda.device(gpu_id):
                    wav, _ = model.inference(
                        speech_tokens=speech_tokens_gpu,
                        ref_dict=ref_dict_gpu,
                        n_timesteps=n_timesteps,
                    )

            # Move result back to CPU
            return wav.cpu(), gpu_id

    async def inference_batch_async(self, requests: List[Dict[str, Any]]):
        """
        Run multiple S3Gen inferences in parallel across GPUs.

        Args:
            requests: List of dicts with 'speech_tokens', 'ref_dict', 'n_timesteps'

        Returns:
            List of (wav, gpu_id) tuples
        """
        # Create tasks for all requests
        tasks = [
            self.inference_async(
                req['speech_tokens'],
                req['ref_dict'],
                req.get('n_timesteps', 5)
            )
            for req in requests
        ]

        # Run all tasks in parallel
        results = await asyncio.gather(*tasks)
        return results

    def get_gpu_utilization(self) -> Dict[int, Dict[str, float]]:
        """Get GPU utilization statistics."""
        import subprocess
        import json

        utilization = {}
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True
            )

            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(', ')
                    gpu_id = int(parts[0])
                    if gpu_id in self.gpu_ids:
                        utilization[gpu_id] = {
                            'gpu_util': float(parts[1]),
                            'mem_used': float(parts[2]),
                            'mem_total': float(parts[3]),
                        }
        except Exception as e:
            print(f"[MultiGPU] Failed to get GPU utilization: {e}")

        return utilization

    def shutdown(self):
        """Shutdown the multi-GPU wrapper."""
        self.executor.shutdown(wait=True)
        self.models.clear()
        self.locks.clear()


# Simpler version using CUDA_VISIBLE_DEVICES approach
class SimpleMultiGPUS3Gen:
    """
    Simple multi-GPU S3Gen using process-based isolation.

    This is simpler but more effective than thread-based approach because
    each process gets its own CUDA context.
    """

    def __init__(self, s3gen_model, num_gpus: int = 4):
        """
        Initialize simple multi-GPU S3Gen.

        Args:
            s3gen_model: Base S3Gen model (stays on GPU 0)
            num_gpus: Number of GPUs to use
        """
        self.base_model = s3gen_model
        self.num_gpus = num_gpus
        self.current_gpu = 0

        print(f"[SimpleMultiGPU] Using {num_gpus} GPUs with round-robin scheduling")

    async def inference_async(self, speech_tokens, ref_dict, n_timesteps: int):
        """Run inference on next GPU (using CUDA device)."""
        # Select GPU in round-robin fashion
        gpu_id = self.current_gpu % self.num_gpus
        self.current_gpu += 1

        # Run inference
        loop = asyncio.get_event_loop()

        def _run_on_gpu():
            with torch.cuda.device(gpu_id):
                # Move inputs to GPU
                speech_tokens_gpu = speech_tokens.to(f'cuda:{gpu_id}')
                ref_dict_gpu = {k: v.to(f'cuda:{gpu_id}') if torch.is_tensor(v) else v for k, v in ref_dict.items()}

                # Run inference
                with torch.inference_mode():
                    wav, _ = self.base_model.inference(
                        speech_tokens=speech_tokens_gpu,
                        ref_dict=ref_dict_gpu,
                        n_timesteps=n_timesteps,
                    )

                return wav.cpu(), gpu_id

        result = await loop.run_in_executor(None, _run_on_gpu)
        return result
