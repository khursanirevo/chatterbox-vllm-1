"""
TensorRT wrapper for S3Gen ConditionalDecoder inference.

This module provides a wrapper class that loads and executes TensorRT engines
for the ConditionalDecoder model, providing 2-3x speedup over PyTorch.

Usage:
    # Initialize with TensorRT engine
    from s3gen_tensorrt_wrapper import TensorRTDecoder

    trt_decoder = TensorRTDecoder(engine_path="s3gen_decoder.engine")

    # Use as drop-in replacement for PyTorch model
    output = trt_decoder.forward(x, mask, mu, t, spks, cond)
"""

import logging
import threading
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TensorRTDecoder:
    """
    TensorRT wrapper for ConditionalDecoder forward pass.

    This class loads a TensorRT engine and provides a forward() method
    compatible with the PyTorch ConditionalDecoder interface.
    """

    def __init__(
        self,
        engine_path: str,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize TensorRT decoder.

        Args:
            engine_path: Path to TensorRT engine file (.engine)
            device: CUDA device to use
        """
        self.engine_path = Path(engine_path)
        self.device = device or torch.device("cuda")

        # Lock for thread-safe TensorRT execution
        self.lock = threading.Lock()

        # Load TensorRT engine
        self._load_engine()

        # Get input/output specs from engine
        self._parse_engine_specs()

        logger.info(f"[TensorRT] Engine loaded: {self.engine_path}")
        logger.info(f"[TensorRT] Device: {self.device}")
        logger.info(f"[TensorRT] Inputs: {self.input_specs}")
        logger.info(f"[TensorRT] Outputs: {self.output_specs}")

    def _load_engine(self):
        """Load TensorRT engine from file."""
        import tensorrt as trt

        TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(TRT_LOGGER)

        if not self.engine_path.exists():
            raise FileNotFoundError(
                f"TensorRT engine not found: {self.engine_path}\n"
                f"Please build the engine first using:\n"
                f"  python build_s3gen_tensorrt.py --export-onnx"
            )

        logger.info(f"[TensorRT] Loading engine from {self.engine_path}...")
        with open(self.engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")

        # Create execution context
        self.context = self.engine.create_execution_context()

        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

    def _parse_engine_specs(self):
        """Parse input/output specifications from TensorRT engine."""
        self.input_specs = {}
        self.output_specs = {}

        # Get input names and shapes
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name, 0)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)

            spec = {
                "name": name,
                "shape": shape,
                "mode": mode,
                "dtype": dtype,
            }

            if self.engine.get_tensor_mode(name, 0) == trt.TensorIOMode.INPUT:
                self.input_specs[name] = spec
            else:
                self.output_specs[name] = spec

    def set_input_shape(self, name: str, shape: Tuple[int, ...]):
        """
        Set dynamic input shape for inference.

        Args:
            name: Input tensor name
            shape: Input shape (batch_size, channels, length)
        """
        with self.lock:
            if not self.context.set_input_shape(name, shape):
                raise RuntimeError(f"Failed to set input shape '{name}' to {shape}")

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through TensorRT engine.

        Args:
            x: Input tensor (batch_size, in_channels, n_mels)
            mask: Mask tensor (batch_size, 1, n_mels)
            mu: Mu tensor (batch_size, n_mels, n_mels)
            t: Timestep tensor (batch_size,)
            spks: Speaker embedding (batch_size, spk_emb_dim)
            cond: Condition tensor (batch_size, n_feats, n_mels)

        Returns:
            Output tensor (batch_size, out_channels, n_mels)
        """
        import tensorrt as trt

        # Ensure all inputs are contiguous and on correct device
        x = x.contiguous().to(self.device)
        mask = mask.contiguous().to(self.device)
        mu = mu.contiguous().to(self.device)
        t = t.contiguous().to(self.device)
        spks = spks.contiguous().to(self.device)
        cond = cond.contiguous().to(self.device)

        batch_size, _, n_mels = x.shape

        # Set dynamic input shapes if needed
        with self.lock:
            # Check if shapes match engine expectations
            for name, tensor in [
                ("x", x),
                ("mask", mask),
                ("mu", mu),
                ("t", t),
                ("spks", spks),
                ("cond", cond),
            ]:
                expected_shape = self.input_specs[name]["shape"]

                # Handle dynamic shapes (-1 indicates dynamic dimension)
                if -1 in expected_shape:
                    # Set actual shape
                    actual_shape = list(tensor.shape)
                    self.context.set_input_shape(name, actual_shape)

            # Allocate output buffer
            output_spec = list(self.output_specs.values())[0]
            output_shape = list(output_spec["shape"])

            # Fix dynamic dimensions in output shape
            output_shape[0] = batch_size  # batch_size
            if -1 in output_shape:
                output_shape[2] = n_mels  # sequence length

            output = torch.empty(output_shape, dtype=torch.float32, device=self.device)

            # Get device pointers
            bindings = [
                x.data_ptr(),
                mask.data_ptr(),
                mu.data_ptr(),
                t.data_ptr(),
                spks.data_ptr(),
                cond.data_ptr(),
                output.data_ptr(),
            ]

            # Execute TensorRT engine
            self.context.execute_v2(bindings=bindings)

        return output

    def __call__(self, *args, **kwargs):
        """Allow calling the wrapper like a PyTorch module."""
        return self.forward(*args, **kwargs)


def load_tensorrt_engine(
    engine_path: str,
    device: Optional[torch.device] = None,
) -> TensorRTDecoder:
    """
    Load TensorRT engine for S3Gen decoder.

    This is a convenience function that creates a TensorRTDecoder instance.

    Args:
        engine_path: Path to TensorRT engine file
        device: CUDA device to use

    Returns:
        TensorRTDecoder instance

    Example:
        >>> decoder = load_tensorrt_engine("s3gen_decoder.engine")
        >>> output = decoder(x, mask, mu, t, spks, cond)
    """
    return TensorRTDecoder(engine_path, device=device)


def is_tensorrt_available() -> bool:
    """Check if TensorRT is available."""
    try:
        import tensorrt as trt
        return True
    except ImportError:
        return False
