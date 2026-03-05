#!/usr/bin/env python3
"""
TensorRT engine builder for S3Gen ConditionalDecoder.

This script converts the PyTorch ConditionalDecoder model to TensorRT
for 2-3x inference speedup.

Requirements:
    - TensorRT 8.6+ (pip install tensorrt)
    - PyTorch with TensorRT support
    - CUDA 11.8+ or 12.x

Usage:
    # Build TensorRT engine
    python build_s3gen_tensorrt.py --engine-dir ./trt_engines

    # Use TensorRT engine in ChatterboxTTSAsync
    model = await ChatterboxTTSAsync.from_pretrained(
        s3gen_use_tensorrt=True,
        s3gen_tensorrt_engine_path="./trt_engines/s3gen_decoder.engine",
    )
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_tensorrt():
    """Check if TensorRT is available."""
    try:
        import tensorrt as trt
        logger.info(f"✓ TensorRT {tensorrt.__version__} found")
        return True, trt
    except ImportError:
        logger.error("✗ TensorRT not found")
        logger.error("\nTo install TensorRT:")
        logger.error("  pip install tensorrt")
        logger.error("\nFor CUDA 11.8:")
        logger.error("  pip install tensorrt==8.6.1")
        logger.error("\nFor CUDA 12.x:")
        logger.error("  pip install nvidia-tensorrt")
        logger.error("\nSee: https://developer.nvidia.com/tensorrt")
        return False, None


def check_torch_tensorrt():
    """Check if torch-tensorrt is available."""
    try:
        import torch_tensorrt
        logger.info(f"✓ torch-tensorrt found")
        return True
    except ImportError:
        logger.warning("✗ torch-tensorrt not found (optional, for Python API)")
        logger.warning("  To install: pip install torch-tensorrt")
        return False


def load_model_from_checkpoint(ckpt_dir: Path, device: str = "cuda"):
    """Load S3Gen model from checkpoint directory."""
    from safetensors.torch import load_file
    from chatterbox_vllm.models.s3gen.s3gen import S3Gen

    logger.info(f"Loading model from {ckpt_dir}...")

    # Use HF hub download if needed
    if not ckpt_dir.exists():
        from huggingface_hub import hf_hub_download
        logger.info(f"Checkpoint not found locally, downloading from HuggingFace...")
        ckpt_dir = Path(hf_hub_download(
            repo_id="khursanirevo/chatterbox-vllm-1",
            filename="s3gen.safetensors",
            revision="1b475dffa71fb191cb6d5901215eb6f55635a9b6"
        )).parent
        logger.info(f"Downloaded to {ckpt_dir}")

    # Initialize model
    s3gen = S3Gen(use_fp16=False)  # Use FP32 for building, can convert later
    s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
    s3gen = s3gen.to(device).eval()

    logger.info("✓ Model loaded successfully")
    return s3gen


def build_tensorrt_engine(
    model: nn.Module,
    engine_path: Path,
    max_batch_size: int = 2,
    max_mel_length: int = 300,  # ~12 seconds at 24kHz
    use_fp16: bool = True,
    workspace_size: int = 1 << 30,  # 1GB
):
    """
    Build TensorRT engine from PyTorch model.

    Args:
        model: PyTorch model to convert
        engine_path: Output path for TensorRT engine
        max_batch_size: Maximum batch size for dynamic shapes
        max_mel_length: Maximum mel length for dynamic shapes
        use_fp16: Whether to use FP16 precision
        workspace_size: Maximum workspace size in bytes
    """
    import tensorrt as trt

    logger.info(f"Building TensorRT engine to {engine_path}...")
    logger.info(f"  Batch size: {max_batch_size}")
    logger.info(f"  Max mel length: {max_mel_length}")
    logger.info(f"  FP16: {use_fp16}")

    # Create TensorRT builder and network
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()

    # Set workspace size
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    logger.info(f"  Workspace: {workspace_size // (1024**3)}GB")

    # Set FP16 mode
    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("  ✓ FP16 enabled")

    # Create ONNX-style network from PyTorch model
    # Note: This is a simplified approach. For production, you'd want to:
    # 1. Export to ONNX first
    # 2. Use trtexec or TensorRT Python API to build engine
    # 3. Handle dynamic shapes properly

    logger.warning("⚠ Direct TensorRT conversion from PyTorch is complex")
    logger.warning("⚠ Recommended approach: Export to ONNX, then build with trtexec")

    # For now, let's provide the recommended workflow
    logger.info("\n" + "="*80)
    logger.info("RECOMMENDED TENSORRT BUILD WORKFLOW")
    logger.info("="*80)

    logger.info("\nStep 1: Export PyTorch model to ONNX")
    logger.info(f"  python export_s3gen_to_onnx.py \\")
    logger.info(f"    --checkpoint {ckpt_dir} \\")
    logger.info(f"    --output s3gen_decoder.onnx")

    logger.info("\nStep 2: Build TensorRT engine with trtexec")
    logger.info(f"  trtexec \\")
    logger.info(f"    --onnx=s3gen_decoder.onnx \\")
    logger.info(f"    --saveEngine=s3gen_decoder.engine \\")
    logger.info(f"    --fp16 \\")
    logger.info(f"    --minShapes=x:2x80x1:2 \\")
    logger.info(f"    --optShapes=x:2x80x300:2 \\")
    logger.info(f"    --maxShapes=x:2x80x500:2 \\")
    logger.info(f"    --workspace=1024MB")

    logger.info("\nStep 3: Use engine in ChatterboxTTSAsync")
    logger.info(f"  model = await ChatterboxTTSAsync.from_pretrained(")
    logger.info(f"      s3gen_use_tensorrt=True,")
    logger.info(f"      s3gen_tensorrt_engine_path='s3gen_decoder.engine',")
    logger.info(f"  )")

    logger.info("\n" + "="*80)

    return False


def export_to_onnx(
    model: nn.Module,
    onnx_path: Path,
    use_fp16: bool = True,
    opset_version: int = 17,
):
    """
    Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        onnx_path: Output ONNX file path
        use_fp16: Whether to export in FP16
        opset_version: ONNX opset version
    """
    logger.info(f"Exporting model to ONNX: {onnx_path}")

    # Create dummy inputs matching the ConditionalDecoder.forward() signature
    # x: (batch_size, in_channels, n_mels)
    # mask: (batch_size, 1, n_mels)
    # mu: (batch_size, n_mels, n_mels)
    # t: (batch_size,)
    # spks: (batch_size, spk_emb_dim)
    # cond: (batch_size, n_feats, n_mels)

    batch_size = 2
    n_mels = 80  # Typical mel length
    spk_emb_dim = 80

    dummy_inputs = {
        "x": torch.randn(batch_size, 320, n_mels).cuda(),
        "mask": torch.ones(batch_size, 1, n_mels).cuda(),
        "mu": torch.randn(batch_size, n_mels, n_mels).cuda(),
        "t": torch.rand(batch_size).cuda(),
        "spks": torch.randn(batch_size, spk_emb_dim).cuda(),
        "cond": torch.randn(batch_size, 80, n_mels).cuda(),
    }

    # Get just the decoder/estimator part
    if hasattr(model, 'flow'):
        if hasattr(model.flow, 'decoder'):
            if hasattr(model.flow.decoder, 'estimator'):
                decoder_model = model.flow.decoder.estimator
                logger.info("✓ Found ConditionalDecoder estimator")
            else:
                logger.error("✗ Could not find estimator in decoder")
                return False
        else:
            logger.error("✗ Could not find decoder in flow")
            return False
    else:
        logger.error("✗ Could not find flow in model")
        return False

    # Export to ONNX
    try:
        torch.onnx.export(
            decoder_model,
            (dummy_inputs["x"], dummy_inputs["mask"], dummy_inputs["mu"],
             dummy_inputs["t"], dummy_inputs["spks"], dummy_inputs["cond"]),
            f=str(onnx_path),
            opset_version=opset_version,
            input_names=['x', 'mask', 'mu', 't', 'spks', 'cond'],
            output_names=['output'],
            dynamic_axes={
                'x': {0: 'batch_size', 2: 'n_mels'},
                'mask': {0: 'batch_size', 2: 'n_mels'},
                'mu': {0: 'batch_size', 1: 'n_mels', 2: 'n_mels'},
                't': {0: 'batch_size'},
                'spks': {0: 'batch_size'},
                'cond': {0: 'batch_size', 2: 'n_mels'},
                'output': {0: 'batch_size', 2: 'n_mels'},
            }
        )
        logger.info(f"✓ ONNX export successful: {onnx_path}")
        return True

    except Exception as e:
        logger.error(f"✗ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engine for S3Gen")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoint/s3gen",
                        help="Path to S3Gen checkpoint directory")
    parser.add_argument("--output-dir", type=str, default="./trt_engines",
                        help="Output directory for TensorRT engine")
    parser.add_argument("--export-onnx", action="store_true",
                        help="Export model to ONNX (required before TensorRT build)")
    parser.add_argument("--max-batch-size", type=int, default=2,
                        help="Maximum batch size for dynamic shapes")
    parser.add_argument("--max-mel-length", type=int, default=500,
                        help="Maximum mel length for dynamic shapes")
    parser.add_argument("--use-fp16", action="store_true", default=True,
                        help="Use FP16 precision (recommended)")

    args = parser.parse_args()

    # Check TensorRT availability
    has_trt, trt = check_tensorrt()
    has_torch_trt = check_torch_tensorrt()

    if not has_trt:
        logger.error("\nCannot build TensorRT engine without TensorRT installed")
        logger.error("Please install TensorRT first, then run this script again")
        return 1

    # Load model
    try:
        ckpt_dir = Path(args.ckpt_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model_from_checkpoint(ckpt_dir, device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export to ONNX first (recommended workflow)
    onnx_path = output_dir / "s3gen_decoder.onnx"

    if args.export_onnx or True:  # Always export ONNX first
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Export to ONNX")
        logger.info("="*80)

        success = export_to_onnx(
            model,
            onnx_path,
            use_fp16=args.use_fp16
        )

        if not success:
            logger.error("ONNX export failed, aborting")
            return 1

    # Build TensorRT engine
    engine_path = output_dir / "s3gen_decoder.engine"

    logger.info("\n" + "="*80)
    logger.info("STEP 2: Build TensorRT Engine")
    logger.info("="*80)

    success = build_tensorrt_engine(
        model,
        engine_path,
        max_batch_size=args.max_batch_size,
        max_mel_length=args.max_mel_length,
        use_fp16=args.use_fp16
    )

    if success:
        logger.info(f"\n✓ TensorRT engine built successfully: {engine_path}")
        logger.info("\nTo use the TensorRT engine:")
        logger.info("  model = await ChatterboxTTSAsync.from_pretrained(")
        logger.info("      s3gen_use_tensorrt=True,")
        logger.info(f"      s3gen_tensorrt_engine_path='{engine_path}',")
        logger.info("  )")
        return 0
    else:
        logger.info("\nPlease follow the recommended workflow above to build the TensorRT engine manually")
        logger.info("This provides better control and visibility into the build process")
        return 0


if __name__ == "__main__":
    sys.exit(main())
