from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import platform
import random
from typing import Tuple, List, Callable, Optional

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np

from torch.ao.quantization import get_default_qconfig, QConfigMapping
from torch.quantization import quantize_fx
import os

from datasets import load_dataset, load_from_disk, Image as HFImage
from context import Context
from download_model import resolve_weights

# ---------------------------
# GLOBAL CONFIGURATION
# ---------------------------
IMAGE_POOL = 1000  # Number of samples utilized for PTQ calibration
SEED = 42          # Deterministic seed for reproducible calibration subsets

HF_CACHE_DIR = os.path.abspath("../data/huggingface_val")        # Directory for initial HuggingFace downloads/cache
HF_SAVED_DIR = os.path.abspath("../data/hf_saved_datasets")      # Directory for locally stored datasets


def static_download(ctx: Context) -> None:
    """
    Executes Post-Training Static Quantization (PTQ) utilizing the PyTorch FX API.
    
    This function:
    1. Loads an FP32 model from the registry.
    2. Prepares the computational graph for static quantization.
    3. Calibrates activation bounds using a subset of a representative dataset.
    4. Converts the model to an INT8 TorchScript artifact and serializes it alongside metadata.

    
    """

    args = ctx.args
    path = args.modeldir
    model_name = args.model
    mod_reg = ctx.MODEL_REGISTRY
    logger = ctx.logger

    if path is None:
        path = "../modelzoo"
        
    # Prepare directory structure for the quantized artifact
    out_dir = Path(path).expanduser().resolve()
    out_dir = out_dir / f"{model_name}_static"
    out_dir.mkdir(parents=True, exist_ok=True)

    if model_name not in mod_reg:
        logger.error(f"Model '{model_name}' not found in the target registry!")
        return

    logger.info(f"Initiating Static Quantization (FX) pipeline for: {model_name}")
    logger.info(f"Target deployment directory: {out_dir}")

    # Infer the optimal quantization backend based on host architecture
    backend = platform.machine().lower()
    torch.backends.quantized.engine = "qnnpack" if ("arm" in backend or "aarch64" in backend) else "fbgemm"

    # Initialize FP32 architecture and weights
    ctor, weights_path, _, dataset_name = mod_reg[model_name]
    weights = resolve_weights(weights_path)

    model_fp32 = ctor(weights=weights).eval()
    categories = weights.meta.get("categories", [])

    # ---------------------------
    # DYNAMIC TRANSFORMS EXTRACTION
    # ---------------------------
    # Replaces hardcoded values to ensure models with non-standard input dimensions 
    # are properly calibrated without shape mismatch errors.
    preprocess = weights.transforms()
    image_size = 224
    resize_side = 256
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def _extract_sizes(obj):
        nonlocal image_size, resize_side, mean, std
        name = obj.__class__.__name__.lower()
        if "resize" in name and hasattr(obj, "size"):
            s = getattr(obj, "size")
            resize_side = int(s[0] if isinstance(s, (list, tuple)) else s)
        elif "centercrop" in name and hasattr(obj, "size"):
            image_size = int(getattr(obj, "size"))
        elif hasattr(obj, "mean") and hasattr(obj, "std"):
            try:
                mean = [float(x) for x in obj.mean]
                std  = [float(x) for x in obj.std]
            except Exception:
                pass

    if hasattr(preprocess, "transforms") and isinstance(getattr(preprocess, "transforms"), (list, tuple)):
        for t in preprocess.transforms:
            _extract_sizes(t)
    elif hasattr(preprocess, "_transforms") and isinstance(getattr(preprocess, "_transforms"), (list, tuple)):
        for t in preprocess._transforms:
            _extract_sizes(t)

    logger.info(f"Utilizing extracted spatial resolution: {image_size}x{image_size}")
    
    # Configure quantization parameters
    qconfig = get_default_qconfig(torch.backends.quantized.engine)
    qconfig_mapping = QConfigMapping().set_global(qconfig)

    # Trace and prepare the graph for calibration
    example_inputs = torch.randn(1, 3, image_size, image_size)
    prepared = quantize_fx.prepare_fx(model_fp32, qconfig_mapping, example_inputs=example_inputs)

    calib_images = IMAGE_POOL  
    calib_seed   = SEED

    # Map architectural display name -> HuggingFace Dataset ID + Split
    DATASET_MAP = {
        "ImageNet-1K": ("imagenet-1k", "validation"),
    }

    hf_id, hf_split = DATASET_MAP.get(dataset_name, (dataset_name, "validation"))

    logger.info(f"Calibration Dataset mapped: {dataset_name} -> HF='{hf_id}', split='{hf_split}', samples={calib_images}, seed={calib_seed}")

    try:
        # Strategy A: Prioritize localized, offline-capable HuggingFace dataset fragments
        LOCAL_HF_DATASETS = Path(os.path.abspath("../data/hf_try1")) 
        parquet_dir = LOCAL_HF_DATASETS / "data"

        if hf_id == "imagenet-1k" and hf_split == "validation":
            files = sorted(str(p) for p in parquet_dir.glob("validation-*.parquet"))
            if not files:
                raise FileNotFoundError(
                    f"No validation-*.parquet shards found in {parquet_dir}. "
                    f"Please ensure prior execution of:\n"
                    f"  hf download imagenet-1k --repo-type dataset --include \"data/validation-*.parquet\" "
                    f"--local-dir {LOCAL_HF_DATASETS}"
                )

            calib_ds = load_dataset("parquet", data_files=files, split="train")
            # Ensure proper casting of image columns to PIL decoded formats
            calib_ds = calib_ds.cast_column("image", HFImage(decode=True))
        else:
            raise RuntimeError(f"No local data loader configured for target: hf_id={hf_id} split={hf_split}.")

        # Apply deterministic shuffle and slice target sequence
        calib_ds = calib_ds.shuffle(seed=calib_seed)

        def to_input(example):
            img = example.get("image", None)
            if img is None:
                return None
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = img.convert("RGB")
            return preprocess(img).unsqueeze(0)

        n = min(calib_images, len(calib_ds))
        with torch.inference_mode():
            taken = 0
            for ex in calib_ds.select(range(n)):
                x = to_input(ex)
                if x is None:
                    continue
                prepared(x)  # Forward pass to calibrate activation observer statistics
                taken += 1

        logger.info(f"Dataset calibration completed successfully. Processed {taken} samples.")

    except Exception as e:
        logger.warning(f"HF dataset calibration failed ({e}). Falling back to local directory subset in ../data/calibration2")

        # Strategy B: Fallback to unstructured local JPEG artifacts
        calib_dir = Path(os.path.abspath("../data/calibration2"))
        calibration_dataset = list(calib_dir.glob("*.jpg"))
        random.Random(calib_seed).shuffle(calibration_dataset)

        def load_and_preprocess(path: Path):
            try:
                img = Image.open(path).convert("RGB")
                return preprocess(img).unsqueeze(0)
            except Exception as ee:
                logger.warning(f"Failed to load image artifact {path}: {ee}")
                return None

        with torch.inference_mode():
            for p in calibration_dataset[:calib_images]:
                x = load_and_preprocess(p)
                if x is None:
                    continue
                prepared(x)

    # ---------------------------
    # CONVERSION & SERIALIZATION
    # ---------------------------
    # Convert prepared graph to quantized INT8 operations
    model_int8 = quantize_fx.convert_fx(prepared).eval()

    example = torch.randn(1, 3, image_size, image_size)
    ts_int8 = torch.jit.trace(model_int8, example)
    ts_path = out_dir / f"{model_name}_static_int8_ts.pt"
    ts_int8.save(str(ts_path))
    
    logger.info(f"Successfully saved TorchScript artifact: {ts_path.name}")

    # Serialize metadata block utilizing dynamically extracted resolutions and normalization metrics
    meta = {
        "architecture": model_name + "_static_int8",            
        "source": "torchvision",
        "weights": weights_path,
        "image_size": image_size,
        "resize_shorter_side": resize_side,
        "center_crop": image_size,
        "normalize_mean": mean,
        "normalize_std": std,
        "categories": categories,
        "framework": {
            "torch": torch.__version__,
            "torchvision": getattr(models, "__version__", "unknown"),
        },
        "quantization": {
            "type": "static",
            "dtype": "int8",
            "layers": "all",
            "engine": torch.backends.quantized.engine,
            "artifact": "torchscript",
        }
    }
    
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Download and quantization process complete. Model is prepared for offline execution.")