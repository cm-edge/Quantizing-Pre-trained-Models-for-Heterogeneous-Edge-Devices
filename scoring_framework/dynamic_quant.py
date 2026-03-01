#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Edge AI Dynamic Quantization Module:
This module manages the downloading of pre-trained FP32 torchvision models and 
applies Dynamic Post-Training Quantization (PTQ). 

Specifically, dynamic quantization converts weights (typically `nn.Linear` layers) 
to INT8 precision ahead of time, while activations are quantized dynamically during 
inference. This reduces model size and accelerates memory-bound operations, 
making it highly suitable for Edge AI deployments.


"""

from __future__ import annotations
import argparse
import json
import logging
import platform
from pathlib import Path
from typing import Callable, List, Tuple, Dict
from context import Context

import torch
from torchvision import models, transforms
from download_model import resolve_weights


# -------------- Helper Functions --------------
def resolve_weights(weights_enum_path: str):
    """
    Dynamically resolves torchvision weight enums from string paths.
    
    Args:
        weights_enum_path (str): e.g., 'MobileNet_V2_Weights.IMAGENET1K_V1'
    Returns:
        Enum: The resolved torchvision weights object.
    """
    enum_name, member = weights_enum_path.split(".", 1)
    enum_obj = getattr(models, enum_name, None)
    if enum_obj is None:
        raise ValueError(f"Weights enum '{enum_name}' not found in torchvision.models")
    return getattr(enum_obj, member)

def safe_make_dir(path: Path) -> None:
    """Safely creates a directory and its parent directories if they do not exist."""
    path.mkdir(parents=True, exist_ok=True)

def infer_default_engine() -> str:
    """
    Infers the optimal quantization backend based on the host architecture.
    ARM architectures (e.g., Raspberry Pi) utilize 'qnnpack', while x86 utilizes 'fbgemm'.
    """
    arch = platform.machine().lower()
    if "arm" in arch or "aarch64" in arch:
        return "qnnpack"
    return "fbgemm"

def extract_preprocess_from_weights(weights) -> dict:
    """
    Introspects the torchvision weights to extract the precise preprocessing parameters 
    (crop size, resize dimensions, normalization mean/std). 
    This is critical for models utilizing Compound Scaling (e.g., EfficientNets) that 
    deviate from standard 224x224 resolutions.
    """
    # ImageNet Defaults
    meta = {
        "resize_shorter_side": 256,
        "center_crop": 224,
        "normalize_mean": weights.meta.get("mean", [0.485, 0.456, 0.406]),
        "normalize_std":  weights.meta.get("std",  [0.229, 0.224, 0.225]),
        "categories": weights.meta.get("categories", []),
    }
    
    tfm = weights.transforms()
    
    def maybe_update(obj):
        name = obj.__class__.__name__.lower()
        if "resize" in name and hasattr(obj, "size"):
            s = getattr(obj, "size")
            meta["resize_shorter_side"] = int(s[0] if isinstance(s, (list, tuple)) else s)
        elif "centercrop" in name and hasattr(obj, "size"):
            meta["center_crop"] = int(getattr(obj, "size"))
        elif hasattr(obj, "mean") and hasattr(obj, "std"):
            try:
                meta["normalize_mean"] = [float(x) for x in obj.mean]
                meta["normalize_std"]  = [float(x) for x in obj.std]
            except Exception:
                pass

    if hasattr(tfm, "transforms") and isinstance(getattr(tfm, "transforms"), (list, tuple)):
        for t in tfm.transforms: 
            maybe_update(t)
    elif hasattr(tfm, "_transforms") and isinstance(getattr(tfm, "_transforms"), (list, tuple)):
        for t in tfm._transforms: 
            maybe_update(t)
            
    return meta

def save_metadata(out_dir: Path, model_name: str, weights_path: str, preprocess: dict, quant: dict) -> None:
    """Serializes deployment configuration and hardware constraints to a metadata.json file."""
    name = model_name + "_dynamic_int8"
    meta = {
        "architecture": name,
        "source": "torchvision",
        "weights": weights_path,
        "image_size": preprocess.get("center_crop", 224),
        **preprocess,
        "quantization": quant,
        "framework": {
            "torch": torch.__version__,
            "torchvision": getattr(models, "__version__", "unknown"),
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

def ensure_fp32_files(modelregistry, model_name: str, out_dir: Path, force_download: bool = False) -> Tuple[Path, Path]:
    """
    Ensures that the foundational FP32 state_dict and metadata exist locally prior to quantization.
    Downloads them from the registry if they are missing.
    
    Returns: 
        Tuple[Path, Path]: File paths to the state_dict and metadata.json.
    """
    sd_path   = out_dir / f"{model_name}_state_dict.pt"
    meta_path = out_dir / "metadata.json"

    if sd_path.exists() and meta_path.exists() and not force_download:
        logging.info("FP32 state_dict and metadata.json already exist locally. Skipping initial download.")
        return sd_path, meta_path

    logging.info("Loading FP-32 Weights from torchvision registry...")
    ctor, weights_path, _, _ = modelregistry[model_name]
    weights = resolve_weights(weights_path)
    model = ctor(weights=weights)
    model.eval()

    # Save initial FP32 state_dict
    torch.save(model.state_dict(), sd_path)
    logging.info(f"Serialized FP32 state_dict saved to: {sd_path}")

    # Extract dynamic preprocessing parameters and save initial metadata
    preprocess = extract_preprocess_from_weights(weights)
    save_metadata(out_dir, model_name, weights_path, preprocess, quant={"type": "none"})
    logging.info(f"Initial metadata saved to: {meta_path}")

    return sd_path, meta_path

# -------------- Core: Dynamic Quantization --------------
def dynamic_quantize(ctx: Context) -> Path:
    """
    Executes Dynamic Quantization on the specified model.
    1. Loads the foundational FP32 state_dict.
    2. Dynamically quantizes linear memory-bound layers (`nn.Linear`) to INT8.
    3. Traces the model to TorchScript utilizing dynamically extracted resolutions.
    4. Serializes the final INT8 TorchScript model.
    
    Returns:
        Path: The file path to the saved quantized TorchScript model.
    """
    args = ctx.args
    model_name = args.model
    path = args.modeldir
    mod_reg = ctx.MODEL_REGISTRY
    logger = ctx.logger

    if path is None:
        path = "../modelzoo"

    out_dir = Path(path).expanduser().resolve()
    out_dir = out_dir / f"{model_name}_dynamic"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Automatically set the target quantization backend
    backend = platform.machine().lower()
    torch.backends.quantized.engine = "qnnpack" if ("arm" in backend or "aarch64" in backend) else "fbgemm"
    engine = torch.backends.quantized.engine

    logging.info(f"Active quantization backend engine set to: {engine}")

    # 1) Ensure FP32 baseline models exist locally
    sd_path, meta_path = ensure_fp32_files(mod_reg, model_name, out_dir, force_download=False)

    # 2) Load architecture and initialize with FP32 weights
    ctor, weights_path, _, _  = mod_reg[model_name]
    weights = resolve_weights(weights_path)
    categories = weights.meta.get("categories", [])
    model = ctor(weights=None)
    model.load_state_dict(torch.load(sd_path, map_location="cpu", weights_only=True))
    model.eval()

    # 3) Apply Dynamic Quantization (targeting nn.Linear layers)
    from torch.ao.quantization import quantize_dynamic
    q_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8).eval()

    # --- Dynamic Resolution Extraction for Tracing ---
    # EfficientNets are properly traced without shape mismatch errors downstream
    preprocess = weights.transforms()
    trace_size = 224
    resize_side = 256
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def _extract_sizes(obj):
        nonlocal trace_size, resize_side, mean, std
        name = obj.__class__.__name__.lower()
        if "resize" in name and hasattr(obj, "size"):
            s = getattr(obj, "size")
            resize_side = int(s[0] if isinstance(s, (list, tuple)) else s)
        elif "centercrop" in name and hasattr(obj, "size"):
            trace_size = int(getattr(obj, "size"))
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

    logger.info(f"Tracing TorchScript utilizing dynamic image size: {trace_size}x{trace_size}")

    # 4) Trace model to TorchScript
    example = torch.randn(1, 3, trace_size, trace_size)
    ts = torch.jit.trace(q_model, example)
    ts_path = out_dir / f"{model_name}_dynamic_int8_ts.pt"
    ts.save(str(ts_path))
    logging.info(f"Quantized TorchScript successfully saved to: {ts_path}")

    # 5) Update and serialize final metadata block
    meta = {
        "architecture": model_name + "_dynamic_int8",           
        "source": "torchvision",
        "weights": weights_path,
        "image_size": trace_size,             # Dynamically extracted
        "resize_shorter_side": resize_side,   # Dynamically extracted
        "center_crop": trace_size,            # Dynamically extracted
        "normalize_mean": mean,               # Dynamically extracted
        "normalize_std": std,                 # Dynamically extracted
        "categories": categories,
        "framework": {
            "torch": torch.__version__,
            "torchvision": getattr(models, "__version__", "unknown"),
        },
        "quantization": {
            "type": "dynamic",
            "dtype": "int8",
            "layers": "Linear",               
            "engine": engine,
            "artifact": "torchscript",
        }
    }
    
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logging.info(f"Final deployment metadata updated: {meta_path}")

    return ts_path