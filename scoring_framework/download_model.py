#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Edge AI Model Management CLI:
- download: Downloads official torchvision weights, saves state_dict, TorchScript, and metadata.
- infer:    Loads TorchScript or state_dict + architecture and executes offline inference on images.

Example Usage:
    Download:
    python main.py download \
    --model mobilenet_v3_small \
    --modeldir ./opt/models/mobilenet_v3_small

    Inference:
    python main.py infer \
    --model mobilenet_v3_small \
    --dir ./opt/models/mobilenet_v3_small
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import platform
import random
from typing import Tuple, List
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import Callable, Optional
import torch.nn.functional as F
from torch.ao.quantization import get_default_qconfig
from torch.quantization import quantize_fx
import os
from datasets import load_dataset
from context import Context

# ---------- Logging Configuration ----------
def setup_logging(verbosity: int) -> None:
    """
    Configures standard logging behavior based on verbosity level.
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=level,
    )


# ---------- Standard Download ----------
def cmd_download(ctx: Context) -> None:
    """
    Downloads FP32 pre-trained weights, extracts necessary preprocessing transforms, 
    and serializes the model to both state_dict and TorchScript formats.
    """
    args = ctx.args
    path = args.modeldir
    mod_reg = ctx.MODEL_REGISTRY
    logger = ctx.logger

    if path is None:
        path = "../modelzoo"

    model_name: str = args.model # default
    out_dir = Path(path).expanduser().resolve()
    out_dir = out_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading: {model_name}")

    if model_name not in mod_reg:
        raise ValueError(f"Unknown model '{model_name}'. Supporting only: {', '.join(mod_reg)}")

    ctor, weights_path, _, _ = mod_reg[model_name]
    weights = resolve_weights(weights_path)

    logger.info(f"Loaded {model_name} with {weights_path} …")
    model = ctor(weights=weights)
    model.eval()

    categories = weights.meta.get("categories", [])
    tfm = weights.transforms()

    # Default values for standard architectures
    resize_side = 256
    crop_size = 224
    mean = weights.meta.get("mean", [0.485, 0.456, 0.406])
    std  = weights.meta.get("std",  [0.229, 0.224, 0.225])

    # Transform Introspection:
    # Dynamically extracts the exact crop size and resize resolution from the torchvision weights.
    # This is critical for models utilizing Compound Scaling (like EfficientNets), which 
    # require higher resolutions (e.g., 380x380 for B4) to maintain accuracy.
    def _maybe_update_from_transform(obj):
        nonlocal resize_side, crop_size, mean, std
        name = obj.__class__.__name__.lower()
        if "resize" in name and hasattr(obj, "size"):
            s = getattr(obj, "size")
            resize_side = int(s[0] if isinstance(s, (list, tuple)) else s)
        elif "centercrop" in name and hasattr(obj, "size"):
            crop_size = int(getattr(obj, "size"))
        elif hasattr(obj, "mean") and hasattr(obj, "std"):
            try:
                mean = [float(x) for x in obj.mean]
                std  = [float(x) for x in obj.std]
            except Exception:
                pass

    if hasattr(tfm, "transforms") and isinstance(getattr(tfm, "transforms"), (list, tuple)):
        for t in tfm.transforms:
            _maybe_update_from_transform(t)
    else:
        inner = getattr(tfm, "_transforms", None)
        if isinstance(inner, (list, tuple)):
            for t in inner:
                _maybe_update_from_transform(t)

    # (A) Save state_dict
    sd_path = out_dir / f"{model_name}_state_dict.pt"
    torch.save(model.state_dict(), sd_path)
    logger.info(f"state_dict saved: {sd_path}")

    # (B) Save TorchScript
    # Tracing is performed dynamically using the specific crop_size of the model architecture.
    example = torch.randn(1, 3, crop_size, crop_size)
    traced = torch.jit.trace(model, example)
    ts_path = out_dir / f"{model_name}_ts.pt"
    traced.save(str(ts_path))
    logger.info(f"TorchScript saved: {ts_path}")

    meta = {
        "architecture": model_name,            
        "source": "torchvision",
        "weights": weights_path,
        "image_size": crop_size,
        "resize_shorter_side": resize_side,
        "center_crop": crop_size,
        "normalize_mean": mean,
        "normalize_std": std,
        "categories": categories,
        "framework": {
            "torch": torch.__version__,
            "torchvision": getattr(models, "__version__", "unknown"),
        },
        "quantization": {
            "type": "None",
            "dtype": "",
            "layers": "",
            "engine": "",
            "artifact": "",
        }
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Model successfully downloaded. Ready for offline use.")


def resolve_weights(weights_enum_path: str):
    """
    Resolves weight enums dynamically from strings (e.g., 'MobileNet_V2_Weights.IMAGENET1K_V1').
    """
    enum_name, member = weights_enum_path.split(".", 1)
    enum_obj = getattr(models, enum_name, None)
    if enum_obj is None:
        raise ValueError(f"Weights enum '{enum_name}' not found in torchvision.models")
    return getattr(enum_obj, member)



# ---------- Inference Preprocessing ----------
def build_preprocess_from_meta(meta: dict) -> transforms.Compose:
    """
    Reconstructs the precise preprocessing pipeline offline utilizing the saved metadata.json.
    """
    resize_side = int(meta.get("resize_shorter_side", 256))
    crop_size = int(meta.get("center_crop", 224))
    mean = meta.get("normalize_mean", [0.485, 0.456, 0.406])
    std = meta.get("normalize_std", [0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize(resize_side),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_image(path: Path, preprocess: transforms.Compose) -> torch.Tensor:
    """Loads an image and applies the constructed preprocessing pipeline."""
    img = Image.open(path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)  # (1,3,H,W)
    return tensor


def softmax_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax implementation for NumPy arrays."""
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


# ---------- Inference Execution ----------
def cmd_infer(ctx: Context) -> None:
    """
    Executes offline inference on a specified image using a deployed model.
    Dynamically routes to CPU/GPU and applies correct qnnpack/fbgemm backends based on quantization.
    """
    logger = ctx.logger
    args = ctx.args
    model_name = args.model
    path = args.dir
    mod_reg= ctx.MODEL_REGISTRY

    model_dir = Path(path).expanduser().resolve()

    try:
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
    except Exception as e:
        print(e)

    ts_path = model_dir / f"{model_name}_ts.pt"
    sd_path = model_dir / f"{model_name}_state_dict.pt"

    if not ts_path.is_file():
        raise FileNotFoundError(f"TorchScript file not found: {ts_path}. Please first download the model.")

    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))    

    preprocess = build_preprocess_from_meta(meta)
    categories = meta.get("categories", [])

    isquantized = False
    try:
        q = meta.get("quantization")
        if q is not None:
            isquantized = True
    except Exception as e:
        pass 

    isquantized = True

    device = torch.device("cuda" if (torch.cuda.is_available() and not isinstance and not isquantized) else "cpu")
    iscudaavailable = torch.cuda.is_available()
    logger.info(f"Target Device: {device}")

    backend = platform.machine().lower()
    backend_pack = "qnnpack" if ("arm" in backend or "aarch64" in backend) else "fbgemm"
    if isquantized:
        torch.backends.quantized.engine = backend_pack

    if ts_path.exists():
        logger.info(f"Loading TorchScript: {ts_path}")
        model = torch.jit.load(str(ts_path), map_location=device)
        print("Checkpoint 1")
        model.eval()
    else:
        logger.error(f"Could not find {model_name} in {ts_path}")

    # Load target image (Hardcoded path maintained for testing purposes as requested)
    img_path = Path(args.image).expanduser().resolve()
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    x = load_image(img_path, preprocess).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = softmax_np(logits.cpu().numpy().squeeze())

    probabilities = torch.nn.functional.softmax(logits[0], dim=0)

    topk = 5
    idxs = np.argsort(probs)[::-1][:topk]
    logger.info(f"Top-{topk} Results:")

    top5_prob, top5_catid = torch.topk(probabilities, 5)

    for i in range(top5_prob.size(0)):    
        print(f"{i+1:>2d}. {categories[top5_catid[i]]:<30s} prob: {top5_prob[i].item():.4f}")