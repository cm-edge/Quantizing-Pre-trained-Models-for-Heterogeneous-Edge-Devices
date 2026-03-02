#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Edge AI Quantization-Aware Training (QAT) Module:

This script executes a Quantization-Aware Training (QAT) pipeline.
Unlike Post-Training Quantization (PTQ), QAT inserts fake quantization nodes into the 
computational graph during training. This allows the neural network to adapt its weights 
to the quantization noise, typically resulting in higher accuracy for INT8 edge deployments.

Pipeline:
1. Loads a pretrained FP32 model from the torchvision registry.
2. Prepares the computational graph for QAT utilizing the PyTorch FX API.
3. Executes a brief fine-tuning loop (simulated QAT) on a local Parquet dataset.
4. Converts the trained graph to a true INT8 representation and serializes it.


"""

import platform
from pathlib import Path
import torch
from torch import nn, optim
from torchvision import models, transforms
from torch.ao.quantization import (
    get_default_qat_qconfig
)
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from datasets import load_dataset, Image as HFImage
from PIL import Image
import random
import os
import json

# ---------------------------
# CONFIGURATION
# ---------------------------
epochs = 2            # Brief fine-tuning duration
lr = 1e-5             # Learning rate for fine-tuning
batch_size = 32       # Batch size
max_batches_per_epoch = 100  # Cap on batches for rapid demonstration/testing
DATASET_PATH = "../data/hf_try1"  # Path to local Parquet dataset shard


def infer_default_engine() -> str:
    """
    Infers the optimal quantization backend based on the host architecture.
    ARM architectures (e.g., Raspberry Pi) utilize 'qnnpack', while x86 utilizes 'fbgemm'.
    """
    arch = platform.machine().lower()
    if "arm" in arch or "aarch64" in arch:
        return "qnnpack"
    return "fbgemm"

def resolve_weights(weights_enum_path: str):
    """
    Dynamically resolves torchvision weight enums from string paths.
    """
    enum_name, member = weights_enum_path.split(".", 1)
    enum_obj = getattr(models, enum_name, None)
    if enum_obj is None:
        raise ValueError(f"Weights enum '{enum_name}' not found in torchvision.models")
    return getattr(enum_obj, member)

def qat_download(ctx: object) -> None:
    """
    Executes the Quantization-Aware Training workflow.
    Prepares the model, fine-tunes it on a local dataset, and exports the quantized INT8 artifact.
    """

    args = ctx.args
    model_name = args.model
    path = args.modeldir
    mod_reg = ctx.MODEL_REGISTRY
    logger = ctx.logger

    if path is None:
        path = "../modelzoo"

    out_dir = Path(path).expanduser().resolve()
    out_dir = out_dir / f"{model_name}_qat"
    out_dir.mkdir(parents=True, exist_ok=True)

    if model_name not in mod_reg:
        raise ValueError(f"Unknown model '{model_name}'. Supporting only: {', '.join(mod_reg)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target Device: {device}")

    # Set architecture-specific quantization backend
    torch.backends.quantized.engine = backend = infer_default_engine()
    logger.info(f"Active Quantization Engine: {torch.backends.quantized.engine}")

    # ---------------------------
    # LOAD MODEL
    # ---------------------------
    ctor, weights_path, _, dataset_name = mod_reg[model_name]
    weights = resolve_weights(weights_path)
    categories = weights.meta.get("categories", [])
    
    # Initialize FP32 model and move to target device
    float_model = ctor(weights=weights).to(device)
    float_model.eval()

    # ---------------------------
    # DYNAMIC TRANSFORMS EXTRACTION
    # ---------------------------
    # Extracts precise resolutions from torchvision weights to support Compound Scaling architectures (e.g., EfficientNets)
    preprocess_transforms = weights.transforms()
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

    if hasattr(preprocess_transforms, "transforms") and isinstance(getattr(preprocess_transforms, "transforms"), (list, tuple)):
        for t in preprocess_transforms.transforms:
            _extract_sizes(t)
    elif hasattr(preprocess_transforms, "_transforms") and isinstance(getattr(preprocess_transforms, "_transforms"), (list, tuple)):
        for t in preprocess_transforms._transforms:
            _extract_sizes(t)

    logger.info(f"Utilizing extracted resolution: Resize={resize_side}, Crop={trace_size}")

    transform = transforms.Compose([
        transforms.Resize(resize_side),
        transforms.CenterCrop(trace_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # ---------------------------
    # DATASET PIPELINE (Local Parquet)
    # ---------------------------
    DATASET_MAP = {
        "ImageNet-1K": ("imagenet-1k", "train"),
    }
    hf_id, hf_split = DATASET_MAP.get(dataset_name, (dataset_name, "train"))

    LOCAL_HF_ROOT = Path(os.path.abspath(DATASET_PATH))
    parquet_dir = LOCAL_HF_ROOT / "data"

    if hf_id == "imagenet-1k" and hf_split == "train":
        parquet_files = sorted(str(p) for p in parquet_dir.glob("train-*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No train-*.parquet files found in {parquet_dir}.\n"
                f"Please execute the following command once:\n"
                f"  hf download imagenet-1k --repo-type dataset "
                f"--include \"data/train-*.parquet\" --local-dir {LOCAL_HF_ROOT}"
            )

        # Load local dataset from Parquet shards
        dataset = load_dataset("parquet", data_files=parquet_files, split="train")
        # Ensure PIL Image decoding
        dataset = dataset.cast_column("image", HFImage(decode=True))
    else:
        raise RuntimeError(f"No local Parquet loader configured for: {hf_id} / {hf_split}")

    # Randomize dataset deterministically
    dataset = dataset.shuffle(seed=0)

    logger.info("\n📦 Preparing DataLoader...")

    def collate(batch):
        """Custom collate function to handle PIL image conversion and transformations."""
        imgs = []
        labels = []
        for b in batch:
            img = b["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            # Ensure 3-channel RGB format
            img = img.convert("RGB")
            imgs.append(transform(img))
            labels.append(b["label"])
        return torch.stack(imgs), torch.tensor(labels)

    logger.info("✅ DataLoader ready.")

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate)

    logger.info("="*80)
    
    # ---------------------------
    # PREPARE FOR QAT
    # ---------------------------
    qconfig_dict = {"": get_default_qat_qconfig(backend)}
    # Utilize dynamically extracted trace_size for model preparation
    example_inputs = (torch.randn(1, 3, trace_size, trace_size, device=device),)
    
    # QAT requires the model to be in training mode
    model_prepared = prepare_qat_fx(float_model.train(), qconfig_dict, example_inputs=example_inputs)
    model_prepared.to(device) 

    optimizer = optim.Adam(model_prepared.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    logger.info("\n🚀 Starting Quantization-Aware Training...")

    #---------------------------
    # TRAINING LOOP
    #---------------------------
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(loader):
            if i >= max_batches_per_epoch:
                break

            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model_prepared(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 0:
                logger.info(f"[Epoch {epoch+1}] Batch {i} | Loss: {loss.item():.4f}")

    logger.info("\n✅ QAT fine-tuning complete.")

    # ---------------------------
    # CONVERT & SAVE
    # ---------------------------
    # Switch to evaluation mode prior to conversion
    model_prepared.eval()
    model_prepared.to("cpu") 
    
    # Convert simulated fake-quantized model to true INT8 representation
    model_quantized = convert_fx(model_prepared)

    # Trace utilizing dynamically extracted dimensions
    example = torch.randn(1, 3, trace_size, trace_size)
    traced = torch.jit.trace(model_quantized, example)

    # Construct and serialize metadata payload
    meta = {
        "architecture": model_name + "_qat_int8",           
        "source": "torchvision",
        "weights": weights_path,
        "image_size": trace_size,              # Dynamically extracted
        "resize_shorter_side": resize_side,    # Dynamically extracted
        "center_crop": trace_size,             # Dynamically extracted
        "normalize_mean": mean,                # Dynamically extracted
        "normalize_std": std,                  # Dynamically extracted
        "categories": categories,
        "framework": {
            "torch": torch.__version__,
            "torchvision": getattr(models, "__version__", "unknown"),
        },
        "quantization": {
            "type": "qat",
            "dtype": "int8",
            "layers": "all",
            "engine": torch.backends.quantized.engine,
            "artifact": "torchscript",
        }
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Serialize TorchScript and State Dict artifacts
    torch.jit.save(traced, os.path.join(out_dir, f"{model_name}_qat_int8_ts.pt"))
    torch.save(model_quantized.state_dict(), os.path.join(out_dir, f"{model_name}_qat_int8_state.pt"))
    
    logger.info(f"💾 Saved quantized artifacts to {out_dir}")