#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
from time import time
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import IterableDataset
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
import sys
from pathlib import Path
from torch.utils.data import get_worker_info
import numpy as np
from datetime import datetime
import json

import platform as _platform


from third_prot.context import Context

# -------- Config --------
#MODEL_PATH = "/home/marceldavis/University/BA/FirstZoo/opt/models/efficientnet_b0/efficientnet_b0_ts.pt"   # dein TorchScript-INT8-Modell
BATCH_SIZE = 64                              # CPU-geeignet; an deine Maschine anpassen
NUM_WORKERS = 50                             # DataLoader-Worker (0 auf Windows)
PIN_MEMORY = False                           # CPU-Only -> False
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
RESIZE_SHORTER = 256
CROP_SIZE = 224
SPLIT = "validation"                         # 50k val
SUBSET = None                                # z.B. "validation[:5000]" für schnellen Test
MAX_SAMPLES = 1000                        # number of samples to eval (total will be MAX_SAMPLES * NUM_WORKERS if NUM_WORKERS>0)
MODEL_PATH = "../modelzoo/"


class HFImageNetDataset(IterableDataset):
    def __init__(self, hf_ds, tfm, max_samples: int = None):
        self.ds = hf_ds
        self.tfm = tfm
        self.max_samples = max_samples

    def __iter__(self):
        wi = get_worker_info()
        ds = self.ds
        if wi is not None:
            # jeder Worker bekommt eigenes Shard
            ds = ds.shard(num_shards=wi.num_workers, index=wi.id)

        count = 0
        for item in ds:
            img = item["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img))
            img = img.convert("RGB")
            x = self.tfm(img)
            y = int(item["label"])
            yield x, y
            count += 1
            if self.max_samples is not None and count >= self.max_samples:
                break

def append_json_record(json_path: Path, record: dict):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
    else:
        data = []

    data.append(record)
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def build_preprocess() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(RESIZE_SHORTER),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def topk_correct(logits: torch.Tensor, target: torch.Tensor, k: int = 5) -> int:
    """
    Zählt, wie viele Targets in den Top-k Vorhersagen liegen.
    logits: (B, C), target: (B,)
    """
    _, pred = torch.topk(logits, k, dim=1)
    correct = (pred == target.view(-1, 1)).any(dim=1).sum().item()
    return correct

def main(ctx: Context = None):
    if ctx is None:
        return
    args = ctx.args
    path = args.modeldir
    samplesize = args.samplesize
    all = args.all
    logpath = args.logpath
    mod_reg = ctx.MODEL_REGISTRY
    logger = ctx.logger

    if path is None:
        path = MODEL_PATH

    ALLOWED_MODELS = ["mobilenet_v3_small", "mnasnet0_5", "shufflenet_v2_x0_5", "mobilenet_v2", "regnet_x_400mf",  "resnet18","ConvNeXt_Base","vit_b_16"]	


    if all == False:
        if path is None:
            logger.error("Please provide model directory path using --modeldir")
            return
        dir = Path(path).expanduser().resolve()
        pt_path = next(
                (p for p in dir.rglob("*.pt") if "dict" not in p.name.lower() and "state" not in p.name.lower()),
                None
            )
        if pt_path is None:
            raise FileNotFoundError(f"Keine .pt Datei in {dir} gefunden")
        if samplesize is None:
            samplesize = MAX_SAMPLES
        benchmark(pt_path, samplesize,logger,logpath)
    
    if all == True:
        root = Path(path).expanduser().resolve()
        for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            model_name = model_dir.name
            
        
            if not any(model_name.startswith(allowed) for allowed in ALLOWED_MODELS):
                continue

        
            pt_path = next((p for p in model_dir.rglob("*.pt") if "dict" not in p.name.lower() and "state" not in p.name.lower()), None)

            if pt_path is None:
                logger.error(f"Keine passende .pt Datei (ohne 'dict') in {model_dir} gefunden, überspringe...")
                continue
            
            if samplesize is None:
                samplesize = MAX_SAMPLES

            logger.info(f"Benchmarking model: {model_name} ({pt_path.name})")
            benchmark(pt_path, samplesize, logger, logpath)


def benchmark(path, size,logger,logp):

    WARMUP_STEPS = 5
    timed_images = 0
    timed_seconds = 0.0

    path_str = str(path.resolve())


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if "int8" in path_str.lower():
        device = torch.device("cpu") 

    # 1)Load Model (TorchScript INT8 -> CPU)
    logger.info(f"Loading TorchScript model from: {path}")
    model = torch.jit.load(path, map_location= device)
    #activates inference mode, forecast gets deterministic, 
    model.eval()
    logger.info(f"Model loaded. Running on {device}.")

    # 2) Load dataset (HuggingFace)
    split_str = SUBSET if SUBSET is not None else SPLIT
    logger.info(f"Loading ImageNet-1k split from Hugging Face: {split_str}")


    root = Path("/home/marceldavis/University/BA/FirstZoo/data/data") 
    files = sorted(str(p) for p in (root / "").glob("validation-*.parquet"))

    hf_ds = load_dataset("parquet", data_files=files, split="validation")#,cache_dir="/home/marceldavis/University/BA/

    # 3) Preprocessing + DataLoader
    preprocess = build_preprocess()
    ds = HFImageNetDataset(hf_ds, preprocess,max_samples=size)

  
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        #cache_dir="/home/marceldavis/University/BA/FirstZoo/data"
    )

    total = 0
    top1_correct = 0
    top5_correct = 0


    t0 = time()
    with torch.inference_mode():
        for step_idx, (x, y) in enumerate(tqdm(loader, desc="Evaluating")):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize()


            t1 = time()
            logits = model(x)   # (B, 1000)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time()   

            if step_idx >= WARMUP_STEPS:
                timed_seconds += (t2 - t1)
                timed_images += x.size(0) 

            # Top-1
            pred1 = logits.argmax(dim=1)    # (B,)
            top1_correct += (pred1 == y).sum().item()
            # Top-5
            top5_correct += topk_correct(logits, y, k=5)
            total += y.size(0)
            print(f"\r[INFO] Processed {total} samples...", end="", flush=True)
            
    if timed_images > 0 and timed_seconds > 0:
        latency_ms_per_sample = (timed_seconds / timed_images) * 1000.0
        throughput_fps = timed_images / timed_seconds
    else:
        latency_ms_per_sample = None
        throughput_fps = None

    dt = time() - t0
    top1 = top1_correct / total
    top5 = top5_correct / total

    logger.info("\n================= Results =================")
    logger.info(f"Model            : {path}")
    logger.info(f"Device           : {device}")
    logger.info(f"Samples evaluated : {total}")
    logger.info(f"Top-1 Accuracy    : {top1:.4%}")
    logger.info(f"Top-5 Accuracy    : {top5:.4%}")
    logger.info(f"Total time (s)    : {dt:.1f}")
    logger.info(f"Throughput (img/s): {total / dt:.1f}")
    logger.info("===========================================\n")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if logp != None:
        log_path = logp + f"accuracy_benchmark_results_{ts}.txt"
    else:
        log_path = "accuracy_benchmark_results.txt"

    line = (
        "\n================= Results =================\n"
        f"Model: {path}\n"
        f"Device: {device}\n"
        f"Samples: {total}, \n"
        f"Top-1: {top1:.4%}, \n"
        f"Top-5: {top5:.4%}, \n"
        f"Time: {dt:.1f}s, \n"
        f"Throughput: {total/dt:.1f} img/s\n"
        "===========================================\n"
    )

 
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)

    logger.info(f"[INFO] Results appended to {log_path}")

    ts_iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


    p_lower = path_str.lower()
    precision = "int8" if "int8" in p_lower else "fp32"
    variant = "int8" if precision == "int8" else "fp32"


    quant_block = None
    if precision == "int8":
        quant_block = {"type": "static_or_qat", "engine": "fbgemm/qnnpack"} 


    record = {
        "model_name": Path(path_str).parent.name,  
        "format": "torchscript",
        "precision": precision,
        "quantization": quant_block if precision != "fp32" else {"type": "None", "engine": "None"},

        "hardware": {
            "DEVICE_NAME": "Notebook",
            "CPU": {
                "model": "Intel Core i7-12700H",
                "cores": 20,
                "base_clock": "2.3 GHz",
                "max_clock": "4.7 GHz",
                "architecture": "x86_64"
            },
            "CPU_ONLY": False,
            "GPU": {
                "model": "NVIDIA RTX 3070",
                "vram": "8GB",              
                "cuda_cores": 16384,            
                "tensor_cores": 512, #0 if not supported
                "cuda_version": "11.4",
                "driver_version": "470.42.01",
            },
            "TPU": {
                "model": "Edge TPU",
                "performance_tops": 4.0,
                "tensor_cores": 4,
                "version": "v2",
                "accelerator": False,
            },
            "RAM": {
                "total": "16GB",
                "type": "DDR4",
                "filter": False
            },
            "Storage": {
                "type": "SSD",
                "capacity": "1000GB",
                "speed": "3.0 GB/s"
            }
        },

        "benchmark": {
            "dataset": f"imagenet_val_subset{total}",
            "batch_size": BATCH_SIZE,
            "latency_ms_per_sample": latency_ms_per_sample,
            "throughput_fps": throughput_fps,
            "peak_ram_mb": None,  # optional später
            "accuracy_top1": float(top1),
            "accuracy_top5": float(top5),
            "num_samples_eval": int(total),
        },

        "artifact": {
            "path": str(path_str),
            "disk_size_mb": round(os.path.getsize(path_str) / (1024 * 1024), 3),
        },

        "env": {
            "torch": torch.__version__,
            "torchvision": getattr(transforms, "__module__", "unknown"),
            "os": _platform.platform(),
        },

        "timestamp": ts_iso,
    }
    if device.type == "cuda":
        record["hardware"]["device_name"] = torch.cuda.get_device_name(0)

    json_path = Path(logp) / "benchmark.json" if logp else Path("benchmark.json")
    append_json_record(json_path, record)
    logger.info(f"[INFO] JSON appended to {json_path}")


if __name__ == "__main__":
    main()
