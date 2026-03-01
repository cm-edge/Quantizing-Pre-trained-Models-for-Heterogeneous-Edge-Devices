import argparse
import logging
from pathlib import Path
import json
import numpy as np

from download_model import cmd_download
from dynamic_quant import dynamic_quantize
from static_quant import static_download
from qat import qat_download
from context import Context
from download_model import cmd_infer
import sys
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from accuracy_benchmark.accuracy_benchmark import main as accracy_benchmark    

import argparse
import json
import logging
import platform
from pathlib import Path
from typing import Callable, List, Tuple, Dict
import textwrap

import torch
from torchvision import models, transforms

"""
Recommendation System and Model Zoo Management for Edge Devices.

This module provides a comprehensive command-line interface and recommendation engine 
tailored for deploying deep learning models on Edge AI devices. It filters models 
based on user-defined constraints and hardware specifications, applying the TOPSIS 
(Technique for Order of Preference by Similarity to Ideal Solution) algorithm to 
rank models according to multiple criteria (e.g., accuracy, latency, throughput, storage).



Example Usage for recommendation:
    python main.py recom --input input.json

Example Usage for download of MobileNetV2 with static quantization:
    python main.py download --model mobilenet_v2 --modeldir ../opt/models --static True

Environment Initialization:
    conda activate mobilemlzoo
"""

TASK_CLASSIFICATION = "classification"
DS_IMAGENET1K = "ImageNet-1K"
DS_IMAGENET1K_SWAG = "ImageNet-1K (SWAG pretrain)"
PATH_TO_BENCHMARKDB = "utility_files/modelbenchmark_db.json" # Adjust path if necessary

# Model registry containing supported architectures, their corresponding pre-trained weights, task type, and dataset.
MODEL_REGISTRY: Dict[str, Tuple[Callable[..., torch.nn.Module], str, str, str]] = {
    # AlexNet
    "alexnet": (models.alexnet, "AlexNet_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),

    # ConvNeXt
    "convnext_tiny":  (models.convnext_tiny,  "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",  TASK_CLASSIFICATION, DS_IMAGENET1K),
    "convnext_small": (models.convnext_small, "ConvNeXt_Small_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "convnext_base":  (models.convnext_base,  "ConvNeXt_Base_Weights.IMAGENET1K_V1",  TASK_CLASSIFICATION, DS_IMAGENET1K),
    "convnext_large": (models.convnext_large, "ConvNeXt_Large_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),

    # DenseNet
    "densenet121": (models.densenet121, "DenseNet121_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "densenet161": (models.densenet161, "DenseNet161_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "densenet169": (models.densenet169, "DenseNet169_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "densenet201": (models.densenet201, "DenseNet201_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),

    # EfficientNet
    "efficientnet_b0": (models.efficientnet_b0, "EfficientNet_B0_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "efficientnet_b1": (models.efficientnet_b1, "EfficientNet_B1_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "efficientnet_b2": (models.efficientnet_b2, "EfficientNet_B2_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "efficientnet_b3": (models.efficientnet_b3, "EfficientNet_B3_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "efficientnet_b4": (models.efficientnet_b4, "EfficientNet_B4_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "efficientnet_b5": (models.efficientnet_b5, "EfficientNet_B5_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "efficientnet_b6": (models.efficientnet_b6, "EfficientNet_B6_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "efficientnet_b7": (models.efficientnet_b7, "EfficientNet_B7_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),

    # EfficientNetV2
    "efficientnet_v2_s": (models.efficientnet_v2_s, "EfficientNet_V2_S_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "efficientnet_v2_m": (models.efficientnet_v2_m, "EfficientNet_V2_M_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "efficientnet_v2_l": (models.efficientnet_v2_l, "EfficientNet_V2_L_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),

    # GoogLeNet / InceptionV3
    "googlenet":   (models.googlenet,   "GoogLeNet_Weights.IMAGENET1K_V1",   TASK_CLASSIFICATION, DS_IMAGENET1K),
    "inception_v3": (models.inception_v3, "Inception_V3_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),

    # MaxVit
    "maxvit_t": (models.maxvit_t, "MaxVit_T_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),

    # MNASNet
    "mnasnet0_5":  (models.mnasnet0_5,  "MNASNet0_5_Weights.IMAGENET1K_V1",  TASK_CLASSIFICATION, DS_IMAGENET1K),
    "mnasnet0_75": (models.mnasnet0_75, "MNASNet0_75_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "mnasnet1_0":  (models.mnasnet1_0,  "MNASNet1_0_Weights.IMAGENET1K_V1",  TASK_CLASSIFICATION, DS_IMAGENET1K),
    "mnasnet1_3":  (models.mnasnet1_3,  "MNASNet1_3_Weights.IMAGENET1K_V1",  TASK_CLASSIFICATION, DS_IMAGENET1K),

    # MobileNetV2 / V3
    "mobilenet_v2":       (models.mobilenet_v2,       "MobileNet_V2_Weights.IMAGENET1K_V1",       TASK_CLASSIFICATION, DS_IMAGENET1K),
    "mobilenet_v3_small": (models.mobilenet_v3_small, "MobileNet_V3_Small_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "mobilenet_v3_large": (models.mobilenet_v3_large, "MobileNet_V3_Large_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),

    # RegNet
    "regnet_x_400mf": (models.regnet_x_400mf, "RegNet_X_400MF_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "regnet_x_800mf": (models.regnet_x_800mf, "RegNet_X_800MF_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "regnet_x_1_6gf": (models.regnet_x_1_6gf, "RegNet_X_1_6GF_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "regnet_x_3_2gf": (models.regnet_x_3_2gf, "RegNet_X_3_2GF_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "regnet_x_8gf":   (models.regnet_x_8gf,   "RegNet_X_8GF_Weights.IMAGENET1K_V1",   TASK_CLASSIFICATION, DS_IMAGENET1K),
    "regnet_x_16gf":  (models.regnet_x_16gf,  "RegNet_X_16GF_Weights.IMAGENET1K_V1",  TASK_CLASSIFICATION, DS_IMAGENET1K),
    "regnet_x_32gf":  (models.regnet_x_32gf,  "RegNet_X_32GF_Weights.IMAGENET1K_V1",  TASK_CLASSIFICATION, DS_IMAGENET1K),

    "regnet_y_400mf": (models.regnet_y_400mf, "RegNet_Y_400MF_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "regnet_y_800mf": (models.regnet_y_800mf, "RegNet_Y_800MF_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "regnet_y_1_6gf": (models.regnet_y_1_6gf, "RegNet_Y_1_6GF_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "regnet_y_3_2gf": (models.regnet_y_3_2gf, "RegNet_Y_3_2GF_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "regnet_y_8gf":   (models.regnet_y_8gf,   "RegNet_Y_8GF_Weights.IMAGENET1K_V1",   TASK_CLASSIFICATION, DS_IMAGENET1K),
    "regnet_y_16gf":  (models.regnet_y_16gf,  "RegNet_Y_16GF_Weights.IMAGENET1K_V1",  TASK_CLASSIFICATION, DS_IMAGENET1K),
    "regnet_y_32gf":  (models.regnet_y_32gf,  "RegNet_Y_32GF_Weights.IMAGENET1K_V1",  TASK_CLASSIFICATION, DS_IMAGENET1K),

    # Special case: SWAG -> ImageNet-1K fine-tune
    "regnet_y_128gf": (models.regnet_y_128gf, "RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1", TASK_CLASSIFICATION, DS_IMAGENET1K_SWAG),

    # ResNet
    "resnet18":  (models.resnet18,  "ResNet18_Weights.IMAGENET1K_V1",  TASK_CLASSIFICATION, DS_IMAGENET1K),
    "resnet34":  (models.resnet34,  "ResNet34_Weights.IMAGENET1K_V1",  TASK_CLASSIFICATION, DS_IMAGENET1K),
    "resnet50":  (models.resnet50,  "ResNet50_Weights.IMAGENET1K_V1",  TASK_CLASSIFICATION, DS_IMAGENET1K),
    "resnet101": (models.resnet101, "ResNet101_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "resnet152": (models.resnet152, "ResNet152_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),

    # ResNeXt
    "resnext50_32x4d":  (models.resnext50_32x4d,  "ResNeXt50_32X4D_Weights.IMAGENET1K_V1",  TASK_CLASSIFICATION, DS_IMAGENET1K),
    "resnext101_32x8d": (models.resnext101_32x8d, "ResNeXt101_32X8D_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "resnext101_64x4d": (models.resnext101_64x4d, "ResNeXt101_64X4D_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),

    # ShuffleNetV2
    "shufflenet_v2_x0_5": (models.shufflenet_v2_x0_5, "ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "shufflenet_v2_x1_0": (models.shufflenet_v2_x1_0, "ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "shufflenet_v2_x1_5": (models.shufflenet_v2_x1_5, "ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "shufflenet_v2_x2_0": (models.shufflenet_v2_x2_0, "ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),

    # SqueezeNet
    "squeezenet1_0": (models.squeezenet1_0, "SqueezeNet1_0_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "squeezenet1_1": (models.squeezenet1_1, "SqueezeNet1_1_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),

    # Swin Transformer (v1 + v2)
    "swin_t":    (models.swin_t,    "Swin_T_Weights.IMAGENET1K_V1",    TASK_CLASSIFICATION, DS_IMAGENET1K),
    "swin_s":    (models.swin_s,    "Swin_S_Weights.IMAGENET1K_V1",    TASK_CLASSIFICATION, DS_IMAGENET1K),
    "swin_b":    (models.swin_b,    "Swin_B_Weights.IMAGENET1K_V1",    TASK_CLASSIFICATION, DS_IMAGENET1K),
    "swin_v2_t": (models.swin_v2_t, "Swin_V2_T_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "swin_v2_s": (models.swin_v2_s, "Swin_V2_S_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "swin_v2_b": (models.swin_v2_b, "Swin_V2_B_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),

    # VGG
    "vgg11":    (models.vgg11,    "VGG11_Weights.IMAGENET1K_V1",    TASK_CLASSIFICATION, DS_IMAGENET1K),
    "vgg11_bn": (models.vgg11_bn, "VGG11_BN_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "vgg13":    (models.vgg13,    "VGG13_Weights.IMAGENET1K_V1",    TASK_CLASSIFICATION, DS_IMAGENET1K),
    "vgg13_bn": (models.vgg13_bn, "VGG13_BN_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "vgg16":    (models.vgg16,    "VGG16_Weights.IMAGENET1K_V1",    TASK_CLASSIFICATION, DS_IMAGENET1K),
    "vgg16_bn": (models.vgg16_bn, "VGG16_BN_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "vgg19":    (models.vgg19,    "VGG19_Weights.IMAGENET1K_V1",    TASK_CLASSIFICATION, DS_IMAGENET1K),
    "vgg19_bn": (models.vgg19_bn, "VGG19_BN_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),

    # VisionTransformer (ViT)
    "vit_b_16": (models.vit_b_16, "ViT_B_16_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "vit_b_32": (models.vit_b_32, "ViT_B_32_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "vit_l_16": (models.vit_l_16, "ViT_L_16_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "vit_l_32": (models.vit_l_32, "ViT_L_32_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "vit_h_14": (models.vit_h_14, "ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1", TASK_CLASSIFICATION, DS_IMAGENET1K_SWAG),

    # Wide ResNet
    "wide_resnet50_2":  (models.wide_resnet50_2,  "Wide_ResNet50_2_Weights.IMAGENET1K_V1",  TASK_CLASSIFICATION, DS_IMAGENET1K),
    "wide_resnet101_2": (models.wide_resnet101_2, "Wide_ResNet101_2_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
}


def recomStart(ctx: object, args: argparse.Namespace):
    """
    Recommendation Engine for Edge AI Models.
    
    This function processes user input preferences to recommend the most optimal model 
    utilizing a three-step pipeline:
    1. Hardware-Aware Matching: Calculates similarity scores between the user's edge hardware 
       and benchmarked target hardware.
    2. Constraint Filtering: Eliminates models that do not fulfill strict boundaries (e.g., latency limits).
    3. TOPSIS Scoring: Applies multi-criteria decision making to rank the remaining models 
       based on accuracy, latency, throughput, and storage footprint.

    Args:
        ctx (object): Application context containing standard configurations and loggers.
        args (argparse.Namespace): Command-line arguments containing the path to the input JSON.
    """

    logger.info("Starting recommendation process based on user preferences.")
    logger.info("="*80)

    path = Path(args.inputfile).expanduser().resolve()

    # 1. Load input data
    logger.info(f"Loading input data from {path}...")
    with path.open("r", encoding="utf-8") as f:
        inputdata = json.load(f)

    logger.info("Input data loaded successfully.")
    
    # Extract basic preferences for logging and filtering
    hw_input = inputdata["hardware"]
    model_input = inputdata["model"]
    metrics_input = inputdata["metrics"]
    constraints_input = inputdata.get("constraints", {}) # Fallback if empty
    
    logger.info("Preferences extracted.")
    logger.info(f"Target Device: {hw_input.get('DEVICE_NAME', 'Unknown')}")
    logger.info(f"Constraints: {constraints_input}")
    logger.info("="*80)

    # 2. Load benchmark database
    logger.info("Loading model benchmark database...")
    db_path = Path(PATH_TO_BENCHMARKDB) # Adjust path if necessary
    with db_path.open("r", encoding="utf-8") as f:
        benchDB = json.load(f)

    logger.info(f"Loaded {len(benchDB)} entries from benchmark database.")
    logger.info("="*80)

    # --- HELPER FUNCTIONS ---

    def parse_numeric(val):
        """
        Extracts numerical values from strings with units (e.g., '4.7 GHz', '16GB', '20').
        
        Args:
            val (Any): Input value, potentially containing alphanumeric characters.
        Returns:
            float: Cleaned floating point representation. Returns 0.0 if extraction fails.
        """
        if isinstance(val, (int, float)):
            return float(val)
        if not val or val == "None":
            return 0.0
        # Remove everything except digits and decimal point
        clean = re.sub(r'[^\d.]', '', str(val))
        try:
            return float(clean)
        except ValueError:
            return 0.0

    def calculate_ratio_score(val_user, val_bench):
        """
        Calculates the similarity ratio between two numerical values (bounded between 0.0 and 1.0).
        A score of 1.0 indicates an exact match.
        """
        v_u = parse_numeric(val_user)
        v_b = parse_numeric(val_bench)
        
        if v_u == 0 or v_b == 0:
            return 0.0
            
        # Calculate ratio (smaller / larger value)
        return min(v_u, v_b) / max(v_u, v_b)

    # --- MAIN FUNCTION FOR SIMILARITY ---

    def calculate_hardware_similarity(user_hw, bench_hw):
        """
        Calculates a detailed similarity score based on hardware specifications.
        Considers CPU (Cores, Clock speed), GPU (VRAM, CUDA Cores), and TPU (TOPS).
        
        Args:
            user_hw (dict): Target edge device specifications provided by the user.
            bench_hw (dict): Hardware specifications of the benchmarked device.
        Returns:
            float: An accumulated similarity score.
        """
        score = 0.0
        
        # 1. Architecture Check (Foundation)
        # If architecture mismatches (e.g., x86 vs ARM), comparison becomes nearly irrelevant
        user_arch = user_hw["CPU"].get("architecture", "").lower()
        bench_arch = bench_hw["CPU"].get("architecture", bench_hw.get("arch", "")).lower()
        
        if user_arch and bench_arch and user_arch == bench_arch:
            score += 10
        elif user_arch in bench_arch or bench_arch in user_arch:
            score += 5
            
        # --- DECISION LOGIC: WHICH ACCELERATOR TO COMPARE? ---
        
        # Logic: We only compare the component utilized for inference tasks.
        
        # Check user intent
        user_wants_gpu = (not user_hw.get("CPU_ONLY", False)) and (not user_hw["GPU"].get("filter", False))
        user_wants_tpu = user_hw["TPU"].get("filter", False) # Assuming TPU filter=True signifies intent to use TPU
        
        # Check what the benchmark utilized
        # Fallback to old schema structure if new keys are missing
        bench_used_cuda = bench_hw["GPU"].get("cuda_cores", 0) > 0 or bench_hw.get("used_cuda", False)
        bench_used_tpu = bench_hw["TPU"].get("accelerator", False) or bench_hw.get("device_name", "").lower().find("tpu") != -1
        bench_cpu_only = not bench_used_cuda and not bench_used_tpu

        # --- COMPARISON LOGIC ---

        # A. GPU COMPARISON (High Weighting)
        if user_wants_gpu and bench_used_cuda:
            score += 30 # Baseline Match: Both utilize a GPU
            
            u_gpu = user_hw["GPU"]
            b_gpu = bench_hw["GPU"]
            
            # 1. VRAM Match (Crucial factor for model feasibility)
            # E.g., User 8GB vs Bench 8GB yields a score of 1.0
            vram_sim = calculate_ratio_score(u_gpu.get("vram"), b_gpu.get("vram"))
            score += 20 * vram_sim
            
            # 2. Compute Power (CUDA Cores)
            cuda_sim = calculate_ratio_score(u_gpu.get("cuda_cores"), b_gpu.get("cuda_cores"))
            score += 15 * cuda_sim
            
            # 3. AI Power (Tensor Cores) - If available
            tensor_sim = calculate_ratio_score(u_gpu.get("tensor_cores"), b_gpu.get("tensor_cores"))
            score += 10 * tensor_sim

        # B. TPU COMPARISON
        elif user_wants_tpu and bench_used_tpu:
            score += 40 # Baseline Match
            
            u_tpu = user_hw["TPU"]
            b_tpu = bench_hw["TPU"]
            
            # TOPS Comparison (Trillion Operations Per Second)
            tops_sim = calculate_ratio_score(u_tpu.get("performance_tops"), b_tpu.get("performance_tops"))
            score += 35 * tops_sim

        # C. CPU COMPARISON (Fallback or explicitly requested)
        # CPU specifications are consistently evaluated to a lesser degree (as system baseline), 
        # but are weighted highly if CPU_ONLY inference is configured.
        else:
            # Case: Both are CPU-Only or a mismatch occurred (User GPU vs Bench CPU)
            cpu_weight_factor = 1.0
            
            if (not user_wants_gpu) and bench_cpu_only:
                score += 30 # Baseline Match: Both rely on CPU
                cpu_weight_factor = 2.0 # CPU Specifications carry doubled importance
            
            u_cpu = user_hw["CPU"]
            b_cpu = bench_hw["CPU"]
            
            # 1. Core Count
            cores_sim = calculate_ratio_score(u_cpu.get("cores"), b_cpu.get("cores"))
            score += (10 * cpu_weight_factor) * cores_sim
            
            # 2. Clock Speed (Max Clock frequency)
            clock_sim = calculate_ratio_score(u_cpu.get("max_clock"), b_cpu.get("max_clock"))
            score += (10 * cpu_weight_factor) * clock_sim

        return score

    def select_best_benchmarks(bench_db, user_hw):
        """
        Groups benchmark data by model name and selects the specific benchmark entry 
        that yields the highest hardware similarity score relative to the user's edge device.
        """
        grouped = {}
        for entry in bench_db:
            name = entry["model_name"]
            if name not in grouped:
                grouped[name] = []
            grouped[name].append(entry)
        
        selected = []
        for name, entries in grouped.items():
            best_entry = None
            best_score = -1
            
            for entry in entries:
                sim_score = calculate_hardware_similarity(user_hw, entry["hardware"])
                if sim_score > best_score:
                    best_score = sim_score
                    best_entry = entry
            
            if best_entry:
                # Append metadata for downstream debugging
                best_entry["_similarity_score"] = best_score
                selected.append(best_entry)
                logger.debug(f"Selected benchmark for '{name}' with hardware similarity: {best_score}")
        
        return selected

    def check_constraints(model, constraints):
        """
        Evaluates hard constraints against benchmark data. 
        Returns False and an error string if a specific constraint is violated.
        
        Args:
            model (dict): A selected model benchmark entry.
            constraints (dict): Dictionary specifying thresholds (accuracy, latency, throughput, storage).
        Returns:
            Tuple[bool, str]: Validation status and corresponding reasoning.
        """
        bench = model["benchmark"]
        artifact = model["artifact"]
        
        # 1. Accuracy (Minimum acceptable threshold)
        if "accuracy" in constraints and constraints["accuracy"] is not None:
            # Note: Benchmarks typically scale 0.0-1.0, while User Input often scales 0-100. Normalization applied.
            acc_val = bench["accuracy_top1"]
            if acc_val <= 1.0: acc_val *= 100 # Scale to percentage
            if acc_val < float(constraints["accuracy"]):
                return False, f"Accuracy too low ({acc_val:.2f} < {constraints['accuracy']})"

        # 2. Latency (Maximum acceptable threshold)
        if "latency" in constraints and constraints["latency"] is not None:
            if bench["latency_ms_per_sample"] > float(constraints["latency"]):
                return False, f"Latency too high ({bench['latency_ms_per_sample']:.2f} > {constraints['latency']})"

        # 3. Throughput (Minimum acceptable threshold)
        if "throughput" in constraints and constraints["throughput"] is not None:
            if bench["throughput_fps"] < float(constraints["throughput"]):
                return False, f"Throughput too low ({bench['throughput_fps']:.2f} < {constraints['throughput']})"

        # 4. Storage (Maximum acceptable threshold) - Optional parameter
        if "storage" in constraints and constraints["storage"] is not None:
             # Assumption: The provided constraint is specified in MB, aligning with the Database format.
             if artifact["disk_size_mb"] > float(constraints["storage"]):
                 return False, f"Model too large ({artifact['disk_size_mb']:.2f} > {constraints['storage']})"
                 
        return True, "OK"

    # --- MAIN EXECUTION WORKFLOW ---

    # STEP 1: Hardware Matching
    # The database is not filtered immediately; instead, the most compatible hardware benchmarks are identified first.
    logger.info("Step 1: Matching benchmarks to user hardware...")
    matched_models = select_best_benchmarks(benchDB, hw_input)
    logger.info(f"Selected {len(matched_models)} best-matching benchmark entries.")


    # STEP 2: Basic Attribute Filtering (Input Specifications)
    logger.info("Step 2: Applying basic model filters...")
    filtered_list = matched_models
    
    # Filter by required precision
    if model_input.get("precision"):
        filtered_list = [m for m in filtered_list if model_input["precision"] in m["precision"]]
    
    # Filter by required quantization technique
    if model_input.get("quantization"):
        filtered_list = [m for m in filtered_list if m["quantization"]["type"] == model_input["quantization"]]

    # Filter by specific model name if requested
    if model_input.get("filter") and model_input.get("name"):
        filtered_list = [m for m in filtered_list if m["model_name"] == model_input["name"]]

    logger.info(f"Models after basic filtering: {len(filtered_list)}")


    # STEP 3: Constraint Checking (Gatekeeper logic)
    valid_models = []
    logger.info("Step 3: Checking hard constraints...")
    
    for m in filtered_list:
        is_valid, reason = check_constraints(m, constraints_input)
        if is_valid:
            valid_models.append(m)
        else:
            logger.debug(f"Model {m['model_name']} rejected: {reason}")
            
    logger.info(f"Models passing constraints: {len(valid_models)}")

    if not valid_models:
        logger.warning("No models found that satisfy all constraints!")
        return # Alternatively, return an empty list here

    # STEP 4: TOPSIS Scoring (Technique for Order of Preference by Similarity to Ideal Solution)
    logger.info("Step 4: Calculating TOPSIS scores...")

    # A. Construct Decision Matrix
    # Columns map to: [Accuracy (+), Latency (-), Storage (-), Throughput (+)]
    # (+) = Benefit criterion (higher is better), (-) = Cost criterion (lower is better)
    data_matrix = []
    for m in valid_models:
        data_matrix.append([
            m["benchmark"]["accuracy_top1"],
            m["benchmark"]["latency_ms_per_sample"],
            m["artifact"]["disk_size_mb"],
            m["benchmark"]["throughput_fps"]
        ])
    
    X = np.array(data_matrix, dtype=np.float64)

    # B. Vector Normalization
    # Prevents zero-division issues utilizing np.clip or epsilon-safety overrides
    column_norms = np.sqrt(np.sum(X**2, axis=0))
    column_norms[column_norms == 0] = 1.0 # Safety fallback
    X_norm = X / column_norms

    # C. Weight Application
    # Normalize user-defined metric weights (Target Sum = 1)
    raw_weights = [
        float(metrics_input["accuracy"]),
        float(metrics_input["inference_speed"]),     # Maps to Latency
        float(metrics_input["storage_consumption"]), # Maps to Storage
        float(metrics_input["throughput"])
    ]
    total_weight = sum(raw_weights)
    if total_weight == 0: total_weight = 1 # Safety fallback
    weights = np.array(raw_weights) / total_weight

    X_weighted = X_norm * weights

    # D. Determine Ideal Best and Ideal Worst Solutions
    # Acc (Index 0): Maximum is Best
    # Lat (Index 1): Minimum is Best
    # Stor (Index 2): Minimum is Best
    # Thr (Index 3): Maximum is Best
    
    ideal_best = [
        np.max(X_weighted[:, 0]), # Max Acc
        np.min(X_weighted[:, 1]), # Min Lat
        np.min(X_weighted[:, 2]), # Min Size
        np.max(X_weighted[:, 3])  # Max Thr
    ]
    
    ideal_worst = [
        np.min(X_weighted[:, 0]), # Min Acc
        np.max(X_weighted[:, 1]), # Max Lat
        np.max(X_weighted[:, 2]), # Max Size
        np.min(X_weighted[:, 3])  # Min Thr
    ]

    # E. Calculate Euclidean Distances
    # axis=1 performs row-wise summation over the criteria columns per candidate model
    dist_best = np.sqrt(np.sum((X_weighted - ideal_best)**2, axis=1))
    dist_worst = np.sqrt(np.sum((X_weighted - ideal_worst)**2, axis=1))

    # F. Compute the Final TOPSIS Score
    # Relative Closeness Score = Dist_Worst / (Dist_Best + Dist_Worst)
    # Values approaching 1.0 indicate proximity to the ideal solution and maximum distance from the worst
    topsis_scores = dist_worst / (dist_best + dist_worst + 1e-9) # Epsilon added for numerical stability

    # Aggregate Results
    final_results = []
    for i, model in enumerate(valid_models):
        final_results.append({
            "model": model["model_name"],
            "score": topsis_scores[i],
            "benchmark_used": model["benchmark"], # Informational data mapped for user visibility
            "hardware_context": model["hardware"]["DEVICE_NAME"] # Traceability metric indicating source of data
        })

    # Execute Ranking
    results_sorted = sorted(final_results, key=lambda x: x['score'], reverse=True)

    # Standard Output Delivery
    logger.info(f"Top recommended models (TOPSIS Ranked):")
    logger.info("-" * 80)
    
    for i, entry in enumerate(results_sorted[:5]): # Limits display to the Top 5 candidates
        logger.info(f"{i+1}. | {entry['model']}")
        logger.info(f"    Score: {entry['score']:.4f}")
        logger.info(f"    Hardware Match: {entry['hardware_context']}")
        logger.info(f"    Metrics: Acc={entry['benchmark_used']['accuracy_top1']:.2f}, "
                    f"Lat={entry['benchmark_used']['latency_ms_per_sample']:.2f}ms, "
                    f"FPS={entry['benchmark_used']['throughput_fps']:.1f}")
        logger.info("-" * 40)

def inferece(ctx: Context, args: argparse.Namespace):
    """
    Executes an inference run on an image using the specified model. This command can be used for functional testing, demonstration purposes and allowing users to validate models usability.
    """
    logger.info("Starting inference process...")
    logger.info("="*80)
    cmd_infer(ctx) 

def download(ctx: Context, args: argparse.Namespace):
    """
    Handles model download processes including normal versions, 
    dynamic quantization, static quantization, and Quantization-Aware Training (QAT).
    """
    # Download routines for all registry models if requested
    # if executed all models and their coresponding quantization variants will be downloaded, if flagged with --... True (exp. --dynamic True)
    args = ctx.args
    if args.all:
        logger.info("Download requested for all models.")
        logger.info("="*80)
        for model in ctx.MODEL_REGISTRY.keys():
            logger.info(f"Download requested for model: {model}")
            ctx.args.model = model
            if args.dynamic == True:
                logger.info(f"Starting dynamic quantization download for model: {model}")
                logger.info("="*50)
                dynamic_quantize(ctx)
                logger.info("="*50)
            if args.static == True:
                logger.info(f"Starting static quantization download for model: {model}")
                logger.info("="*50)
                static_download(ctx)
                logger.info("="*50)
            if args.qat == True:
                logger.info(f"Starting QAT download for model: {model}")
                logger.info("="*50)
                qat_download(ctx)
                logger.info("="*50)
            if args.normal == True:
                logger.info(f"Starting normal download for model: {model}")
                logger.info("="*50)
                cmd_download(ctx)
                logger.info("="*50)

        return   
        
    ctx.logger.info(f"Download requested for model: {args.model}")
    # Normal download routine mapped to a specific target model
    if args.dynamic == True:
        logger.info("="*50)
        logger.info(f"Starting dynamic quantization download for model: {args.model}")
        dynamic_quantize(ctx)
        logger.info("="*50)
    if args.static == True:
        logger.info("="*50)
        logger.info(f"Starting static quantization download for model: {args.model}")
        static_download(ctx)
        logger.info("="*50)
    if args.qat == True:
        logger.info("="*50)
        logger.info(f"Starting QAT download for model: {args.model}")
        qat_download(ctx)
        logger.info("="*50)
    if args.normal == True:
        logger.info("="*50)
        logger.info(f"Starting normal download for model: {args.model}")
        cmd_download(ctx)
        logger.info("="*50)

def get_models(ctx: Context, args: argparse.Namespace):
    """
    Fetches all models from an external server for edge device testing.
    Note: Function architecture ensures --model and --all are mutually exclusive.
    """
    # To implement if external server infrastructure is established for model hosting. This function will handle bulk retrieval of models for edge device testing and validation purposes.
    pass

def test(ctx: Context,args: argparse.Namespace):
    """
    Development function for testing variable context states. (Scheduled for deprecation)
    """
    print(ctx.logger)
    print(ctx.MODEL_REGISTRY)
    print(ctx.args)

def setup_logging():
    """
    Configures standard system logging format and handlers.
    
    Returns:
        logging.Logger: The configured logger instance.
    """
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            #logging.FileHandler('app.log') 
        ]
    )

    logger = logging.getLogger(__name__)
    return logger


def benchmark_accuracy(ctx: Context, args: argparse.Namespace):
    """
    Orchestrates the accuracy benchmarking workflow.
    """
    ctx.logger.info("Benchmarking accuracy...")
    ctx.logger.info("="*80)
    accracy_benchmark(ctx)


logger = setup_logging()

def build_parser(ctx:Context) -> argparse.ArgumentParser:
    """
    Constructs the argparse structure for the Command Line Interface (CLI).
    
    Args:
        ctx (Context): The application context holding registry text for epilog generation.
    Returns:
        argparse.ArgumentParser: Populated parser block.
    """
    model_names = "\n".join(ctx.MODEL_REGISTRY.keys())
    
    epilog_text = textwrap.dedent(f"""
    Available Models:
    -----------------
    {model_names}
    """)
    p = argparse.ArgumentParser(
        description="Edge AI Model Zoo and Recommendation System CLI. Use this tool to recommend, download, benchmark, and run inference on various neural networks.", 
        epilog=epilog_text,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase output verbosity (-v for INFO, -vv for DEBUG).")

    sub = p.add_subparsers(dest="command", required=True)

    # -------------------- recommendation system --------------------
    p_dl = sub.add_parser("recom", help="Recommend optimal edge AI models based on user preferences and constraints. Example: python main.py recom --inputfile input.json")
    p_dl.add_argument("--inputfile", type=str, required=True, help="Path to the JSON configuration file containing hardware specs, metrics, and constraints.")
    p_dl.set_defaults(func=recomStart)

    # -------------------- inference on image --------------------
    p_inf = sub.add_parser("infer", help="Execute inference on a directory of images using a specified model.")
    p_inf.add_argument("--model", type=str, required=True, help="Name of the target model to use for inference (e.g., mobilenet_v2).")
    p_inf.add_argument("--dir", type=str, required=True, help="Path to the directory containing the input images for inference.")
    p_inf.add_argument("--image", type=str, required=True, help="Path to the input image (jpg/png) for inference.")
    p_inf.set_defaults(func=inferece)

    # -------------------- download specified model --------------------
    p_dl_model = sub.add_parser("download", help="Download pre-trained models. Supports standard, static, dynamic, and QAT formats.")
    p_dl_model.add_argument("--model", type=str, required=False, help="Specific model name to download from the registry.")
    p_dl_model.add_argument("--modeldir", type=str, required=False, help="Target directory to save the downloaded model weights.")
    p_dl_model.add_argument("--all", type=str, required=False, default=False, help="Set to True to download all models available in the registry.")

    p_dl_model.add_argument("--normal", type=bool, default=False, help="Download the standard FP32 (unquantized) baseline model.")
    p_dl_model.add_argument("--static", type=bool, default=False, help="Download the statically quantized (INT8) version of the model.")
    p_dl_model.add_argument("--dynamic", type=bool, default=False, help="Download the dynamically quantized version of the model.")
    p_dl_model.add_argument("--qat", type=bool, default=False, help="Download the Quantization-Aware Training (QAT) optimized model.")
    p_dl_model.set_defaults(func=download)

    # -------------------- fetch all models from server (for edge device testing) --------------------
    p_get = sub.add_parser("get", help="Fetch models from a remote server for edge device deployment and testing.")
    p_get.add_argument("--model", type=str, required=True, help="Name of the specific model to fetch from the server.")
    p_get.add_argument("--all", type=str, required=False, default=False, help="Set to True to fetch all available models from the remote server.")
    p_get.set_defaults(func=get_models)

    # -------------------- dev testing purposes (delete later) --------------------
    p_test = sub.add_parser("test", help="Internal testing and debugging command. Prints current context and arguments.")
    p_test.add_argument("--model", type=str, required=False, help="Target model name for the test execution.")
    p_test.add_argument("--all", type=str, required=False, default=False, help="Run the test sequence across all models.")
    p_test.set_defaults(func=test)

    # ---------------------------benchmark accuracy--------------------------------------
    p_bench = sub.add_parser("accbenchmark", help="Run accuracy benchmarks on downloaded models against standard datasets.")
    p_bench.add_argument("--modeldir", type=str, required=False, help="Directory containing the downloaded models to be benchmarked.")
    p_bench.add_argument("--logpath", type=str, required=False, help="File path to save the generated benchmark logs and results.")
    p_bench.add_argument("--samplesize", type=int, required=False, help="Number of images/samples to evaluate during the accuracy benchmark.")
    p_bench.add_argument("--all", type=bool, required=False, default=False, help="Set to True to benchmark all models found in the specified model directory.")
    p_bench.set_defaults(func=benchmark_accuracy)

    return p

def main():
    """
    Main entry point for the application. Evaluates CLI commands and invokes target modules.
    """
    ctx = Context(logger= setup_logging(), registry= MODEL_REGISTRY)
    parser = build_parser(ctx)
    args = parser.parse_args()
    #setup_logging(args.verbose)
    ctx.args=args
    args.func(ctx, args)

if __name__ == "__main__":
    main()