# Quantizing Pre-Trained Models for Heterogeneous Edge Devices

This repository was created for the implemenation that was part of the bachelor thesis "Quantizing Pre-Trained Models for Heterogeneous Edge Devices". The goal is to create a scoring framework that bridges the gap between raw benchmark data, optimizing models and deployement. The framework can be set up as follows:

## 1. Setup Environment
You need **Miniconda** (or Anaconda) to run this project. Install all dependencies using the provided environment file:

```bash
conda env create -f environment.yml
conda activate <your_env_name>
```

You might run into errors regarding mismatches of dependency versions. In this case make sure to use Python version 3.8.20. If further errors occur, feel free to reach out.

## 2. Prepare Dataset (For Quantization)
If you want to use **Static Quantization** or **Quantization Aware Training (QAT)**, you need the ImageNet dataset for recalibration.

1. Create a data folder:
   ```bash
   mkdir data/
   ```
2. Download ImageNet-1k via Hugging Face:
   ```bash
   huggingface-cli download imagenet-1k --repo-type dataset --local-dir data/
   ```
*Note: If you use a different folder name than `data/`, you must update `PATH_SET_DATA` in `static_quant.py` and `DATASET_PATH` in `qat.py`.*

## 3. Create Model Folder
We recommend creating a folder to store all your downloaded models:
```bash
mkdir modelzoo/
```

---

## 4. Commands Overview
Use `python main.py <command>` to run the tool. You can always add `-v` for more detailed terminal output.

### Recommend a Model
Find the best model based on your hardware constraints (provided via a JSON file). Inoder to load a benchmarking database change the Variable PATH_TO_BENCHMARKDB in the main.py to the location of the database. It's recommended to place it in the utilit_files/ folder. A template for the input as well as the input file and database for the proof of concept section in chapter 4 is included. The database should be one json array containing json objects, similar to the specified in the accuracy_benchmark/accuracy_benchmark.py file.
```bash
python main.py recom --inputfile constraints.json
```

### Download Models
Download pretrained models. You can specify if you want the standard model or a quantized version.
```bash
python main.py download --model resnet18 --modeldir modelzoo/ --static True
```
*Options: `--normal True`, `--static True`, `--dynamic True`, `--qat True`, `--all True`*

### Benchmark Accuracy
Test how accurate your downloaded models are. Before benchmarking adjust the hardwarespecs inside the accuracy_benchmark.py file.
```bash
python main.py accbenchmark --modeldir modelzoo/ --samplesize 500 --logpath results.txt
```

### Run Inference
Test a model on a specific image or a folder of images.
```bash
python main.py infer --model resnet18 --dir ./images --image ./images/test.jpg
```
---
###Disclaimer

The code in this repository was written by me, but includes common quantization and algorithmic workflows. If code parts show similarities with your code, it is pure accident. In case of violation, please contact me instatly!