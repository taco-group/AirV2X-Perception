# AirV2X‑Perception

**Official implementation of**  
**“AirV2X: Unified Air–Ground/Vehicle‑to‑Everything Collaboration for Perception”**


<!-- [![Paper](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/2506.19283)
[![Project Page](https://img.shields.io/badge/Project-Page-1f72ff.svg)](https://xiangbogaobarry.github.io/AirV2X/)
[![Code](https://img.shields.io/badge/GitHub-Code-black.svg)](https://github.com/taco-group/AirV2X-Perception)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow.svg)](https://huggingface.co/datasets/xiangbog/AirV2X-Perception) 
-->

---

## 🌐 Dataset

Download **AirV2X‑Perception** from Hugging Face and extract it to any location:

```bash
mkdir dataset
cd dataset # Use another directory to avoid naming conflict
conda install -c conda-forge git-lfs
git lfs install --skip-smudge
git clone https://huggingface.co/datasets/xiangbog/AirV2X-Perception
cd AirV2X-Perception
git lfs pull
# git lfs pull --include "path/to/folder"   # If you would like to download only partial of the dataset
```

We also provide a *mini batch* for quick testing and debugging.

---

## 🔧 Installation

Detailed instructions and environment specifications are in [`doc/INSTALL.md`](doc/INSTALL.md).

---

## 🚀 Model Training

### Single‑GPU

```bash
python opencood/tools/train.py \
    -y /path/to/config_file.yaml
```

Example: train **Where2Comm** (LiDAR‑only)

```bash
python opencood/tools/train.py \
    -y opencood/hypes_yaml/airv2x/lidar/det/airv2x_intermediate_where2com.yaml
```

> **Tip**  
> Some models such as **V2X‑ViT** and **CoBEVT** consume a large amount of VRAM.  
> Enable mixed‑precision with `--amp` if you encounter OOM, but watch out for *NaN/Inf* instability.

```bash
python opencood/tools/train.py \ 
    -y opencood/hypes_yaml/airv2x/lidar/det/airv2x_intermediate_v2xvit.yaml       
    --amp
```

### Multi‑GPU (DDP)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --standalone --nproc_per_node=4 \     
    opencood/tools/train.py \
        -y /path/to/config_file.yaml
```

Example: LiDAR‑only **Where2Comm** with 8 GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \       
    --standalone\
    --nproc_per_node=8 \
    opencood/tools/train.py \
        -y opencood/hypes_yaml/airv2x/lidar/det/airv2x_intermediate_where2com.yaml
```

### Multi‑Stage Models (HEAL, STAMP)

These models were trained on **2 nodes × 1 GPU (batch size 1)**.  
If you change the number of GPUs or batch size, adjust the learning rate accordingly.

---

## 📝 Evaluation

```bash
python opencood/tools/inference_multi_scenario.py \ 
    --model_dir opencood/logs/airv2x_intermediate_where2comm/default__2025_07_10_09_17_28 \
    --eval_best_epoch \
    --save_vis
```

---

## 📦 Pre‑Trained Checkpoints

| Modality | Model | AP@0.3 | AP@0.5 | AP@0.7 | Config & Checkpoint |
|---|---|---|---|---|---|
| **LiDAR** | When2Com | 0.1824 | 0.0787 | 0.0025 | *Coming soon* |
|  | CoBEVT | **0.4922** | **0.4585** | **0.2582** | [HF](https://huggingface.co/xiangbog/AirV2X-Perception-Checkpoints/tree/main/airv2x_intermediate_cobevt/release) |
|  | Where2Comm | 0.4366 | 0.4015 | 0.1538 | [HF](https://huggingface.co/xiangbog/AirV2X-Perception-Checkpoints/tree/main/airv2x_intermediate_where2comm/release) |
|  | V2X‑ViT | 0.4401 | 0.3821 | 0.1638 | [HF](https://huggingface.co/xiangbog/AirV2X-Perception-Checkpoints/tree/main/airv2x_intermediate_v2xvit/release) |

> **Note**  
> The above checkpoints were trained for **50 epochs** with `batch_size=2`, so their numbers may differ slightly from the paper.

Additional Camera‑only and Multi‑modal checkpoints are on the way.

---

## 🔍 Visualization

```bash
tensorboard --logdir opencood/logs --port 10000 --bind_all
```

---

## 📄 Citation

```bibtex
@article{gao2025airv2x,
  title   = {AirV2X: Unified Air--Ground/Vehicle-to-Everything Collaboration for Perception},
  author  = {Gao, Xiangbo and Tu, Zhengzhong and others},
  journal = {arXiv preprint arXiv:2506.19283},
  year    = {2025}
}
```

---

We will continuously update this repository with code, checkpoints, and documentation.  
Feel free to open issues or pull requests — contributions are welcome! 🚀
