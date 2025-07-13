Official implementation of AirV2X: Unified Air-Ground\\Vehicle-to-Everything Collaboration



[![Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://www.arxiv.org/abs/2506.19283) 
[![Project Page](https://img.shields.io/badge/Project-Page-1f72ff.svg)](https://xiangbogaobarry.github.io/AirV2X/) 
[![Code](https://img.shields.io/badge/GitHub-Code-black.svg)](https://github.com/taco-group/AirV2X-Perception)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow.svg)](https://huggingface.co/datasets/xiangbog/AirV2X-Perception)


Please download the dataset at [AirV2X-Perception](https://huggingface.co/datasets/xiangbog/AirV2X-Perception). We provide mini dataset batch for test training.

The training and evaluation codes are released.

The usage instructions will be provided soon!

## Installation

Please refer to (doc/INSTALL.md){doc/INSTALL.md} for installation guildance.

## Model Training

### Single GPU
```bash
python opencood/tools/train.py -y /path/to/config_file
```

For exmaple, for training where2com with lidar only, run the following
```bash
python opencood/tools/train.py -y opencood/hypes_yaml/airv2x/lidar/det/airv2x_intermediate_where2com.yaml
```

Note that some model such as V2X-ViT and CobevT may have high VRAM usage. Consider using `--amp` for mix-precision trainning. However, mix-precision model may cause precision overflow (nan, inf) for some models, so be careful.
```bash
python opencood/tools/train.py -y opencood/hypes_yaml/airv2x/lidar/det/airv2x_intermediate_v2xvit.yaml --amp
```

### Multi GPU

For multi GPU training, please run the following
```bash
CUDA_VISIBLE_DEVICES=gpu_idx torchrun --standalone --nproc_per_node=num_gpus opencood/tools/train.py -y /path/to/config_file
```
For exmaple, for training lidar only where2com with 8 GPUs, run the following
```bash
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' torchrun --standalone --nproc_per_node=8 opencood/tools/train.py -y opencood/hypes_yaml/airv2x/lidar/det/airv2x_intermediate_where2com.yaml
```

### Training Multi-Stage Model (HEAL, STAMP)



Note that all the models are trainined using 2 GPUs (Nodes) with batchsize=1 for each node. For model training with different GPU numbers or batchsize, please change the batchsize and learning rate coorespodiningly.

## Evaluation

```bash
python opencood/tools/inference_multi_scenario.py --model_dir opencood/logs/airv2x_intermediate_where2comm/default__2025_07_10_09_17_28 --eval_best_epoch --save_vis
```

The checkpoints of some baseline models are provided as follow:

<!-- Please convert this into table format -->
### Lidar only

TBD

### Camera only

TBD

### Lidar Camera Fusion

TBD


## Visualization
```bash
tensorboard --logdir opencood/logs/ --port 10000 --bind_all
```