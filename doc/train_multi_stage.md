HEAL and STAMP requires to stage training. We slightly simplifed the training process, that is, for both model, we train each agent seperately and then train all the model collaboratively. For official implementation, please refer to (STAMP){https://github.com/taco-group/STAMP} and (HEAL){https://github.com/yifanlu0227/HEAL?tab=readme-ov-file}.

For training each agent seperately, taking STAMP as an example, please run
```bash
CUDA_VISIBLE_DEVICES='0,1' torchrun --standalone --nproc_per_node=2 opencood/tools/train.py -y opencood/hypes_yaml/airv2x/lidar/det/airv2x_stamp/single/airv2x_stamp_vehicle_lidar.yaml

CUDA_VISIBLE_DEVICES='2,3' torchrun --standalone --nproc_per_node=2 opencood/tools/train.py -y opencood/hypes_yaml/airv2x/lidar/det/airv2x_stamp/single/airv2x_stamp_rsu_lidar.yaml

CUDA_VISIBLE_DEVICES='4,5' torchrun --standalone --nproc_per_node=2 opencood/tools/train.py -y opencood/hypes_yaml/airv2x/lidar/det/airv2x_stamp/single/airv2x_stamp_drone_lidar.yaml
```
After training finished for each agent, please run the following to train all agents collaboratively.
```bash
CUDA_VISIBLE_DEVICES='4,5' torchrun --standalone --nproc_per_node=2 opencood/tools/train_stamp.py -y opencood/hypes_yaml/airv2x/lidar/det/airv2x_stamp/airv2x_stamp_collab_lidar.yaml    --vehicle_dir opencood/logs/airv2x_HEAL_vehicle_lidar/default__2025_07_11_11_17_34 --rsu_dir opencood/logs/airv2x_HEAL_rsu_lidar/default__2025_07_11_11_14_22 --drone_dir opencood/logs/airv2x_HEAL_drone_lidar/default__2025_07_11_11_16_23
```


