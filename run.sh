# train baselines
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --use_env opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/skylink/skylink_intermediate_bm2cp.yaml --tag 'baseline' --worker 32

# sicp training
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --use_env opencood/tools/train_sicp_multiclass.py --hypes_yaml opencood/hypes_yaml/skylink/skylink_intermediate_sicp.yaml --tag 'baseline' --worker 32

# evaluate
CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference.py --model_dir opencood/logs/skylink_intermediate/test__2025_04_26_16_38_48 --eval_epoch 21  --save_vis 1

# tensorboard vis
tensorboard --logdir opencood/logs/ --port 10000 --bind_all