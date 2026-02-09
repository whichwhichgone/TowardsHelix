#!/bin/bash

# 设置环境变量
export WANDB_API_KEY=83793606f810aa3d385ea5d12dbd352514ac54e1
# export CUDA_LAUNCH_BLOCKING=1

# 执行 torchrun 命令
/zhaowei/miniconda3/envs/cogact/bin/torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node 8 \
    "${PWD}/scripts/train.py" \
    --pretrained_checkpoint "/zhaowei/workspace/CogACT/logs/piper_oe_crossattn--image_aug/checkpoints/step-007620-epoch-03-loss=0.0087.pt" \
    --vla.type "prism-dinosiglip-224px+oxe+diffusion" \
    --vla.data_mix "piper_oe" \
    --vla.expected_world_size 8 \
    --vla.global_batch_size 128 \
    --vla.per_device_batch_size 16 \
    --vla.learning_rate 1.5e-5 \
    --vla.lr_scheduler_type "linear-warmup+cosine-decay" \
    --vla.epochs 6 \
    --data_root_dir "/zhaowei/data/piper_oe_new" \
    --run_root_dir "./logs" \
    --run_id "piper_oe_crossattn" \
    --image_aug "True" \
    --wandb_project "cogact_new_icml" \
    --wandb_entity "yijiulanpishu" \
    --save_interval 10000 \
    --repeated_diffusion_steps 8 \
    --future_action_window_size 15 \
    --action_model_type "DiT-B" \
    --is_resume "True" \
    --resume_step 7620 \
    --resume_epoch 3 \
