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
    --vla.type "qwen3-vl-8b+oxe+diffusion" \
    --vla.data_mix "calvin_abc2d_oe" \
    --vla.expected_world_size 8 \
    --vla.global_batch_size 128 \
    --vla.per_device_batch_size 16 \
    --vla.learning_rate 1.5e-5 \
    --vla.lr_scheduler_type "linear-warmup+cosine-decay" \
    --vla.epochs 3 \
    --data_root_dir "/zhaowei/data/flex_vla_data" \
    --run_root_dir "./logs" \
    --run_id "calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8" \
    --image_aug "False" \
    --wandb_project "nips2026_flex" \
    --wandb_entity "yijiulanpishu" \
    --save_interval 10000 \
    --repeated_diffusion_steps 8 \
    --future_action_window_size 9 \
    --action_model_type "DiT-B" \
    --is_resume "false" \
