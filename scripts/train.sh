#!/bin/bash

# 设置环境变量
export WANDB_API_KEY=83793606f810aa3d385ea5d12dbd352514ac54e1
# export CUDA_LAUNCH_BLOCKING=1

# 执行 torchrun 命令
/ssdwork/liujinxin/miniconda3/envs/cogact/bin/torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node 2 \
    "${PWD}/scripts/train.py" \
    --pretrained_checkpoint "/liujinxin/zhaowei/CogACT/models/CogACT-Base/checkpoints/CogACT-Base.pt" \
    --vla.type "prism-dinosiglip-224px+oxe+diffusion" \
    --vla.data_mix "calvin_abc2d_oe" \
    --vla.expected_world_size 2 \
    --vla.global_batch_size 2 \
    --vla.per_device_batch_size 1 \
    --vla.learning_rate 2e-5 \
    --data_root_dir "/liujinxin/zhaowei/rlds_dataset_builder/tensorflow_datasets" \
    --run_root_dir "./logs" \
    --run_id "debug_oe" \
    --image_aug "False" \
    --wandb_project "CogACT_debug_oe" \
    --wandb_entity "yijiulanpishu" \
    --save_interval 5000 \
    --repeated_diffusion_steps 8 \
    --future_action_window_size 9 \
    --action_model_type "DiT-B" \
    --is_resume "False" \
