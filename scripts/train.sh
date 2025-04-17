#!/bin/bash

# 设置环境变量
# official
#export WANDB_API_KEY=83793606f810aa3d385ea5d12dbd352514ac54e1

# local
export WANDB_API_KEY=local-73e66a04fa97e2a5d5c573a97e65bf1194533e1f
export WANDB_BASE_URL="http://10.28.0.22:30437"
# export CUDA_LAUNCH_BLOCKING=1

# 执行 torchrun 命令
/ssdwork/liujinxin/miniconda3/envs/cogact/bin/torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node 6 \
    "${PWD}/scripts/train.py" \
    --pretrained_checkpoint "/liujinxin/code/CogACT/models/CogACT-Base/checkpoints/CogACT-Base.pt" \
    --vla.type "prism-dinosiglip-224px+oxe+diffusion" \
    --vla.data_mix "agibot_1k" \
    --vla.expected_world_size 6 \
    --vla.global_batch_size 120 \
    --vla.per_device_batch_size 20 \
    --vla.learning_rate 2e-5 \
    --data_root_dir "/ssdwork/liujinxin/agibot_rlds" \
    --run_root_dir "./logs" \
    --run_id "agibot_1k" \
    --image_aug "True" \
    --wandb_project "CogACT_dual" \
    --wandb_entity "weizhao" \
    --save_interval 5000 \
    --repeated_diffusion_steps 8 \
    --future_action_window_size 15 \
    --action_model_type "DiT-B" \
    --is_resume "False"
