#!/bin/bash

# 设置环境变量
export WANDB_API_KEY=83793606f810aa3d385ea5d12dbd352514ac54e1
# export CUDA_LAUNCH_BLOCKING=1

# 执行 torchrun 命令
/ssdwork/liujinxin/miniconda3/envs/cogact/bin/torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node 4 \
    "${PWD}/scripts/train.py" \
    --pretrained_checkpoint "/liujinxin/code/CogACT/models/CogACT-Base/checkpoints/CogACT-Base.pt" \
    --vla.type "prism-dinosiglip-224px+oxe+diffusion" \
    --vla.data_mix "piper_table_mix" \
    --vla.expected_world_size 4 \
    --vla.global_batch_size 128 \
    --vla.per_device_batch_size 32 \
    --vla.learning_rate 2e-5 \
    --data_root_dir "/liujinxin/code/rlds_dataset_builder/tensorflow_datasets" \
    --run_root_dir "./logs" \
    --run_id "piper_table_mix" \
    --image_aug "True" \
    --wandb_project "cogact_piper_single" \
    --wandb_entity "yijiulanpishu" \
    --save_interval 10000 \
    --repeated_diffusion_steps 8 \
    --future_action_window_size 15 \
    --action_model_type "DiT-B" \
    --is_resume "False"
