#!/bin/bash

# 设置环境变量
export WANDB_API_KEY=83793606f810aa3d385ea5d12dbd352514ac54e1
# export CUDA_LAUNCH_BLOCKING=1

# 执行 torchrun 命令
/ssdwork/liujinxin/miniconda3/envs/cogact/bin/torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node 5 \
    "${PWD}/scripts/train.py" \
    --pretrained_checkpoint "/liujinxin/code/CogACT/logs/ur5e_benchmark_v0_cogact/checkpoints/step-005000-epoch-03-loss=0.0429.pt" \
    --vla.type "prism-dinosiglip-224px+oxe+diffusion" \
    --vla.data_mix "ur5e_benchmark_v0" \
    --vla.expected_world_size 5 \
    --vla.global_batch_size 80 \
    --vla.per_device_batch_size 16 \
    --vla.learning_rate 2e-5 \
    --data_root_dir "/liujinxin/code/rlds_dataset_builder/tensorflow_datasets" \
    --run_root_dir "./logs" \
    --run_id "ur5e_benchmark_v0_cogact" \
    --image_aug "False" \
    --wandb_project "nips2025_rebuttal" \
    --wandb_entity "yijiulanpishu" \
    --save_interval 5000 \
    --repeated_diffusion_steps 8 \
    --future_action_window_size 9 \
    --action_model_type "DiT-B" \
    --is_resume "True" \
    --resume_step 5000 \
    --resume_epoch 3 \
