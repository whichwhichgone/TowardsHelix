#!/bin/bash

# 设置环境变量
export WANDB_API_KEY=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
# export CUDA_LAUNCH_BLOCKING=1

# 执行 torchrun 命令
/ssdwork/liujinxin/miniconda3/envs/cogact_zhaowei/bin/torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node 8 \
    "${PWD}/scripts/train.py" \
    --vla.type "qwen3-vl-8b+oxe+diffusion" \
    --vla.freeze_vision_backbone "false" \
    --vla.freeze_llm_backbone "false" \
    --vla.data_mix "fractal20220817_data" \
    --vla.expected_world_size 8 \
    --vla.global_batch_size 128 \
    --vla.per_device_batch_size 16 \
    --vla.learning_rate 1.5e-5 \
    --vla.lr_scheduler_type "linear-warmup+cosine-decay" \
    --vla.epochs 3 \
    --data_root_dir "/defaultShare/zhaowei_global/flex_vla_data/fractal20220817_data_oe_latact_traj" \
    --run_root_dir "./logs" \
    --run_id "fractal20220817_data_oe_latact_traj" \
    --image_aug "False" \
    --wandb_project "iclr_flex_pretrain" \
    --wandb_entity "weizhao0817-westlake-university" \
    --save_interval 100000 \
    --repeated_diffusion_steps 8 \
    --future_action_window_size 9 \
    --action_model_type "DiT-B" \
    --is_resume "false" \
