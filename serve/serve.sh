#!/bin/bash

CUDA_VISIBLE_DEVICES="0,1,2,3"
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    port=$(($IDX+9002))
    echo "Running port $port on GPU ${GPULIST[$IDX]}"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python serve/flask_server.py \
        --model-path "/liujinxin/zhaowei/CogACT/logs/calvin_cogact_abc2d/checkpoints/step-033492-epoch-01-loss=0.0494.pt" \
        --future-action-window-size 9 \
        --port $port &
done

wait