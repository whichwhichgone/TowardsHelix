#!/bin/bash

CUDA_VISIBLE_DEVICES="0,1,2,3,4"
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    port=$(($IDX+9002))
    echo "Running port $port on GPU ${GPULIST[$IDX]}"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python serve/flask_server.py \
        --model-path /liujinxin/code/CogACT/logs/calvin_cogact/checkpoints/step-056000-epoch-01-loss=0.0687.pt \
        --port $port &
done

wait