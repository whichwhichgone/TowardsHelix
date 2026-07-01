#!/bin/bash

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    port=$(($IDX+9002))
    echo "Running port $port on GPU ${GPULIST[$IDX]}"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python serve/flask_server.py \
        --model-path "/zhaowei/workspace/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_5dots_tail/checkpoints/step-024042-epoch-03-loss=0.3572.pt" \
        --unnorm-key "calvin_abc2d_oe_latact" \
        --future-action-window-size 9 \
        --port $port &
done

wait