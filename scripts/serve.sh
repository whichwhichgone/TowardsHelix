#!/bin/bash

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

# Stagger the startup so the processes do not read the checkpoint from disk at the same time.
STAGGER_SECONDS=${STAGGER_SECONDS:-60}

for IDX in $(seq 0 $((CHUNKS-1))); do
    port=$(($IDX+9002))
    if [ $IDX -gt 0 ]; then
        echo "Waiting ${STAGGER_SECONDS}s before starting the next server"
        sleep $STAGGER_SECONDS
    fi
    echo "Running port $port on GPU ${GPULIST[$IDX]}"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python serve/flask_server.py \
        --model-path "/zhaowei/workspace/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_3dots_hightol/checkpoints/step-016726-epoch-02-loss=0.2463.pt" \
        --unnorm-key "calvin_abc2d_oe_latact" \
        --future-action-window-size 9 \
        --port $port &
done

wait