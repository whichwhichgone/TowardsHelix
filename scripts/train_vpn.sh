#!/bin/bash

# Change to the mvp directory and launch clash script in background
cd /liujinxin/zhaowei/mvp
bash /liujinxin/zhaowei/mvp/launch_clash.sh &

# Wait a moment for clash to initialize
sleep 5

# Set up terminal proxy
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

# Run training script
cd /liujinxin/code/CogACT_speedup
bash /liujinxin/code/CogACT_speedup/scripts/train.sh
