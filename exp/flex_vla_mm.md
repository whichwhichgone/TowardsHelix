# Flex VLA 多模态指令实验记录（新）

## 实验记录 - 2026-09-08

### 服务端配置

来源: `scripts/serve.sh`

- **模型 checkpoint**: `/defaultShare/zhaowei_global/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_h800/checkpoints/step-024474-epoch-03-loss=0.1728.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_h800`
  - Step: 024474
  - Epoch: 03
  - Loss: 0.1728
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增（进程间启动间隔 60s）

### 客户端配置

评测脚本: `evaluate_policy_multiserver_interleave_mmins.py`

- **instruction_type**: `goal_image`
- **question_file**: `/home/zw/workspace/calvin/calvin_models/calvin_agent/evaluation/question_mm_bench_v1.json`
- **instruction_path**: `/home/zw/workspace/calvin/calvin_models/conf/annotations/interleaved/goal_new_playtable_validation.yaml`
- **custom_model**: 开启
- 按 chunk 并行评测，每个 chunk 对应一台服务端口

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 4.768

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 99.6% |
| 2 | 97.2% |
| 3 | 94.4% |
| 4 | 93.6% |
| 5 | 92.0% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 23 / 23 | 100.0% |
| move_slider_right | 77 / 80 | 96.2% |
| lift_red_block_slider | 38 / 39 | 97.4% |
| place_in_slider | 79 / 82 | 96.3% |
| turn_off_lightbulb | 38 / 38 | 100.0% |
| push_blue_block_left | 14 / 14 | 100.0% |
| close_drawer | 70 / 70 | 100.0% |
| push_red_block_left | 17 / 18 | 94.4% |
| lift_blue_block_table | 58 / 59 | 98.3% |
| turn_on_lightbulb | 49 / 49 | 100.0% |
| move_slider_left | 69 / 69 | 100.0% |
| rotate_red_block_left | 21 / 22 | 95.5% |
| turn_off_led | 56 / 56 | 100.0% |
| lift_pink_block_table | 31 / 32 | 96.9% |
| stack_block | 25 / 27 | 92.6% |
| open_drawer | 95 / 95 | 100.0% |
| place_in_drawer | 75 / 75 | 100.0% |
| lift_pink_block_drawer | 6 / 6 | 100.0% |
| rotate_pink_block_right | 7 / 8 | 87.5% |
| rotate_blue_block_left | 20 / 20 | 100.0% |
| lift_red_block_table | 41 / 42 | 97.6% |
| push_red_block_right | 11 / 11 | 100.0% |
| push_pink_block_left | 16 / 16 | 100.0% |
| rotate_red_block_right | 21 / 22 | 95.5% |
| lift_pink_block_slider | 40 / 40 | 100.0% |
| turn_on_led | 54 / 54 | 100.0% |
| rotate_pink_block_left | 19 / 20 | 95.0% |
| push_pink_block_right | 16 / 17 | 94.1% |
| push_into_drawer | 30 / 30 | 100.0% |
| lift_blue_block_drawer | 9 / 9 | 100.0% |
| push_blue_block_right | 13 / 13 | 100.0% |
| lift_blue_block_slider | 38 / 39 | 97.4% |
| lift_red_block_drawer | 5 / 6 | 83.3% |
| unstack_block | 11 / 11 | 100.0% |

## 实验记录 - 2026-09-08 (2)

### 服务端配置

来源: `scripts/serve.sh`

- **模型 checkpoint**: `/defaultShare/zhaowei_global/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_h800/checkpoints/step-024474-epoch-03-loss=0.1728.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_h800`
  - Step: 024474
  - Epoch: 03
  - Loss: 0.1728
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增（进程间启动间隔 60s）

### 客户端配置

评测脚本: `evaluate_policy_multiserver_interleave_mmins.py`

- **instruction_type**: `imitation_video`
- **question_file**: `/home/zw/workspace/calvin/calvin_models/calvin_agent/evaluation/question_mm_bench_v1.json`
- **instruction_path**: `/home/zw/workspace/calvin/calvin_models/conf/annotations/interleaved/video_new_playtable_validation.yaml`
- **custom_model**: 开启
- 按 chunk 并行评测，每个 chunk 对应一台服务端口

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 4.76

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 99.6% |
| 2 | 96.4% |
| 3 | 94.0% |
| 4 | 93.6% |
| 5 | 92.4% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 23 / 23 | 100.0% |
| move_slider_right | 78 / 78 | 100.0% |
| lift_red_block_slider | 38 / 39 | 97.4% |
| place_in_slider | 80 / 80 | 100.0% |
| turn_off_lightbulb | 38 / 38 | 100.0% |
| push_blue_block_left | 14 / 14 | 100.0% |
| close_drawer | 69 / 69 | 100.0% |
| push_red_block_left | 17 / 18 | 94.4% |
| lift_blue_block_table | 57 / 60 | 95.0% |
| turn_on_lightbulb | 49 / 49 | 100.0% |
| move_slider_left | 70 / 70 | 100.0% |
| rotate_red_block_left | 22 / 22 | 100.0% |
| turn_off_led | 54 / 54 | 100.0% |
| lift_pink_block_table | 31 / 32 | 96.9% |
| stack_block | 24 / 29 | 82.8% |
| open_drawer | 97 / 97 | 100.0% |
| place_in_drawer | 74 / 74 | 100.0% |
| lift_pink_block_drawer | 7 / 7 | 100.0% |
| rotate_pink_block_right | 7 / 8 | 87.5% |
| rotate_blue_block_left | 20 / 20 | 100.0% |
| lift_red_block_table | 40 / 42 | 95.2% |
| push_red_block_right | 11 / 11 | 100.0% |
| push_pink_block_left | 16 / 16 | 100.0% |
| rotate_red_block_right | 20 / 22 | 90.9% |
| lift_pink_block_slider | 42 / 42 | 100.0% |
| turn_on_led | 54 / 54 | 100.0% |
| rotate_pink_block_left | 20 / 20 | 100.0% |
| push_pink_block_right | 16 / 17 | 94.1% |
| push_into_drawer | 29 / 30 | 96.7% |
| lift_blue_block_drawer | 7 / 7 | 100.0% |
| push_blue_block_right | 13 / 13 | 100.0% |
| lift_blue_block_slider | 36 / 37 | 97.3% |
| lift_red_block_drawer | 6 / 6 | 100.0% |
| unstack_block | 11 / 11 | 100.0% |

## 实验记录 - 2026-09-08 (3)

### 服务端配置

来源: `scripts/serve.sh`

- **模型 checkpoint**: `/defaultShare/zhaowei_global/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_h800/checkpoints/step-024474-epoch-03-loss=0.1728.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_h800`
  - Step: 024474
  - Epoch: 03
  - Loss: 0.1728
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增（进程间启动间隔 60s）

### 客户端配置

评测脚本: `evaluate_policy_multiserver_interleave_mmins.py`

- **instruction_type**: `text`
- **question_file**: `/home/zw/workspace/calvin/calvin_models/calvin_agent/evaluation/question_mm_bench_v1.json`
- **instruction_path**: `/home/zw/workspace/calvin/calvin_models/conf/annotations/new_playtable_validation.yaml`
- **custom_model**: 开启
- 按 chunk 并行评测，每个 chunk 对应一台服务端口

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 4.656

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 97.6% |
| 2 | 96.0% |
| 3 | 92.8% |
| 4 | 90.8% |
| 5 | 88.4% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 23 / 23 | 100.0% |
| move_slider_right | 77 / 77 | 100.0% |
| lift_red_block_slider | 37 / 37 | 100.0% |
| place_in_slider | 74 / 77 | 96.1% |
| turn_off_lightbulb | 40 / 40 | 100.0% |
| push_blue_block_left | 14 / 14 | 100.0% |
| close_drawer | 69 / 69 | 100.0% |
| push_red_block_left | 17 / 17 | 100.0% |
| lift_blue_block_table | 55 / 55 | 100.0% |
| turn_on_lightbulb | 46 / 47 | 97.9% |
| move_slider_left | 71 / 71 | 100.0% |
| rotate_red_block_left | 22 / 22 | 100.0% |
| turn_off_led | 54 / 54 | 100.0% |
| lift_pink_block_table | 30 / 31 | 96.8% |
| stack_block | 25 / 27 | 92.6% |
| open_drawer | 93 / 93 | 100.0% |
| place_in_drawer | 73 / 73 | 100.0% |
| lift_pink_block_drawer | 7 / 7 | 100.0% |
| rotate_pink_block_right | 8 / 8 | 100.0% |
| rotate_blue_block_left | 20 / 20 | 100.0% |
| lift_red_block_table | 40 / 41 | 97.6% |
| push_red_block_right | 9 / 11 | 81.8% |
| rotate_red_block_right | 21 / 22 | 95.5% |
| lift_pink_block_slider | 38 / 40 | 95.0% |
| turn_on_led | 57 / 57 | 100.0% |
| rotate_pink_block_left | 20 / 20 | 100.0% |
| push_pink_block_right | 11 / 17 | 64.7% |
| push_into_drawer | 30 / 31 | 96.8% |
| lift_blue_block_drawer | 9 / 9 | 100.0% |
| push_blue_block_right | 10 / 13 | 76.9% |
| push_pink_block_left | 12 / 16 | 75.0% |
| lift_blue_block_slider | 35 / 37 | 94.6% |
| lift_red_block_drawer | 6 / 6 | 100.0% |
| unstack_block | 11 / 11 | 100.0% |

## 实验记录 - 2026-09-09

### 服务端配置

来源: `scripts/serve.sh`

- **模型 checkpoint**: `/defaultShare/zhaowei_global/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_h800/checkpoints/step-024474-epoch-03-loss=0.1728.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_h800`
  - Step: 024474
  - Epoch: 03
  - Loss: 0.1728
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增（进程间启动间隔 60s）

### 客户端配置

评测脚本: `evaluate_policy_multiserver_interleave_mmins.py`

- **instruction_type**: `mmins`
- **question_file**: `/home/zw/workspace/calvin/calvin_models/calvin_agent/evaluation/question_mm_bench_v1.json`
- **instruction_path**: `/home/zw/workspace/calvin/calvin_models/conf/annotations/interleaved/object_new_playtable_validation.yaml`
- **custom_model**: 开启
- 按 chunk 并行评测，每个 chunk 对应一台服务端口

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 4.452

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 97.6% |
| 2 | 95.2% |
| 3 | 89.2% |
| 4 | 83.6% |
| 5 | 79.6% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 23 / 23 | 100.0% |
| move_slider_right | 75 / 76 | 98.7% |
| lift_red_block_slider | 34 / 34 | 100.0% |
| place_in_slider | 76 / 79 | 96.2% |
| turn_off_lightbulb | 39 / 39 | 100.0% |
| push_blue_block_left | 13 / 13 | 100.0% |
| close_drawer | 65 / 65 | 100.0% |
| push_red_block_left | 18 / 18 | 100.0% |
| lift_blue_block_table | 50 / 50 | 100.0% |
| turn_on_lightbulb | 46 / 46 | 100.0% |
| move_slider_left | 57 / 70 | 81.4% |
| rotate_red_block_left | 20 / 21 | 95.2% |
| turn_off_led | 52 / 52 | 100.0% |
| lift_pink_block_table | 32 / 32 | 100.0% |
| stack_block | 27 / 28 | 96.4% |
| open_drawer | 95 / 95 | 100.0% |
| place_in_drawer | 70 / 71 | 98.6% |
| lift_pink_block_drawer | 5 / 6 | 83.3% |
| rotate_pink_block_right | 8 / 8 | 100.0% |
| rotate_blue_block_left | 20 / 20 | 100.0% |
| lift_red_block_table | 39 / 39 | 100.0% |
| push_red_block_right | 10 / 11 | 90.9% |
| push_pink_block_left | 15 / 16 | 93.8% |
| rotate_red_block_right | 20 / 22 | 90.9% |
| lift_pink_block_slider | 41 / 42 | 97.6% |
| turn_on_led | 41 / 53 | 77.4% |
| rotate_pink_block_left | 18 / 18 | 100.0% |
| push_into_drawer | 28 / 29 | 96.6% |
| lift_blue_block_drawer | 9 / 9 | 100.0% |
| push_blue_block_right | 8 / 13 | 61.5% |
| lift_blue_block_slider | 34 / 36 | 94.4% |
| unstack_block | 11 / 11 | 100.0% |
| push_pink_block_right | 10 / 15 | 66.7% |
| lift_red_block_drawer | 4 / 4 | 100.0% |

## 实验记录 - 2026-09-09 (2)

### 服务端配置

来源: `scripts/serve.sh`

- **模型 checkpoint**: `/defaultShare/zhaowei_global/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_h800/checkpoints/step-024474-epoch-03-loss=0.1728.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_h800`
  - Step: 024474
  - Epoch: 03
  - Loss: 0.1728
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增（进程间启动间隔 60s）

### 客户端配置

评测脚本: `evaluate_policy_multiserver_interleave_mmins_long.py`

- **instruction_type**: `goal_image_v2`
- **question_file**: `/home/zw/workspace/calvin/calvin_models/calvin_agent/evaluation/question_mm_bench_v2_merged_new.json`
- **instruction_path**: `/home/zw/workspace/calvin/calvin_models/conf/annotations/interleaved/goal_new_playtable_validation.yaml`
- **env_sample_strategy**: `$ENV`（环境 ABC 随机）
- **view_strategy**: `$VIEW`（视角正常）
- **custom_model**: 开启
- 按 chunk 并行评测，每个 chunk 对应一台服务端口
- **Prompt 采样条件**: 视角正常，环境 ABC 随机

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 4.044

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 94.8% |
| 2 | 85.6% |
| 3 | 80.0% |
| 4 | 74.4% |
| 5 | 69.6% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 22 / 22 | 100.0% |
| move_slider_right | 68 / 71 | 95.8% |
| lift_red_block_slider | 34 / 36 | 94.4% |
| place_in_slider | 59 / 77 | 76.6% |
| turn_off_lightbulb | 30 / 33 | 90.9% |
| push_blue_block_left | 11 / 12 | 91.7% |
| close_drawer | 62 / 62 | 100.0% |
| push_red_block_left | 18 / 18 | 100.0% |
| lift_blue_block_table | 47 / 48 | 97.9% |
| turn_on_lightbulb | 40 / 41 | 97.6% |
| stack_block | 18 / 23 | 78.3% |
| move_slider_left | 46 / 63 | 73.0% |
| turn_off_led | 49 / 49 | 100.0% |
| open_drawer | 84 / 84 | 100.0% |
| lift_pink_block_table | 26 / 27 | 96.3% |
| place_in_drawer | 67 / 68 | 98.5% |
| lift_pink_block_drawer | 6 / 6 | 100.0% |
| rotate_pink_block_right | 7 / 8 | 87.5% |
| rotate_blue_block_left | 17 / 18 | 94.4% |
| lift_red_block_table | 37 / 37 | 100.0% |
| push_pink_block_left | 14 / 16 | 87.5% |
| rotate_red_block_right | 19 / 19 | 100.0% |
| lift_pink_block_slider | 39 / 41 | 95.1% |
| turn_on_led | 51 / 51 | 100.0% |
| rotate_pink_block_left | 15 / 16 | 93.8% |
| push_pink_block_right | 13 / 16 | 81.2% |
| push_into_drawer | 28 / 28 | 100.0% |
| lift_blue_block_slider | 35 / 38 | 92.1% |
| rotate_red_block_left | 17 / 19 | 89.5% |
| lift_red_block_drawer | 5 / 5 | 100.0% |
| unstack_block | 7 / 7 | 100.0% |
| push_blue_block_right | 7 / 11 | 63.6% |
| lift_blue_block_drawer | 6 / 6 | 100.0% |
| push_red_block_right | 7 / 11 | 63.6% |

## 实验记录 - 2026-09-09 (3)

### 服务端配置

来源: `scripts/serve.sh`

- **模型 checkpoint**: `/defaultShare/zhaowei_global/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_h800/checkpoints/step-024474-epoch-03-loss=0.1728.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_h800`
  - Step: 024474
  - Epoch: 03
  - Loss: 0.1728
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增（进程间启动间隔 60s）

### 客户端配置

评测脚本: `evaluate_policy_multiserver_interleave_mmins_long.py`

- **instruction_type**: `imitation_video_v2`
- **question_file**: `/home/zw/workspace/calvin/calvin_models/calvin_agent/evaluation/question_mm_bench_v2_merged_new.json`
- **instruction_path**: `/home/zw/workspace/calvin/calvin_models/conf/annotations/interleaved/video_new_playtable_validation.yaml`
- **env_sample_strategy**: `$ENV`（环境 ABC 随机）
- **view_strategy**: `$VIEW`（视角正常）
- **custom_model**: 开启
- 按 chunk 并行评测，每个 chunk 对应一台服务端口
- **Prompt 采样条件**: 视角正常，环境 ABC 随机

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 4.032

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 97.6% |
| 2 | 89.2% |
| 3 | 79.6% |
| 4 | 71.6% |
| 5 | 65.2% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 20 / 21 | 95.2% |
| move_slider_right | 71 / 71 | 100.0% |
| lift_red_block_slider | 32 / 36 | 88.9% |
| place_in_slider | 51 / 73 | 69.9% |
| turn_off_lightbulb | 30 / 32 | 93.8% |
| push_blue_block_left | 12 / 12 | 100.0% |
| close_drawer | 58 / 58 | 100.0% |
| push_red_block_left | 18 / 18 | 100.0% |
| lift_blue_block_table | 49 / 49 | 100.0% |
| move_slider_left | 50 / 65 | 76.9% |
| stack_block | 19 / 26 | 73.1% |
| turn_off_led | 51 / 51 | 100.0% |
| open_drawer | 86 / 86 | 100.0% |
| lift_pink_block_table | 24 / 26 | 92.3% |
| place_in_drawer | 68 / 69 | 98.6% |
| lift_pink_block_drawer | 6 / 6 | 100.0% |
| rotate_pink_block_right | 7 / 8 | 87.5% |
| rotate_blue_block_left | 18 / 19 | 94.7% |
| push_pink_block_left | 15 / 16 | 93.8% |
| rotate_red_block_right | 18 / 19 | 94.7% |
| lift_pink_block_slider | 35 / 42 | 83.3% |
| turn_on_led | 51 / 52 | 98.1% |
| rotate_pink_block_left | 16 / 18 | 88.9% |
| push_pink_block_right | 13 / 17 | 76.5% |
| push_into_drawer | 28 / 28 | 100.0% |
| lift_blue_block_drawer | 8 / 8 | 100.0% |
| turn_on_lightbulb | 39 / 40 | 97.5% |
| lift_blue_block_slider | 34 / 38 | 89.5% |
| rotate_red_block_left | 17 / 19 | 89.5% |
| lift_red_block_table | 34 / 35 | 97.1% |
| lift_red_block_drawer | 6 / 6 | 100.0% |
| unstack_block | 7 / 8 | 87.5% |
| push_blue_block_right | 7 / 12 | 58.3% |
| push_red_block_right | 10 / 11 | 90.9% |

## 实验记录 - 2026-09-09 (4)

### 服务端配置

来源: `scripts/serve.sh`

- **模型 checkpoint**: `/defaultShare/zhaowei_global/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_h800/checkpoints/step-024474-epoch-03-loss=0.1728.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_h800`
  - Step: 024474
  - Epoch: 03
  - Loss: 0.1728
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增（进程间启动间隔 60s）

### 客户端配置

评测脚本: `evaluate_policy_multiserver_interleave_mmins_long.py`

- **instruction_type**: `goal_image_v2`
- **question_file**: `/home/zw/workspace/calvin/calvin_models/calvin_agent/evaluation/question_mm_bench_v2_merged_new.json`
- **instruction_path**: `/home/zw/workspace/calvin/calvin_models/conf/annotations/interleaved/goal_new_playtable_validation.yaml`
- **env_sample_strategy**: `D`（环境固定为 D）
- **view_strategy**: `random`（视角随机）
- **custom_model**: 开启
- 按 chunk 并行评测，每个 chunk 对应一台服务端口
- **Prompt 采样条件**: 环境固定为 D，视角 random

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 0.18

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 16.4% |
| 2 | 1.6% |
| 3 | 0.0% |
| 4 | 0.0% |
| 5 | 0.0% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_red_block_right | 4 / 9 | 44.4% |
| move_slider_right | 1 / 37 | 2.7% |
| turn_on_lightbulb | 7 / 9 | 77.8% |
| push_red_block_left | 1 / 9 | 11.1% |
| close_drawer | 7 / 12 | 58.3% |
| push_pink_block_right | 4 / 6 | 66.7% |
| push_blue_block_right | 2 / 7 | 28.6% |
| open_drawer | 7 / 27 | 25.9% |
| turn_on_led | 1 / 17 | 5.9% |
| push_pink_block_left | 1 / 11 | 9.1% |
| push_into_drawer | 2 / 7 | 28.6% |
| push_red_block_right | 2 / 7 | 28.6% |
| turn_off_led | 1 / 19 | 5.3% |
| lift_pink_block_table | 3 / 11 | 27.3% |
| rotate_blue_block_right | 2 / 12 | 16.7% |
| push_blue_block_left | 0 / 5 | 0.0% |
| move_slider_left | 0 / 21 | 0.0% |
| lift_red_block_slider | 0 / 12 | 0.0% |
| rotate_pink_block_right | 0 / 3 | 0.0% |
| turn_off_lightbulb | 0 / 8 | 0.0% |
| lift_blue_block_table | 0 / 5 | 0.0% |
| lift_pink_block_slider | 0 / 9 | 0.0% |
| rotate_blue_block_left | 0 / 3 | 0.0% |
| lift_blue_block_slider | 0 / 9 | 0.0% |
| rotate_red_block_left | 0 / 7 | 0.0% |
| rotate_pink_block_left | 0 / 8 | 0.0% |
| lift_red_block_table | 0 / 2 | 0.0% |
| place_in_drawer | 0 / 3 | 0.0% |

## 实验记录 - 2026-09-09 (5)

### 服务端配置

来源: `scripts/serve.sh`

- **模型 checkpoint**: `/defaultShare/zhaowei_global/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_h800/checkpoints/step-024474-epoch-03-loss=0.1728.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_h800`
  - Step: 024474
  - Epoch: 03
  - Loss: 0.1728
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增（进程间启动间隔 60s）

### 客户端配置

评测脚本: `evaluate_policy_multiserver_interleave_mmins_long.py`

- **instruction_type**: `imitation_video_v2`
- **question_file**: `/home/zw/workspace/calvin/calvin_models/calvin_agent/evaluation/question_mm_bench_v2_merged_new.json`
- **instruction_path**: `/home/zw/workspace/calvin/calvin_models/conf/annotations/interleaved/video_new_playtable_validation.yaml`
- **env_sample_strategy**: `D`（环境固定为 D）
- **view_strategy**: `random`（视角随机）
- **custom_model**: 开启
- 按 chunk 并行评测，每个 chunk 对应一台服务端口
- **Prompt 采样条件**: 环境固定为 D，视角 random

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 0.284

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 23.6% |
| 2 | 4.4% |
| 3 | 0.4% |
| 4 | 0.0% |
| 5 | 0.0% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| move_slider_left | 4 / 19 | 21.1% |
| rotate_red_block_right | 4 / 12 | 33.3% |
| turn_on_lightbulb | 10 / 12 | 83.3% |
| close_drawer | 9 / 13 | 69.2% |
| turn_on_led | 6 / 16 | 37.5% |
| rotate_blue_block_right | 4 / 13 | 30.8% |
| move_slider_right | 13 / 33 | 39.4% |
| push_pink_block_right | 3 / 6 | 50.0% |
| push_blue_block_right | 3 / 7 | 42.9% |
| turn_off_lightbulb | 2 / 9 | 22.2% |
| open_drawer | 5 / 31 | 16.1% |
| push_into_drawer | 3 / 8 | 37.5% |
| turn_off_led | 1 / 21 | 4.8% |
| lift_pink_block_table | 1 / 10 | 10.0% |
| place_in_drawer | 1 / 1 | 100.0% |
| push_red_block_right | 1 / 8 | 12.5% |
| push_pink_block_left | 1 / 11 | 9.1% |
| push_blue_block_left | 0 / 5 | 0.0% |
| push_red_block_left | 0 / 12 | 0.0% |
| rotate_red_block_left | 0 / 9 | 0.0% |
| lift_red_block_slider | 0 / 15 | 0.0% |
| rotate_pink_block_right | 0 / 3 | 0.0% |
| lift_blue_block_table | 0 / 6 | 0.0% |
| lift_pink_block_slider | 0 / 13 | 0.0% |
| rotate_blue_block_left | 0 / 4 | 0.0% |
| lift_red_block_table | 0 / 6 | 0.0% |
| lift_blue_block_slider | 0 / 9 | 0.0% |
| rotate_pink_block_left | 0 / 9 | 0.0% |

## 实验记录 - 2026-09-09 (6)

### 服务端配置

来源: `scripts/serve.sh`

- **模型 checkpoint**: `/defaultShare/zhaowei_global/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_nolmloss_h800/checkpoints/step-008158-epoch-01-loss=0.1327.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_nolmloss_h800`
  - Step: 008158
  - Epoch: 01
  - Loss: 0.1327
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增（进程间启动间隔 60s）

### 客户端配置

评测脚本: `evaluate_policy_multiserver_interleave_mmins.py`

- **instruction_type**: `text`
- **question_file**: `/home/zw/workspace/calvin/calvin_models/calvin_agent/evaluation/question_mm_bench_v1.json`
- **instruction_path**: `/home/zw/workspace/calvin/calvin_models/conf/annotations/new_playtable_validation.yaml`
- **excute_length**: `$EXCUTE_LENGTH`
- **custom_model**: 开启
- 按 chunk 并行评测，每个 chunk 对应一台服务端口

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 4.344

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 97.2% |
| 2 | 91.6% |
| 3 | 86.8% |
| 4 | 81.6% |
| 5 | 77.2% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 23 / 23 | 100.0% |
| move_slider_right | 74 / 74 | 100.0% |
| lift_red_block_slider | 34 / 37 | 91.9% |
| place_in_slider | 58 / 77 | 75.3% |
| turn_off_lightbulb | 33 / 33 | 100.0% |
| push_blue_block_left | 12 / 12 | 100.0% |
| close_drawer | 61 / 61 | 100.0% |
| move_slider_left | 63 / 64 | 98.4% |
| rotate_red_block_left | 20 / 20 | 100.0% |
| turn_off_led | 54 / 54 | 100.0% |
| lift_pink_block_table | 31 / 32 | 96.9% |
| stack_block | 22 / 27 | 81.5% |
| open_drawer | 94 / 94 | 100.0% |
| place_in_drawer | 72 / 73 | 98.6% |
| lift_pink_block_drawer | 4 / 5 | 80.0% |
| rotate_pink_block_right | 7 / 8 | 87.5% |
| lift_blue_block_table | 53 / 53 | 100.0% |
| rotate_blue_block_left | 18 / 18 | 100.0% |
| lift_red_block_table | 40 / 40 | 100.0% |
| push_red_block_right | 9 / 11 | 81.8% |
| push_pink_block_left | 14 / 16 | 87.5% |
| rotate_red_block_right | 20 / 21 | 95.2% |
| lift_pink_block_slider | 40 / 41 | 97.6% |
| turn_on_led | 48 / 49 | 98.0% |
| rotate_pink_block_left | 17 / 19 | 89.5% |
| push_pink_block_right | 13 / 17 | 76.5% |
| push_into_drawer | 26 / 29 | 89.7% |
| lift_blue_block_drawer | 8 / 9 | 88.9% |
| turn_on_lightbulb | 44 / 44 | 100.0% |
| push_blue_block_right | 10 / 13 | 76.9% |
| lift_blue_block_slider | 34 / 36 | 94.4% |
| push_red_block_left | 15 / 18 | 83.3% |
| lift_red_block_drawer | 5 / 5 | 100.0% |
| unstack_block | 10 / 10 | 100.0% |

## 实验记录 - 2026-09-09 (7)

### 服务端配置

来源: `scripts/serve.sh`

- **模型 checkpoint**: `/defaultShare/zhaowei_global/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_nolmloss_h800/checkpoints/step-008158-epoch-01-loss=0.1327.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_nolmloss_h800`
  - Step: 008158
  - Epoch: 01
  - Loss: 0.1327
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增（进程间启动间隔 60s）

### 客户端配置

评测脚本: `evaluate_policy_multiserver_interleave_mmins.py`

- **instruction_type**: `mmins`
- **question_file**: `/home/zw/workspace/calvin/calvin_models/calvin_agent/evaluation/question_mm_bench_v1.json`
- **instruction_path**: `/home/zw/workspace/calvin/calvin_models/conf/annotations/interleaved/object_new_playtable_validation.yaml`
- **excute_length**: `$EXCUTE_LENGTH`
- **custom_model**: 开启
- 按 chunk 并行评测，每个 chunk 对应一台服务端口

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 2.82

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 84.8% |
| 2 | 66.4% |
| 3 | 54.0% |
| 4 | 43.6% |
| 5 | 33.2% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 17 / 17 | 100.0% |
| move_slider_right | 50 / 65 | 76.9% |
| lift_red_block_slider | 29 / 30 | 96.7% |
| place_in_slider | 47 / 68 | 69.1% |
| push_blue_block_left | 9 / 9 | 100.0% |
| close_drawer | 41 / 42 | 97.6% |
| push_red_block_left | 14 / 17 | 82.4% |
| lift_blue_block_table | 32 / 32 | 100.0% |
| turn_on_lightbulb | 20 / 35 | 57.1% |
| stack_block | 17 / 22 | 77.3% |
| rotate_pink_block_right | 6 / 7 | 85.7% |
| open_drawer | 67 / 69 | 97.1% |
| place_in_drawer | 51 / 53 | 96.2% |
| push_red_block_right | 8 / 10 | 80.0% |
| turn_off_led | 42 / 42 | 100.0% |
| push_pink_block_left | 14 / 16 | 87.5% |
| lift_pink_block_table | 22 / 22 | 100.0% |
| rotate_red_block_right | 15 / 15 | 100.0% |
| lift_pink_block_slider | 31 / 32 | 96.9% |
| turn_on_led | 24 / 42 | 57.1% |
| rotate_pink_block_left | 13 / 14 | 92.9% |
| move_slider_left | 13 / 55 | 23.6% |
| push_pink_block_right | 9 / 12 | 75.0% |
| push_into_drawer | 15 / 16 | 93.8% |
| rotate_blue_block_left | 9 / 9 | 100.0% |
| rotate_red_block_left | 13 / 13 | 100.0% |
| lift_red_block_table | 21 / 22 | 95.5% |
| lift_blue_block_slider | 28 / 30 | 93.3% |
| unstack_block | 6 / 7 | 85.7% |
| lift_red_block_drawer | 3 / 3 | 100.0% |
| push_blue_block_right | 7 / 9 | 77.8% |
| lift_blue_block_drawer | 5 / 5 | 100.0% |
| turn_off_lightbulb | 4 / 29 | 13.8% |
| lift_pink_block_drawer | 3 / 3 | 100.0% |

## 实验记录 - 2026-09-09 (8)

### 服务端配置

来源: `scripts/serve.sh`

- **模型 checkpoint**: `/defaultShare/zhaowei_global/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_nolmloss_h800/checkpoints/step-008158-epoch-01-loss=0.1327.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_nolmloss_h800`
  - Step: 008158
  - Epoch: 01
  - Loss: 0.1327
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增（进程间启动间隔 60s）

### 客户端配置

评测脚本: `evaluate_policy_multiserver_interleave_mmins.py`

- **instruction_type**: `goal_image`
- **question_file**: `/home/zw/workspace/calvin/calvin_models/calvin_agent/evaluation/question_mm_bench_v1.json`
- **instruction_path**: `/home/zw/workspace/calvin/calvin_models/conf/annotations/interleaved/goal_new_playtable_validation.yaml`
- **excute_length**: `$EXCUTE_LENGTH`
- **custom_model**: 开启
- 按 chunk 并行评测，每个 chunk 对应一台服务端口

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 4.448

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 97.6% |
| 2 | 94.0% |
| 3 | 87.6% |
| 4 | 84.8% |
| 5 | 80.8% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 23 / 23 | 100.0% |
| move_slider_right | 77 / 77 | 100.0% |
| lift_red_block_slider | 34 / 35 | 97.1% |
| place_in_slider | 75 / 76 | 98.7% |
| turn_off_lightbulb | 34 / 34 | 100.0% |
| push_blue_block_left | 14 / 14 | 100.0% |
| close_drawer | 66 / 66 | 100.0% |
| push_red_block_left | 16 / 18 | 88.9% |
| lift_blue_block_table | 55 / 57 | 96.5% |
| turn_on_lightbulb | 45 / 46 | 97.8% |
| move_slider_left | 66 / 66 | 100.0% |
| rotate_red_block_left | 21 / 22 | 95.5% |
| turn_off_led | 54 / 54 | 100.0% |
| lift_pink_block_table | 27 / 30 | 90.0% |
| stack_block | 18 / 27 | 66.7% |
| open_drawer | 92 / 94 | 97.9% |
| place_in_drawer | 67 / 70 | 95.7% |
| lift_pink_block_drawer | 7 / 7 | 100.0% |
| rotate_pink_block_right | 6 / 8 | 75.0% |
| rotate_blue_block_left | 18 / 19 | 94.7% |
| lift_red_block_table | 39 / 39 | 100.0% |
| push_red_block_right | 9 / 11 | 81.8% |
| push_pink_block_left | 14 / 15 | 93.3% |
| rotate_red_block_right | 20 / 22 | 90.9% |
| lift_pink_block_slider | 38 / 40 | 95.0% |
| turn_on_led | 49 / 49 | 100.0% |
| rotate_pink_block_left | 19 / 20 | 95.0% |
| push_pink_block_right | 16 / 17 | 94.1% |
| push_into_drawer | 30 / 30 | 100.0% |
| lift_blue_block_drawer | 8 / 9 | 88.9% |
| push_blue_block_right | 10 / 13 | 76.9% |
| lift_blue_block_slider | 31 / 38 | 81.6% |
| lift_red_block_drawer | 5 / 5 | 100.0% |
| unstack_block | 9 / 9 | 100.0% |

## 实验记录 - 2026-09-09 (9)

### 服务端配置

来源: `scripts/serve.sh`

- **模型 checkpoint**: `/defaultShare/zhaowei_global/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_nolmloss_h800/checkpoints/step-008158-epoch-01-loss=0.1327.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_nolmloss_h800`
  - Step: 008158
  - Epoch: 01
  - Loss: 0.1327
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增（进程间启动间隔 60s）

### 客户端配置

评测脚本: `evaluate_policy_multiserver_interleave_mmins.py`

- **instruction_type**: `imitation_video`
- **question_file**: `/home/zw/workspace/calvin/calvin_models/calvin_agent/evaluation/question_mm_bench_v1.json`
- **instruction_path**: `/home/zw/workspace/calvin/calvin_models/conf/annotations/interleaved/video_new_playtable_validation.yaml`
- **excute_length**: `$EXCUTE_LENGTH`
- **custom_model**: 开启
- 按 chunk 并行评测，每个 chunk 对应一台服务端口

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 4.62

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 98.4% |
| 2 | 94.4% |
| 3 | 91.2% |
| 4 | 89.6% |
| 5 | 88.4% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 23 / 23 | 100.0% |
| move_slider_right | 78 / 78 | 100.0% |
| lift_red_block_slider | 37 / 37 | 100.0% |
| place_in_slider | 76 / 77 | 98.7% |
| turn_off_lightbulb | 37 / 37 | 100.0% |
| push_blue_block_left | 14 / 14 | 100.0% |
| close_drawer | 68 / 68 | 100.0% |
| push_red_block_left | 16 / 17 | 94.1% |
| lift_blue_block_table | 55 / 57 | 96.5% |
| turn_on_lightbulb | 48 / 48 | 100.0% |
| move_slider_left | 70 / 70 | 100.0% |
| rotate_red_block_left | 21 / 21 | 100.0% |
| turn_off_led | 54 / 54 | 100.0% |
| lift_pink_block_table | 26 / 29 | 89.7% |
| stack_block | 21 / 24 | 87.5% |
| open_drawer | 95 / 96 | 99.0% |
| place_in_drawer | 72 / 72 | 100.0% |
| lift_pink_block_drawer | 6 / 6 | 100.0% |
| rotate_pink_block_right | 7 / 8 | 87.5% |
| rotate_blue_block_left | 20 / 20 | 100.0% |
| lift_red_block_table | 41 / 41 | 100.0% |
| push_red_block_right | 9 / 11 | 81.8% |
| push_pink_block_left | 16 / 16 | 100.0% |
| rotate_red_block_right | 20 / 21 | 95.2% |
| lift_pink_block_slider | 38 / 42 | 90.5% |
| turn_on_led | 54 / 54 | 100.0% |
| rotate_pink_block_left | 19 / 20 | 95.0% |
| push_pink_block_right | 16 / 17 | 94.1% |
| push_into_drawer | 29 / 30 | 96.7% |
| lift_blue_block_drawer | 9 / 9 | 100.0% |
| push_blue_block_right | 12 / 13 | 92.3% |
| lift_blue_block_slider | 32 / 38 | 84.2% |
| lift_red_block_drawer | 6 / 6 | 100.0% |
| unstack_block | 10 / 10 | 100.0% |

## 实验记录 - 2026-09-09 (10)

### 服务端配置

来源: `scripts/serve.sh`

- **模型 checkpoint**: `/defaultShare/zhaowei_global/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_nolmloss_h800/checkpoints/step-008158-epoch-01-loss=0.1327.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_nolmloss_h800`
  - Step: 008158
  - Epoch: 01
  - Loss: 0.1327
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增（进程间启动间隔 60s）

### 客户端配置

评测脚本: `evaluate_policy_multiserver_interleave_mmins_long.py`

- **instruction_type**: `goal_image_v2`
- **question_file**: `/home/zw/workspace/calvin/calvin_models/calvin_agent/evaluation/question_mm_bench_v2_merged_new.json`
- **instruction_path**: `/home/zw/workspace/calvin/calvin_models/conf/annotations/interleaved/goal_new_playtable_validation.yaml`
- **env_sample_strategy**: `$ENV`（环境固定为 D）
- **view_strategy**: `$VIEW`（视角随机）
- **excute_length**: `$EXCUTE_LENGTH`
- **custom_model**: 开启
- 按 chunk 并行评测，每个 chunk 对应一台服务端口
- **Prompt 采样条件**: 环境固定为 D，视角 random

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 0.336

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 25.6% |
| 2 | 6.4% |
| 3 | 1.2% |
| 4 | 0.4% |
| 5 | 0.0% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| push_blue_block_left | 2 / 5 | 40.0% |
| close_drawer | 9 / 16 | 56.2% |
| lift_blue_block_table | 2 / 8 | 25.0% |
| move_slider_right | 2 / 38 | 5.3% |
| turn_on_lightbulb | 9 / 15 | 60.0% |
| open_drawer | 21 / 26 | 80.8% |
| push_into_drawer | 4 / 6 | 66.7% |
| push_pink_block_right | 3 / 6 | 50.0% |
| push_red_block_left | 2 / 10 | 20.0% |
| push_blue_block_right | 4 / 8 | 50.0% |
| push_red_block_right | 6 / 8 | 75.0% |
| push_pink_block_left | 3 / 10 | 30.0% |
| lift_pink_block_table | 8 / 13 | 61.5% |
| rotate_pink_block_right | 1 / 4 | 25.0% |
| place_in_slider | 2 / 6 | 33.3% |
| move_slider_left | 1 / 22 | 4.5% |
| turn_off_led | 1 / 19 | 5.3% |
| place_in_drawer | 2 / 5 | 40.0% |
| lift_red_block_slider | 2 / 15 | 13.3% |
| rotate_blue_block_right | 0 / 12 | 0.0% |
| turn_off_lightbulb | 0 / 9 | 0.0% |
| rotate_red_block_right | 0 / 9 | 0.0% |
| turn_on_led | 0 / 19 | 0.0% |
| lift_pink_block_slider | 0 / 10 | 0.0% |
| rotate_blue_block_left | 0 / 5 | 0.0% |
| lift_blue_block_slider | 0 / 12 | 0.0% |
| rotate_red_block_left | 0 / 8 | 0.0% |
| rotate_pink_block_left | 0 / 8 | 0.0% |
| lift_red_block_table | 0 / 1 | 0.0% |
| stack_block | 0 / 1 | 0.0% |

## 实验记录 - 2026-09-09 (11)

### 服务端配置

来源: `scripts/serve.sh`

- **模型 checkpoint**: `/defaultShare/zhaowei_global/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_nolmloss_h800/checkpoints/step-008158-epoch-01-loss=0.1327.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_nolmloss_h800`
  - Step: 008158
  - Epoch: 01
  - Loss: 0.1327
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增（进程间启动间隔 60s）

### 客户端配置

评测脚本: `evaluate_policy_multiserver_interleave_mmins_long.py`

- **instruction_type**: `imitation_video_v2`
- **question_file**: `/home/zw/workspace/calvin/calvin_models/calvin_agent/evaluation/question_mm_bench_v2_merged_new.json`
- **instruction_path**: `/home/zw/workspace/calvin/calvin_models/conf/annotations/interleaved/video_new_playtable_validation.yaml`
- **env_sample_strategy**: `$ENV`（环境固定为 D）
- **view_strategy**: `$VIEW`（视角随机）
- **excute_length**: `$EXCUTE_LENGTH`
- **custom_model**: 开启
- 按 chunk 并行评测，每个 chunk 对应一台服务端口
- **Prompt 采样条件**: 环境固定为 D，视角 random

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 0.368

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 28.0% |
| 2 | 6.8% |
| 3 | 1.2% |
| 4 | 0.4% |
| 5 | 0.4% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_pink_block_right | 3 / 5 | 60.0% |
| open_drawer | 19 / 29 | 65.5% |
| move_slider_right | 2 / 39 | 5.1% |
| push_into_drawer | 5 / 7 | 71.4% |
| turn_on_lightbulb | 10 / 13 | 76.9% |
| push_red_block_left | 6 / 10 | 60.0% |
| push_pink_block_left | 6 / 10 | 60.0% |
| close_drawer | 9 / 14 | 64.3% |
| push_pink_block_right | 3 / 6 | 50.0% |
| push_blue_block_right | 3 / 7 | 42.9% |
| push_red_block_right | 5 / 8 | 62.5% |
| push_blue_block_left | 2 / 5 | 40.0% |
| lift_pink_block_table | 8 / 13 | 61.5% |
| move_slider_left | 3 / 22 | 13.6% |
| place_in_slider | 2 / 5 | 40.0% |
| turn_off_led | 1 / 20 | 5.0% |
| place_in_drawer | 2 / 5 | 40.0% |
| lift_red_block_slider | 1 / 15 | 6.7% |
| lift_blue_block_slider | 1 / 13 | 7.7% |
| lift_blue_block_table | 1 / 9 | 11.1% |
| rotate_blue_block_right | 0 / 12 | 0.0% |
| turn_off_lightbulb | 0 / 9 | 0.0% |
| rotate_red_block_right | 0 / 9 | 0.0% |
| turn_on_led | 0 / 18 | 0.0% |
| lift_pink_block_slider | 0 / 13 | 0.0% |
| rotate_blue_block_left | 0 / 4 | 0.0% |
| rotate_red_block_left | 0 / 9 | 0.0% |
| rotate_pink_block_left | 0 / 8 | 0.0% |
| lift_red_block_table | 0 / 3 | 0.0% |
| stack_block | 0 / 1 | 0.0% |

## 实验记录 - 2026-09-09 (12)

### 服务端配置

来源: `scripts/serve.sh`

- **模型 checkpoint**: `/defaultShare/zhaowei_global/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_nolmloss_h800/checkpoints/step-008158-epoch-01-loss=0.1327.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_nolmloss_h800`
  - Step: 008158
  - Epoch: 01
  - Loss: 0.1327
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增（进程间启动间隔 60s）

### 客户端配置

评测脚本: `evaluate_policy_multiserver_interleave_mmins_long.py`

- **instruction_type**: `goal_image_v2`
- **question_file**: `/home/zw/workspace/calvin/calvin_models/calvin_agent/evaluation/question_mm_bench_v2_merged_new.json`
- **instruction_path**: `/home/zw/workspace/calvin/calvin_models/conf/annotations/interleaved/goal_new_playtable_validation.yaml`
- **env_sample_strategy**: `random`（环境 ABC 随机）
- **view_strategy**: `normal`（视角正常）
- **excute_length**: `$EXCUTE_LENGTH`
- **custom_model**: 开启
- 按 chunk 并行评测，每个 chunk 对应一台服务端口
- **Prompt 采样条件**: 环境 ABC 随机，视角正常

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.668

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 92.4% |
| 2 | 79.6% |
| 3 | 70.8% |
| 4 | 65.2% |
| 5 | 58.8% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 21 / 21 | 100.0% |
| move_slider_right | 66 / 67 | 98.5% |
| lift_red_block_slider | 35 / 37 | 94.6% |
| place_in_slider | 54 / 67 | 80.6% |
| push_red_block_left | 16 / 17 | 94.1% |
| lift_blue_block_table | 38 / 40 | 95.0% |
| stack_block | 15 / 22 | 68.2% |
| move_slider_left | 36 / 57 | 63.2% |
| turn_off_led | 49 / 50 | 98.0% |
| open_drawer | 83 / 84 | 98.8% |
| place_in_drawer | 64 / 67 | 95.5% |
| push_red_block_right | 8 / 9 | 88.9% |
| close_drawer | 56 / 56 | 100.0% |
| push_pink_block_left | 14 / 15 | 93.3% |
| lift_pink_block_table | 22 / 24 | 91.7% |
| rotate_red_block_right | 18 / 19 | 94.7% |
| lift_pink_block_slider | 36 / 37 | 97.3% |
| turn_on_led | 52 / 54 | 96.3% |
| rotate_pink_block_left | 16 / 17 | 94.1% |
| push_pink_block_right | 12 / 15 | 80.0% |
| push_into_drawer | 23 / 25 | 92.0% |
| lift_blue_block_drawer | 6 / 6 | 100.0% |
| turn_off_lightbulb | 16 / 27 | 59.3% |
| turn_on_lightbulb | 31 / 35 | 88.6% |
| lift_blue_block_slider | 30 / 36 | 83.3% |
| lift_pink_block_drawer | 5 / 5 | 100.0% |
| push_blue_block_left | 7 / 10 | 70.0% |
| rotate_blue_block_left | 16 / 16 | 100.0% |
| rotate_red_block_left | 18 / 20 | 90.0% |
| lift_red_block_table | 32 / 34 | 94.1% |
| lift_red_block_drawer | 6 / 6 | 100.0% |
| push_blue_block_right | 6 / 12 | 50.0% |
| rotate_pink_block_right | 5 / 8 | 62.5% |
| unstack_block | 5 / 5 | 100.0% |

## 实验记录 - 2026-09-09 (13)

### 服务端配置

来源: `scripts/serve.sh`

- **模型 checkpoint**: `/defaultShare/zhaowei_global/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_nolmloss_h800/checkpoints/step-008158-epoch-01-loss=0.1327.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft_nolmloss_h800`
  - Step: 008158
  - Epoch: 01
  - Loss: 0.1327
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增（进程间启动间隔 60s）

### 客户端配置

评测脚本: `evaluate_policy_multiserver_interleave_mmins_long.py`

- **instruction_type**: `imitation_video_v2`
- **question_file**: `/home/zw/workspace/calvin/calvin_models/calvin_agent/evaluation/question_mm_bench_v2_merged_new.json`
- **instruction_path**: `/home/zw/workspace/calvin/calvin_models/conf/annotations/interleaved/video_new_playtable_validation.yaml`
- **env_sample_strategy**: `random`（环境 ABC 随机）
- **view_strategy**: `normal`（视角正常）
- **excute_length**: `$EXCUTE_LENGTH`
- **custom_model**: 开启
- 按 chunk 并行评测，每个 chunk 对应一台服务端口
- **Prompt 采样条件**: 环境 ABC 随机，视角正常

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.94

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 94.0% |
| 2 | 85.2% |
| 3 | 78.8% |
| 4 | 72.4% |
| 5 | 63.6% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 22 / 23 | 95.7% |
| move_slider_right | 66 / 66 | 100.0% |
| lift_red_block_slider | 35 / 36 | 97.2% |
| place_in_slider | 62 / 76 | 81.6% |
| push_blue_block_left | 10 / 10 | 100.0% |
| close_drawer | 60 / 60 | 100.0% |
| push_red_block_left | 18 / 18 | 100.0% |
| lift_blue_block_table | 39 / 41 | 95.1% |
| turn_on_lightbulb | 30 / 40 | 75.0% |
| move_slider_left | 49 / 59 | 83.1% |
| rotate_red_block_left | 16 / 20 | 80.0% |
| turn_off_led | 53 / 54 | 98.1% |
| lift_pink_block_table | 26 / 26 | 100.0% |
| stack_block | 21 / 26 | 80.8% |
| open_drawer | 87 / 88 | 98.9% |
| rotate_pink_block_right | 6 / 8 | 75.0% |
| place_in_drawer | 63 / 64 | 98.4% |
| push_red_block_right | 9 / 10 | 90.0% |
| push_pink_block_left | 15 / 15 | 100.0% |
| rotate_red_block_right | 20 / 20 | 100.0% |
| lift_pink_block_slider | 38 / 39 | 97.4% |
| turn_on_led | 55 / 56 | 98.2% |
| rotate_pink_block_left | 17 / 18 | 94.4% |
| push_pink_block_right | 12 / 16 | 75.0% |
| push_into_drawer | 23 / 26 | 88.5% |
| lift_blue_block_drawer | 7 / 7 | 100.0% |
| lift_blue_block_slider | 34 / 36 | 94.4% |
| turn_off_lightbulb | 16 / 35 | 45.7% |
| lift_pink_block_drawer | 4 / 5 | 80.0% |
| rotate_blue_block_left | 17 / 17 | 100.0% |
| lift_red_block_table | 35 / 36 | 97.2% |
| lift_red_block_drawer | 6 / 6 | 100.0% |
| unstack_block | 8 / 9 | 88.9% |
| push_blue_block_right | 6 / 10 | 60.0% |
