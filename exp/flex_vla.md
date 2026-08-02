# Flex VLA 实验记录

## 实验记录 - 2026-07-25

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_3dots_stageB/checkpoints/step-025089-epoch-03-loss=0.1957.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_3dots_stageB`
  - Step: 025089
  - Epoch: 03
  - Loss: 0.1957
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

日志目录: `calvin_models/calvin_agent/evaluation/log_calvin_abc2d_oe`

**Results for Epoch -1:**

Average successful sequence length: 3.731

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 90.3% |
| 2 | 79.9% |
| 3 | 73.4% |
| 4 | 68.5% |
| 5 | 61.0% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 68 / 73 | 93.2% |
| move_slider_right | 252 / 252 | 100.0% |
| lift_red_block_slider | 116 / 124 | 93.5% |
| place_in_slider | 293 / 315 | 93.0% |
| turn_off_lightbulb | 139 / 141 | 98.6% |
| turn_off_led | 150 / 151 | 99.3% |
| push_into_drawer | 87 / 103 | 84.5% |
| lift_blue_block_drawer | 16 / 18 | 88.9% |
| close_drawer | 194 / 194 | 100.0% |
| lift_pink_block_slider | 113 / 120 | 94.2% |
| open_drawer | 316 / 323 | 97.8% |
| rotate_red_block_right | 56 / 73 | 76.7% |
| lift_red_block_table | 143 / 150 | 95.3% |
| lift_pink_block_table | 143 / 149 | 96.0% |
| move_slider_left | 243 / 243 | 100.0% |
| turn_on_lightbulb | 165 / 165 | 100.0% |
| rotate_blue_block_left | 67 / 68 | 98.5% |
| push_blue_block_left | 57 / 69 | 82.6% |
| turn_on_led | 164 / 170 | 96.5% |
| push_red_block_left | 50 / 76 | 65.8% |
| lift_blue_block_table | 150 / 152 | 98.7% |
| place_in_drawer | 160 / 161 | 99.4% |
| rotate_red_block_left | 58 / 62 | 93.5% |
| stack_block | 134 / 164 | 81.7% |
| lift_pink_block_drawer | 10 / 11 | 90.9% |
| unstack_block | 56 / 56 | 100.0% |
| lift_blue_block_slider | 101 / 111 | 91.0% |
| push_red_block_right | 20 / 71 | 28.2% |
| rotate_pink_block_left | 49 / 56 | 87.5% |
| push_pink_block_left | 53 / 76 | 69.7% |
| rotate_pink_block_right | 48 / 71 | 67.6% |
| push_pink_block_right | 28 / 67 | 41.8% |
| lift_red_block_drawer | 15 / 15 | 100.0% |
| push_blue_block_right | 17 / 71 | 23.9% |

## 实验记录 - 2026-07-25 (2)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_3dots_stageB/checkpoints/step-016726-epoch-02-loss=0.2000.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_3dots_stageB`
  - Step: 016726
  - Epoch: 02
  - Loss: 0.2000
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.715

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 90.1% |
| 2 | 80.7% |
| 3 | 74.0% |
| 4 | 67.0% |
| 5 | 59.7% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 71 / 77 | 92.2% |
| move_slider_right | 251 / 251 | 100.0% |
| lift_red_block_slider | 116 / 125 | 92.8% |
| place_in_slider | 274 / 322 | 85.1% |
| turn_off_lightbulb | 131 / 131 | 100.0% |
| turn_off_led | 156 / 157 | 99.4% |
| push_into_drawer | 89 / 105 | 84.8% |
| lift_blue_block_drawer | 17 / 18 | 94.4% |
| close_drawer | 187 / 187 | 100.0% |
| lift_pink_block_slider | 120 / 125 | 96.0% |
| open_drawer | 318 / 318 | 100.0% |
| rotate_red_block_right | 67 / 75 | 89.3% |
| lift_red_block_table | 153 / 158 | 96.8% |
| lift_pink_block_table | 138 / 142 | 97.2% |
| turn_on_lightbulb | 162 / 164 | 98.8% |
| push_blue_block_left | 54 / 70 | 77.1% |
| turn_on_led | 167 / 170 | 98.2% |
| stack_block | 138 / 165 | 83.6% |
| push_red_block_left | 51 / 78 | 65.4% |
| lift_blue_block_table | 153 / 154 | 99.4% |
| rotate_blue_block_left | 64 / 66 | 97.0% |
| place_in_drawer | 154 / 154 | 100.0% |
| move_slider_left | 236 / 237 | 99.6% |
| rotate_red_block_left | 63 / 65 | 96.9% |
| push_pink_block_left | 44 / 75 | 58.7% |
| lift_pink_block_drawer | 11 / 13 | 84.6% |
| rotate_pink_block_right | 44 / 70 | 62.9% |
| unstack_block | 61 / 62 | 98.4% |
| lift_blue_block_slider | 100 / 108 | 92.6% |
| push_red_block_right | 21 / 70 | 30.0% |
| rotate_pink_block_left | 50 / 55 | 90.9% |
| push_pink_block_right | 22 / 66 | 33.3% |
| lift_red_block_drawer | 14 / 14 | 100.0% |
| push_blue_block_right | 18 / 71 | 25.4% |

## 实验记录 - 2026-07-26

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_3dots_stageB_fromA/checkpoints/step-025089-epoch-03-loss=0.2410.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_3dots_stageB_fromA`
  - Step: 025089
  - Epoch: 03
  - Loss: 0.2410
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

日志目录: `calvin_models/calvin_agent/evaluation/log_calvin_abc2d_oe`

**Results for Epoch -1:**

Average successful sequence length: 3.697

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 91.1% |
| 2 | 82.6% |
| 3 | 73.2% |
| 4 | 66.1% |
| 5 | 56.7% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 71 / 77 | 92.2% |
| move_slider_right | 260 / 260 | 100.0% |
| lift_red_block_slider | 123 / 124 | 99.2% |
| place_in_slider | 215 / 338 | 63.6% |
| turn_off_lightbulb | 132 / 132 | 100.0% |
| turn_off_led | 154 / 155 | 99.4% |
| push_into_drawer | 85 / 98 | 86.7% |
| lift_blue_block_drawer | 18 / 18 | 100.0% |
| close_drawer | 177 / 177 | 100.0% |
| lift_pink_block_slider | 117 / 125 | 93.6% |
| open_drawer | 335 / 336 | 99.7% |
| rotate_red_block_right | 59 / 70 | 84.3% |
| lift_red_block_table | 151 / 154 | 98.1% |
| lift_pink_block_table | 149 / 160 | 93.1% |
| move_slider_left | 233 / 233 | 100.0% |
| turn_on_lightbulb | 161 / 161 | 100.0% |
| rotate_blue_block_left | 66 / 68 | 97.1% |
| push_blue_block_left | 59 / 69 | 85.5% |
| turn_on_led | 146 / 153 | 95.4% |
| push_red_block_left | 56 / 72 | 77.8% |
| lift_blue_block_table | 158 / 159 | 99.4% |
| place_in_drawer | 159 / 162 | 98.1% |
| rotate_red_block_left | 57 / 61 | 93.4% |
| stack_block | 143 / 170 | 84.1% |
| push_pink_block_left | 56 / 75 | 74.7% |
| lift_pink_block_drawer | 10 / 11 | 90.9% |
| rotate_pink_block_right | 63 / 69 | 91.3% |
| unstack_block | 55 / 55 | 100.0% |
| lift_blue_block_slider | 106 / 114 | 93.0% |
| push_red_block_right | 21 / 71 | 29.6% |
| rotate_pink_block_left | 49 / 56 | 87.5% |
| push_pink_block_right | 25 / 64 | 39.1% |
| lift_red_block_drawer | 14 / 14 | 100.0% |
| push_blue_block_right | 14 / 69 | 20.3% |

## 实验记录 - 2026-07-26 (2)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_3dots_stageB_fromA/checkpoints/step-016726-epoch-02-loss=0.2580.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_3dots_stageB_fromA`
  - Step: 016726
  - Epoch: 02
  - Loss: 0.2580
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.702

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 89.7% |
| 2 | 80.8% |
| 3 | 73.5% |
| 4 | 66.9% |
| 5 | 59.3% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 72 / 76 | 94.7% |
| move_slider_right | 255 / 255 | 100.0% |
| lift_red_block_slider | 125 / 127 | 98.4% |
| place_in_slider | 262 / 329 | 79.6% |
| turn_off_lightbulb | 134 / 135 | 99.3% |
| turn_off_led | 155 / 156 | 99.4% |
| push_into_drawer | 90 / 106 | 84.9% |
| lift_blue_block_drawer | 19 / 19 | 100.0% |
| lift_pink_block_slider | 124 / 128 | 96.9% |
| open_drawer | 323 / 325 | 99.4% |
| rotate_red_block_right | 54 / 72 | 75.0% |
| lift_red_block_table | 147 / 150 | 98.0% |
| lift_pink_block_table | 132 / 143 | 92.3% |
| turn_on_lightbulb | 163 / 163 | 100.0% |
| rotate_blue_block_left | 66 / 67 | 98.5% |
| push_blue_block_left | 56 / 70 | 80.0% |
| close_drawer | 176 / 176 | 100.0% |
| push_red_block_left | 53 / 75 | 70.7% |
| lift_blue_block_table | 158 / 159 | 99.4% |
| place_in_drawer | 157 / 159 | 98.7% |
| move_slider_left | 229 / 229 | 100.0% |
| rotate_red_block_left | 56 / 63 | 88.9% |
| turn_on_led | 155 / 156 | 99.4% |
| stack_block | 136 / 167 | 81.4% |
| push_pink_block_left | 54 / 76 | 71.1% |
| lift_pink_block_drawer | 12 / 13 | 92.3% |
| rotate_pink_block_right | 58 / 70 | 82.9% |
| unstack_block | 55 / 56 | 98.2% |
| lift_blue_block_slider | 105 / 112 | 93.8% |
| push_red_block_right | 22 / 72 | 30.6% |
| rotate_pink_block_left | 47 / 54 | 87.0% |
| push_pink_block_right | 19 / 66 | 28.8% |
| lift_red_block_drawer | 16 / 16 | 100.0% |
| push_blue_block_right | 17 / 69 | 24.6% |

## 实验记录 - 2026-07-26 (3)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj/checkpoints/step-024474-epoch-03-loss=0.1583.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj`
  - Step: 024474
  - Epoch: 03
  - Loss: 0.1583
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 4.043

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 92.8% |
| 2 | 86.3% |
| 3 | 81.0% |
| 4 | 74.9% |
| 5 | 69.3% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 75 / 78 | 96.2% |
| move_slider_right | 271 / 271 | 100.0% |
| lift_red_block_slider | 123 / 130 | 94.6% |
| place_in_slider | 311 / 340 | 91.5% |
| turn_off_lightbulb | 141 / 141 | 100.0% |
| turn_off_led | 164 / 166 | 98.8% |
| push_into_drawer | 97 / 113 | 85.8% |
| lift_blue_block_drawer | 19 / 20 | 95.0% |
| close_drawer | 199 / 200 | 99.5% |
| lift_pink_block_slider | 122 / 130 | 93.8% |
| open_drawer | 331 / 335 | 98.8% |
| rotate_red_block_right | 71 / 74 | 95.9% |
| lift_red_block_table | 160 / 164 | 97.6% |
| lift_pink_block_table | 157 / 158 | 99.4% |
| move_slider_left | 249 / 249 | 100.0% |
| turn_on_lightbulb | 175 / 175 | 100.0% |
| rotate_blue_block_left | 68 / 68 | 100.0% |
| push_blue_block_left | 64 / 68 | 94.1% |
| turn_on_led | 172 / 178 | 96.6% |
| push_red_block_left | 67 / 79 | 84.8% |
| lift_blue_block_table | 158 / 160 | 98.8% |
| place_in_drawer | 174 / 174 | 100.0% |
| rotate_red_block_left | 61 / 64 | 95.3% |
| stack_block | 154 / 177 | 87.0% |
| push_pink_block_left | 66 / 77 | 85.7% |
| lift_blue_block_slider | 119 / 129 | 92.2% |
| lift_pink_block_drawer | 12 / 12 | 100.0% |
| unstack_block | 65 / 66 | 98.5% |
| push_red_block_right | 20 / 71 | 28.2% |
| rotate_pink_block_left | 53 / 56 | 94.6% |
| rotate_pink_block_right | 64 / 71 | 90.1% |
| push_pink_block_right | 24 / 65 | 36.9% |
| lift_red_block_drawer | 18 / 20 | 90.0% |
| push_blue_block_right | 19 / 71 | 26.8% |

## 实验记录 - 2026-07-26 (4)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj/checkpoints/step-016316-epoch-02-loss=0.2169.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj`
  - Step: 016316
  - Epoch: 02
  - Loss: 0.2169
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 4.026

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 93.2% |
| 2 | 86.5% |
| 3 | 80.5% |
| 4 | 75.1% |
| 5 | 67.3% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 74 / 77 | 96.1% |
| move_slider_right | 269 / 269 | 100.0% |
| lift_red_block_slider | 115 / 126 | 91.3% |
| place_in_slider | 303 / 339 | 89.4% |
| turn_off_lightbulb | 145 / 145 | 100.0% |
| turn_off_led | 165 / 167 | 98.8% |
| push_into_drawer | 98 / 116 | 84.5% |
| lift_blue_block_drawer | 17 / 18 | 94.4% |
| close_drawer | 198 / 198 | 100.0% |
| lift_pink_block_slider | 127 / 139 | 91.4% |
| open_drawer | 334 / 334 | 100.0% |
| rotate_red_block_right | 71 / 74 | 95.9% |
| lift_red_block_table | 161 / 165 | 97.6% |
| lift_pink_block_table | 153 / 157 | 97.5% |
| turn_on_lightbulb | 175 / 175 | 100.0% |
| rotate_blue_block_left | 67 / 67 | 100.0% |
| push_blue_block_left | 68 / 68 | 100.0% |
| turn_on_led | 169 / 174 | 97.1% |
| push_red_block_left | 73 / 79 | 92.4% |
| lift_blue_block_table | 161 / 163 | 98.8% |
| place_in_drawer | 172 / 173 | 99.4% |
| move_slider_left | 250 / 251 | 99.6% |
| rotate_red_block_left | 64 / 66 | 97.0% |
| stack_block | 150 / 175 | 85.7% |
| push_pink_block_left | 65 / 76 | 85.5% |
| lift_blue_block_slider | 119 / 133 | 89.5% |
| push_red_block_right | 23 / 72 | 31.9% |
| lift_pink_block_drawer | 12 / 14 | 85.7% |
| rotate_pink_block_right | 58 / 71 | 81.7% |
| unstack_block | 64 / 64 | 100.0% |
| rotate_pink_block_left | 52 / 55 | 94.5% |
| push_pink_block_right | 23 / 67 | 34.3% |
| push_blue_block_right | 14 / 68 | 20.6% |
| lift_red_block_drawer | 17 / 18 | 94.4% |

## 实验记录 - 2026-07-31

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_3dots_stageA2B_unlabel/checkpoints/step-025089-epoch-03-loss=0.7006.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_3dots_stageA2B_unlabel`
  - Step: 025089
  - Epoch: 03
  - Loss: 0.7006
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.193

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 91.0% |
| 2 | 76.0% |
| 3 | 62.0% |
| 4 | 50.2% |
| 5 | 40.1% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 59 / 66 | 89.4% |
| move_slider_right | 220 / 221 | 99.5% |
| lift_red_block_slider | 113 / 118 | 95.8% |
| turn_off_led | 126 / 127 | 99.2% |
| push_into_drawer | 69 / 83 | 83.1% |
| lift_blue_block_drawer | 14 / 14 | 100.0% |
| lift_pink_block_slider | 110 / 116 | 94.8% |
| open_drawer | 285 / 289 | 98.6% |
| lift_pink_block_table | 144 / 149 | 96.6% |
| push_blue_block_left | 50 / 61 | 82.0% |
| close_drawer | 165 / 165 | 100.0% |
| rotate_red_block_right | 58 / 67 | 86.6% |
| turn_on_led | 139 / 142 | 97.9% |
| stack_block | 142 / 171 | 83.0% |
| push_pink_block_right | 25 / 60 | 41.7% |
| push_red_block_left | 52 / 69 | 75.4% |
| lift_blue_block_table | 150 / 152 | 98.7% |
| rotate_blue_block_left | 53 / 59 | 89.8% |
| place_in_drawer | 164 / 164 | 100.0% |
| turn_off_lightbulb | 113 / 113 | 100.0% |
| place_in_slider | 21 / 328 | 6.4% |
| turn_on_lightbulb | 137 / 137 | 100.0% |
| move_slider_left | 206 / 206 | 100.0% |
| push_pink_block_left | 59 / 67 | 88.1% |
| lift_pink_block_drawer | 11 / 11 | 100.0% |
| rotate_pink_block_right | 58 / 63 | 92.1% |
| unstack_block | 57 / 60 | 95.0% |
| lift_blue_block_slider | 109 / 113 | 96.5% |
| lift_red_block_table | 141 / 149 | 94.6% |
| push_red_block_right | 19 / 67 | 28.4% |
| rotate_pink_block_left | 47 / 49 | 95.9% |
| rotate_red_block_left | 47 / 56 | 83.9% |
| lift_red_block_drawer | 15 / 16 | 93.8% |
| push_blue_block_right | 15 / 64 | 23.4% |

## 实验记录 - 2026-08-01

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_3dots_stageA2B_unlabel/checkpoints/step-016726-epoch-02-loss=0.2015.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_3dots_stageA2B_unlabel`
  - Step: 016726
  - Epoch: 02
  - Loss: 0.2015
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.275

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 91.8% |
| 2 | 78.3% |
| 3 | 64.5% |
| 4 | 51.9% |
| 5 | 41.0% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 63 / 68 | 92.6% |
| move_slider_right | 226 / 227 | 99.6% |
| lift_red_block_slider | 114 / 120 | 95.0% |
| turn_off_led | 133 / 135 | 98.5% |
| push_into_drawer | 76 / 90 | 84.4% |
| lift_blue_block_drawer | 16 / 16 | 100.0% |
| lift_pink_block_slider | 113 / 115 | 98.3% |
| lift_pink_block_table | 150 / 154 | 97.4% |
| open_drawer | 301 / 302 | 99.7% |
| push_blue_block_left | 51 / 64 | 79.7% |
| close_drawer | 165 / 165 | 100.0% |
| rotate_red_block_right | 62 / 67 | 92.5% |
| turn_on_led | 146 / 149 | 98.0% |
| stack_block | 134 / 168 | 79.8% |
| push_red_block_left | 50 / 67 | 74.6% |
| lift_blue_block_table | 156 / 157 | 99.4% |
| rotate_blue_block_left | 58 / 60 | 96.7% |
| place_in_drawer | 160 / 162 | 98.8% |
| turn_off_lightbulb | 109 / 110 | 99.1% |
| place_in_slider | 44 / 342 | 12.9% |
| turn_on_lightbulb | 136 / 136 | 100.0% |
| move_slider_left | 210 / 210 | 100.0% |
| rotate_red_block_left | 49 / 58 | 84.5% |
| push_pink_block_left | 61 / 71 | 85.9% |
| lift_red_block_table | 143 / 148 | 96.6% |
| lift_pink_block_drawer | 10 / 11 | 90.9% |
| rotate_pink_block_right | 59 / 65 | 90.8% |
| lift_blue_block_slider | 104 / 112 | 92.9% |
| unstack_block | 58 / 58 | 100.0% |
| push_red_block_right | 18 / 68 | 26.5% |
| rotate_pink_block_left | 46 / 49 | 93.9% |
| push_pink_block_right | 23 / 61 | 37.7% |
| lift_red_block_drawer | 15 / 16 | 93.8% |
| push_blue_block_right | 16 / 64 | 25.0% |

## 实验记录 - 2026-08-02

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_3dots_stageA(1epoch)2B_unlabel/checkpoints/step-025089-epoch-03-loss=0.4780.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_3dots_stageA(1epoch)2B_unlabel`
  - Step: 025089
  - Epoch: 03
  - Loss: 0.4780
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.223

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 91.3% |
| 2 | 77.1% |
| 3 | 62.5% |
| 4 | 50.9% |
| 5 | 40.5% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 65 / 68 | 95.6% |
| move_slider_right | 216 / 217 | 99.5% |
| lift_red_block_slider | 116 / 118 | 98.3% |
| turn_off_led | 135 / 138 | 97.8% |
| push_into_drawer | 71 / 83 | 85.5% |
| lift_blue_block_drawer | 13 / 13 | 100.0% |
| lift_pink_block_slider | 113 / 119 | 95.0% |
| place_in_slider | 31 / 339 | 9.1% |
| open_drawer | 294 / 295 | 99.7% |
| lift_pink_block_table | 145 / 148 | 98.0% |
| move_slider_left | 213 / 213 | 100.0% |
| push_blue_block_left | 54 / 61 | 88.5% |
| close_drawer | 159 / 159 | 100.0% |
| rotate_red_block_right | 64 / 70 | 91.4% |
| turn_on_led | 136 / 142 | 95.8% |
| push_red_block_left | 53 / 68 | 77.9% |
| lift_blue_block_table | 154 / 159 | 96.9% |
| rotate_blue_block_left | 53 / 58 | 91.4% |
| place_in_drawer | 164 / 164 | 100.0% |
| turn_off_lightbulb | 112 / 112 | 100.0% |
| rotate_red_block_left | 51 / 55 | 92.7% |
| stack_block | 136 / 168 | 81.0% |
| push_pink_block_left | 51 / 68 | 75.0% |
| turn_on_lightbulb | 130 / 130 | 100.0% |
| lift_red_block_table | 150 / 151 | 99.3% |
| lift_pink_block_drawer | 13 / 13 | 100.0% |
| rotate_pink_block_right | 57 / 64 | 89.1% |
| unstack_block | 58 / 58 | 100.0% |
| lift_blue_block_slider | 103 / 113 | 91.2% |
| push_red_block_right | 21 / 68 | 30.9% |
| rotate_pink_block_left | 39 / 47 | 83.0% |
| push_pink_block_right | 22 / 59 | 37.3% |
| lift_red_block_drawer | 15 / 16 | 93.8% |
| push_blue_block_right | 16 / 64 | 25.0% |

## 实验记录 - 2026-08-02 (2)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_3dots_stageA(1epoch)2B_unlabel/checkpoints/step-016726-epoch-02-loss=0.2718.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_3dots_stageA(1epoch)2B_unlabel`
  - Step: 016726
  - Epoch: 02
  - Loss: 0.2718
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.269

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 91.8% |
| 2 | 77.3% |
| 3 | 64.1% |
| 4 | 52.2% |
| 5 | 41.5% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 65 / 68 | 95.6% |
| move_slider_right | 227 / 228 | 99.6% |
| lift_red_block_slider | 114 / 116 | 98.3% |
| turn_off_led | 130 / 130 | 100.0% |
| push_into_drawer | 78 / 89 | 87.6% |
| lift_blue_block_drawer | 15 / 16 | 93.8% |
| lift_pink_block_slider | 122 / 123 | 99.2% |
| lift_pink_block_table | 145 / 152 | 95.4% |
| open_drawer | 291 / 294 | 99.0% |
| push_blue_block_left | 55 / 61 | 90.2% |
| close_drawer | 165 / 165 | 100.0% |
| rotate_red_block_right | 59 / 67 | 88.1% |
| turn_on_led | 143 / 149 | 96.0% |
| push_red_block_left | 57 / 67 | 85.1% |
| lift_blue_block_table | 153 / 154 | 99.4% |
| rotate_blue_block_left | 60 / 60 | 100.0% |
| place_in_drawer | 166 / 166 | 100.0% |
| turn_off_lightbulb | 120 / 120 | 100.0% |
| place_in_slider | 22 / 346 | 6.4% |
| turn_on_lightbulb | 134 / 135 | 99.3% |
| move_slider_left | 205 / 205 | 100.0% |
| rotate_red_block_left | 49 / 55 | 89.1% |
| stack_block | 139 / 172 | 80.8% |
| push_pink_block_left | 58 / 69 | 84.1% |
| lift_red_block_table | 146 / 149 | 98.0% |
| lift_pink_block_drawer | 14 / 14 | 100.0% |
| rotate_pink_block_right | 60 / 62 | 96.8% |
| unstack_block | 55 / 55 | 100.0% |
| lift_blue_block_slider | 113 / 115 | 98.3% |
| push_red_block_right | 20 / 68 | 29.4% |
| rotate_pink_block_left | 40 / 49 | 81.6% |
| push_pink_block_right | 20 / 58 | 34.5% |
| lift_red_block_drawer | 16 / 16 | 100.0% |
| push_blue_block_right | 13 / 61 | 21.3% |

