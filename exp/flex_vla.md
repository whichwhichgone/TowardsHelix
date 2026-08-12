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
- **模型 checkpoint**: `logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_h800/checkpoints/step-024474-epoch-03-loss=0.1583.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_h800`
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
- **模型 checkpoint**: `logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_h800/checkpoints/step-016316-epoch-02-loss=0.2169.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_h800`
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

## 实验记录 - 2026-08-03

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/defaultShare/zhaowei_global/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_baseline_h800/checkpoints/step-025116-epoch-03-loss=0.1066.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_baseline_h800`
  - Step: 025116
  - Epoch: 03
  - Loss: 0.1066
- **unnorm-key**: `calvin_abc2d_oe_baseline`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 4 卡（0-3），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.354

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 91.5% |
| 2 | 78.8% |
| 3 | 64.6% |
| 4 | 55.2% |
| 5 | 45.3% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 68 / 69 | 98.6% |
| move_slider_right | 243 / 243 | 100.0% |
| turn_off_led | 144 / 144 | 100.0% |
| push_into_drawer | 82 / 96 | 85.4% |
| lift_blue_block_drawer | 16 / 16 | 100.0% |
| place_in_slider | 134 / 324 | 41.4% |
| close_drawer | 168 / 168 | 100.0% |
| lift_pink_block_slider | 107 / 124 | 86.3% |
| lift_pink_block_table | 145 / 148 | 98.0% |
| open_drawer | 293 / 295 | 99.3% |
| push_blue_block_left | 47 / 62 | 75.8% |
| lift_red_block_slider | 105 / 121 | 86.8% |
| turn_off_lightbulb | 121 / 122 | 99.2% |
| rotate_red_block_right | 64 / 71 | 90.1% |
| turn_on_led | 144 / 145 | 99.3% |
| push_red_block_left | 50 / 73 | 68.5% |
| lift_blue_block_table | 147 / 148 | 99.3% |
| rotate_blue_block_left | 59 / 62 | 95.2% |
| place_in_drawer | 157 / 161 | 97.5% |
| move_slider_left | 213 / 213 | 100.0% |
| rotate_red_block_left | 53 / 55 | 96.4% |
| stack_block | 105 / 169 | 62.1% |
| turn_on_lightbulb | 149 / 150 | 99.3% |
| push_red_block_right | 23 / 70 | 32.9% |
| lift_red_block_table | 150 / 152 | 98.7% |
| lift_pink_block_drawer | 10 / 12 | 83.3% |
| unstack_block | 43 / 43 | 100.0% |
| lift_blue_block_slider | 106 / 113 | 93.8% |
| rotate_pink_block_left | 51 / 53 | 96.2% |
| rotate_pink_block_right | 48 / 66 | 72.7% |
| push_pink_block_left | 50 / 72 | 69.4% |
| push_pink_block_right | 25 / 61 | 41.0% |
| lift_red_block_drawer | 15 / 15 | 100.0% |
| push_blue_block_right | 19 / 65 | 29.2% |

## 实验记录 - 2026-08-03 (2)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/defaultShare/zhaowei_global/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_baseline_h800/checkpoints/step-016744-epoch-02-loss=0.1063.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_baseline_h800`
  - Step: 016744
  - Epoch: 02
  - Loss: 0.1063
- **unnorm-key**: `calvin_abc2d_oe_baseline`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 4 卡（0-3），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.292

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 89.4% |
| 2 | 77.3% |
| 3 | 63.7% |
| 4 | 53.9% |
| 5 | 44.9% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 67 / 69 | 97.1% |
| move_slider_right | 244 / 244 | 100.0% |
| lift_red_block_slider | 104 / 116 | 89.7% |
| place_in_slider | 137 / 314 | 43.6% |
| turn_off_lightbulb | 117 / 118 | 99.2% |
| turn_off_led | 141 / 141 | 100.0% |
| push_into_drawer | 88 / 99 | 88.9% |
| lift_blue_block_drawer | 17 / 18 | 94.4% |
| lift_pink_block_slider | 105 / 125 | 84.0% |
| lift_pink_block_table | 143 / 145 | 98.6% |
| open_drawer | 288 / 303 | 95.0% |
| rotate_red_block_right | 59 / 68 | 86.8% |
| turn_on_led | 143 / 144 | 99.3% |
| push_red_block_left | 43 / 73 | 58.9% |
| lift_blue_block_table | 145 / 146 | 99.3% |
| turn_on_lightbulb | 149 / 149 | 100.0% |
| rotate_blue_block_left | 56 / 58 | 96.6% |
| place_in_drawer | 151 / 151 | 100.0% |
| move_slider_left | 202 / 204 | 99.0% |
| rotate_red_block_left | 57 / 57 | 100.0% |
| close_drawer | 166 / 166 | 100.0% |
| stack_block | 109 / 165 | 66.1% |
| lift_red_block_table | 141 / 143 | 98.6% |
| lift_pink_block_drawer | 9 / 12 | 75.0% |
| rotate_pink_block_right | 49 / 65 | 75.4% |
| unstack_block | 47 / 48 | 97.9% |
| lift_blue_block_slider | 97 / 109 | 89.0% |
| push_red_block_right | 21 / 67 | 31.3% |
| rotate_pink_block_left | 49 / 52 | 94.2% |
| push_pink_block_left | 44 / 73 | 60.3% |
| push_blue_block_left | 43 / 64 | 67.2% |
| push_pink_block_right | 23 / 60 | 38.3% |
| push_blue_block_right | 25 / 64 | 39.1% |
| lift_red_block_drawer | 13 / 13 | 100.0% |

## 实验记录 - 2026-08-03 (3)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/defaultShare/zhaowei_global/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_baseline_h800/checkpoints/step-008372-epoch-01-loss=0.1210.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_baseline_h800`
  - Step: 008372
  - Epoch: 01
  - Loss: 0.1210
- **unnorm-key**: `calvin_abc2d_oe_baseline`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.384

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 92.4% |
| 2 | 79.6% |
| 3 | 66.5% |
| 4 | 55.0% |
| 5 | 44.9% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 68 / 69 | 98.6% |
| move_slider_right | 252 / 253 | 99.6% |
| lift_red_block_slider | 98 / 118 | 83.1% |
| turn_off_led | 141 / 142 | 99.3% |
| push_into_drawer | 85 / 101 | 84.2% |
| lift_blue_block_drawer | 16 / 17 | 94.1% |
| lift_pink_block_slider | 106 / 128 | 82.8% |
| lift_pink_block_table | 144 / 150 | 96.0% |
| open_drawer | 301 / 312 | 96.5% |
| place_in_slider | 150 / 312 | 48.1% |
| turn_on_lightbulb | 154 / 155 | 99.4% |
| rotate_blue_block_left | 61 / 63 | 96.8% |
| push_blue_block_left | 49 / 64 | 76.6% |
| close_drawer | 171 / 171 | 100.0% |
| turn_off_lightbulb | 112 / 119 | 94.1% |
| rotate_red_block_right | 65 / 70 | 92.9% |
| turn_on_led | 148 / 153 | 96.7% |
| push_red_block_left | 59 / 74 | 79.7% |
| lift_blue_block_table | 153 / 154 | 99.4% |
| place_in_drawer | 153 / 158 | 96.8% |
| move_slider_left | 213 / 216 | 98.6% |
| rotate_red_block_left | 54 / 55 | 98.2% |
| stack_block | 96 / 170 | 56.5% |
| push_pink_block_left | 58 / 71 | 81.7% |
| lift_red_block_table | 148 / 150 | 98.7% |
| rotate_pink_block_right | 53 / 66 | 80.3% |
| lift_blue_block_slider | 95 / 119 | 79.8% |
| unstack_block | 36 / 36 | 100.0% |
| push_red_block_right | 27 / 68 | 39.7% |
| rotate_pink_block_left | 48 / 53 | 90.6% |
| lift_pink_block_drawer | 8 / 11 | 72.7% |
| push_pink_block_right | 24 / 60 | 40.0% |
| push_blue_block_right | 22 / 60 | 36.7% |
| lift_red_block_drawer | 16 / 17 | 94.1% |

## 实验记录 - 2026-08-03 (4)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/zhaowei/workspace/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_baseline_h800/checkpoints/step-025116-epoch-03-loss=0.1066.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_baseline_h800`
  - Step: 025116
  - Epoch: 03
  - Loss: 0.1066
- **unnorm-key**: `calvin_abc2d_oe_baseline`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.321

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 90.6% |
| 2 | 77.3% |
| 3 | 64.3% |
| 4 | 55.3% |
| 5 | 44.6% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 68 / 68 | 100.0% |
| move_slider_right | 245 / 245 | 100.0% |
| lift_red_block_slider | 105 / 117 | 89.7% |
| place_in_slider | 135 / 323 | 41.8% |
| turn_off_lightbulb | 123 / 123 | 100.0% |
| turn_off_led | 137 / 138 | 99.3% |
| push_into_drawer | 83 / 97 | 85.6% |
| lift_blue_block_drawer | 17 / 18 | 94.4% |
| lift_pink_block_slider | 106 / 124 | 85.5% |
| lift_pink_block_table | 147 / 150 | 98.0% |
| open_drawer | 295 / 296 | 99.7% |
| turn_on_lightbulb | 145 / 145 | 100.0% |
| rotate_blue_block_left | 62 / 62 | 100.0% |
| push_blue_block_left | 43 / 62 | 69.4% |
| close_drawer | 167 / 167 | 100.0% |
| rotate_red_block_right | 63 / 71 | 88.7% |
| turn_on_led | 142 / 144 | 98.6% |
| push_red_block_left | 50 / 74 | 67.6% |
| lift_blue_block_table | 146 / 149 | 98.0% |
| place_in_drawer | 153 / 154 | 99.4% |
| move_slider_left | 211 / 211 | 100.0% |
| rotate_red_block_left | 55 / 57 | 96.5% |
| stack_block | 99 / 164 | 60.4% |
| lift_red_block_table | 148 / 151 | 98.0% |
| lift_pink_block_drawer | 12 / 15 | 80.0% |
| rotate_pink_block_right | 45 / 66 | 68.2% |
| unstack_block | 43 / 43 | 100.0% |
| lift_blue_block_slider | 97 / 110 | 88.2% |
| push_red_block_right | 21 / 68 | 30.9% |
| rotate_pink_block_left | 49 / 53 | 92.5% |
| push_pink_block_left | 52 / 72 | 72.2% |
| push_pink_block_right | 23 / 62 | 37.1% |
| lift_red_block_drawer | 14 / 14 | 100.0% |
| push_blue_block_right | 20 / 62 | 32.3% |

## 实验记录 - 2026-08-04

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/zhaowei/workspace/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_baseline_h800/checkpoints/step-016744-epoch-02-loss=0.1063.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_baseline_h800`
  - Step: 016744
  - Epoch: 02
  - Loss: 0.1063
- **unnorm-key**: `calvin_abc2d_oe_baseline`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.356

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 90.3% |
| 2 | 77.4% |
| 3 | 65.5% |
| 4 | 55.8% |
| 5 | 46.6% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 66 / 70 | 94.3% |
| move_slider_right | 247 / 247 | 100.0% |
| lift_red_block_slider | 102 / 111 | 91.9% |
| turn_off_led | 144 / 144 | 100.0% |
| push_into_drawer | 84 / 98 | 85.7% |
| lift_blue_block_drawer | 16 / 16 | 100.0% |
| place_in_slider | 152 / 316 | 48.1% |
| close_drawer | 172 / 172 | 100.0% |
| lift_pink_block_slider | 109 / 125 | 87.2% |
| lift_pink_block_table | 135 / 141 | 95.7% |
| open_drawer | 294 / 301 | 97.7% |
| turn_on_lightbulb | 150 / 150 | 100.0% |
| rotate_blue_block_left | 66 / 66 | 100.0% |
| rotate_red_block_right | 63 / 72 | 87.5% |
| turn_on_led | 144 / 147 | 98.0% |
| stack_block | 99 / 161 | 61.5% |
| push_pink_block_right | 24 / 61 | 39.3% |
| push_red_block_left | 44 / 72 | 61.1% |
| lift_blue_block_table | 155 / 155 | 100.0% |
| move_slider_left | 207 / 210 | 98.6% |
| rotate_red_block_left | 60 / 60 | 100.0% |
| push_red_block_right | 22 / 70 | 31.4% |
| lift_red_block_table | 140 / 144 | 97.2% |
| turn_off_lightbulb | 122 / 124 | 98.4% |
| place_in_drawer | 151 / 155 | 97.4% |
| lift_pink_block_drawer | 11 / 13 | 84.6% |
| rotate_pink_block_right | 47 / 65 | 72.3% |
| unstack_block | 43 / 43 | 100.0% |
| lift_blue_block_slider | 105 / 113 | 92.9% |
| rotate_pink_block_left | 51 / 53 | 96.2% |
| push_pink_block_left | 44 / 71 | 62.0% |
| push_blue_block_left | 50 / 63 | 79.4% |
| lift_red_block_drawer | 15 / 16 | 93.8% |
| push_blue_block_right | 22 / 65 | 33.8% |

## 实验记录 - 2026-08-04 (2)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_baseline_rerun/checkpoints/step-025116-epoch-03-loss=0.1088.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_baseline_rerun`
  - Step: 025116
  - Epoch: 03
  - Loss: 0.1088
- **unnorm-key**: `calvin_abc2d_oe_baseline`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.541

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 92.6% |
| 2 | 81.0% |
| 3 | 69.2% |
| 4 | 60.0% |
| 5 | 51.3% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 68 / 71 | 95.8% |
| move_slider_right | 257 / 257 | 100.0% |
| lift_red_block_slider | 97 / 121 | 80.2% |
| place_in_slider | 236 / 322 | 73.3% |
| turn_off_lightbulb | 123 / 124 | 99.2% |
| turn_off_led | 145 / 146 | 99.3% |
| push_into_drawer | 91 / 105 | 86.7% |
| lift_blue_block_drawer | 17 / 18 | 94.4% |
| lift_pink_block_slider | 111 / 126 | 88.1% |
| open_drawer | 312 / 312 | 100.0% |
| rotate_red_block_right | 60 / 73 | 82.2% |
| lift_red_block_table | 151 / 153 | 98.7% |
| lift_pink_block_table | 148 / 151 | 98.0% |
| turn_on_lightbulb | 154 / 155 | 99.4% |
| rotate_blue_block_left | 66 / 66 | 100.0% |
| turn_on_led | 158 / 160 | 98.8% |
| push_pink_block_right | 31 / 62 | 50.0% |
| close_drawer | 181 / 181 | 100.0% |
| lift_blue_block_table | 156 / 162 | 96.3% |
| place_in_drawer | 156 / 156 | 100.0% |
| move_slider_left | 219 / 221 | 99.1% |
| rotate_red_block_left | 57 / 60 | 95.0% |
| stack_block | 71 / 172 | 41.3% |
| push_pink_block_left | 54 / 72 | 75.0% |
| lift_pink_block_drawer | 10 / 10 | 100.0% |
| unstack_block | 30 / 31 | 96.8% |
| lift_blue_block_slider | 112 / 124 | 90.3% |
| push_red_block_right | 24 / 68 | 35.3% |
| rotate_pink_block_left | 51 / 54 | 94.4% |
| rotate_pink_block_right | 54 / 69 | 78.3% |
| push_red_block_left | 50 / 75 | 66.7% |
| push_blue_block_left | 53 / 64 | 82.8% |
| push_blue_block_right | 22 / 69 | 31.9% |
| lift_red_block_drawer | 16 / 18 | 88.9% |

## 实验记录 - 2026-08-04 (3)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_stageA2B/checkpoints/step-024474-epoch-03-loss=0.2663.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_stageA2B`
  - Step: 024474
  - Epoch: 03
  - Loss: 0.2663
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.594

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 93.5% |
| 2 | 82.1% |
| 3 | 70.6% |
| 4 | 61.2% |
| 5 | 52.0% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 65 / 68 | 95.6% |
| move_slider_right | 246 / 246 | 100.0% |
| lift_red_block_slider | 120 / 122 | 98.4% |
| turn_off_led | 156 / 156 | 100.0% |
| push_into_drawer | 84 / 95 | 88.4% |
| lift_blue_block_drawer | 12 / 16 | 75.0% |
| lift_pink_block_slider | 120 / 121 | 99.2% |
| lift_pink_block_table | 160 / 162 | 98.8% |
| place_in_slider | 125 / 349 | 35.8% |
| move_slider_left | 221 / 222 | 99.5% |
| open_drawer | 303 / 317 | 95.6% |
| push_blue_block_left | 54 / 62 | 87.1% |
| close_drawer | 171 / 171 | 100.0% |
| turn_off_lightbulb | 132 / 132 | 100.0% |
| rotate_red_block_right | 67 / 70 | 95.7% |
| turn_on_led | 142 / 149 | 95.3% |
| stack_block | 154 / 185 | 83.2% |
| push_pink_block_right | 28 / 61 | 45.9% |
| push_red_block_left | 53 / 72 | 73.6% |
| lift_blue_block_table | 161 / 162 | 99.4% |
| rotate_blue_block_left | 56 / 60 | 93.3% |
| place_in_drawer | 171 / 173 | 98.8% |
| rotate_red_block_left | 53 / 59 | 89.8% |
| push_pink_block_left | 64 / 72 | 88.9% |
| turn_on_lightbulb | 151 / 151 | 100.0% |
| lift_blue_block_slider | 114 / 121 | 94.2% |
| push_red_block_right | 27 / 69 | 39.1% |
| lift_red_block_table | 156 / 157 | 99.4% |
| lift_pink_block_drawer | 14 / 14 | 100.0% |
| rotate_pink_block_right | 65 / 66 | 98.5% |
| unstack_block | 62 / 62 | 100.0% |
| rotate_pink_block_left | 47 / 50 | 94.0% |
| lift_red_block_drawer | 19 / 19 | 100.0% |
| push_blue_block_right | 21 / 63 | 33.3% |

## 实验记录 - 2026-08-04 (4)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_stageA2B/checkpoints/step-016316-epoch-02-loss=0.2008.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_stageA2B`
  - Step: 016316
  - Epoch: 02
  - Loss: 0.2008
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.513

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 92.7% |
| 2 | 81.3% |
| 3 | 68.7% |
| 4 | 59.0% |
| 5 | 49.6% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 67 / 68 | 98.5% |
| move_slider_right | 243 / 243 | 100.0% |
| lift_red_block_slider | 116 / 119 | 97.5% |
| turn_off_led | 143 / 145 | 98.6% |
| push_into_drawer | 78 / 92 | 84.8% |
| lift_blue_block_drawer | 16 / 16 | 100.0% |
| lift_pink_block_slider | 126 / 127 | 99.2% |
| place_in_slider | 89 / 350 | 25.4% |
| open_drawer | 299 / 317 | 94.3% |
| rotate_red_block_right | 66 / 72 | 91.7% |
| lift_red_block_table | 156 / 157 | 99.4% |
| lift_pink_block_table | 154 / 158 | 97.5% |
| push_blue_block_left | 56 / 62 | 90.3% |
| close_drawer | 169 / 169 | 100.0% |
| turn_on_led | 142 / 151 | 94.0% |
| push_pink_block_right | 25 / 61 | 41.0% |
| push_red_block_left | 57 / 69 | 82.6% |
| lift_blue_block_table | 156 / 157 | 99.4% |
| rotate_blue_block_left | 59 / 61 | 96.7% |
| place_in_drawer | 167 / 168 | 99.4% |
| turn_on_lightbulb | 145 / 145 | 100.0% |
| move_slider_left | 215 / 216 | 99.5% |
| rotate_red_block_left | 62 / 63 | 98.4% |
| stack_block | 167 / 180 | 92.8% |
| push_pink_block_left | 62 / 68 | 91.2% |
| turn_off_lightbulb | 124 / 127 | 97.6% |
| lift_pink_block_drawer | 9 / 11 | 81.8% |
| rotate_pink_block_right | 61 / 64 | 95.3% |
| unstack_block | 67 / 67 | 100.0% |
| lift_blue_block_slider | 116 / 118 | 98.3% |
| push_red_block_right | 24 / 68 | 35.3% |
| rotate_pink_block_left | 48 / 50 | 96.0% |
| lift_red_block_drawer | 15 / 15 | 100.0% |
| push_blue_block_right | 14 / 63 | 22.2% |

## 实验记录 - 2026-08-05

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_stageA2B1epoch/checkpoints/step-024474-epoch-03-loss=0.2812.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_stageA2B1epoch`
  - Step: 024474
  - Epoch: 03
  - Loss: 0.2812
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.76

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 92.5% |
| 2 | 83.3% |
| 3 | 74.5% |
| 4 | 67.0% |
| 5 | 58.7% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 70 / 71 | 98.6% |
| move_slider_right | 257 / 257 | 100.0% |
| lift_red_block_slider | 115 / 120 | 95.8% |
| turn_off_led | 157 / 157 | 100.0% |
| push_into_drawer | 87 / 98 | 88.8% |
| lift_blue_block_drawer | 16 / 17 | 94.1% |
| lift_pink_block_slider | 124 / 127 | 97.6% |
| place_in_slider | 206 / 346 | 59.5% |
| open_drawer | 328 / 333 | 98.5% |
| rotate_red_block_right | 70 / 74 | 94.6% |
| lift_red_block_table | 159 / 159 | 100.0% |
| lift_pink_block_table | 158 / 163 | 96.9% |
| push_blue_block_left | 56 / 67 | 83.6% |
| close_drawer | 177 / 177 | 100.0% |
| turn_off_lightbulb | 134 / 134 | 100.0% |
| turn_on_led | 155 / 161 | 96.3% |
| stack_block | 153 / 181 | 84.5% |
| push_red_block_left | 60 / 75 | 80.0% |
| lift_blue_block_table | 160 / 161 | 99.4% |
| rotate_blue_block_left | 63 / 65 | 96.9% |
| place_in_drawer | 165 / 166 | 99.4% |
| turn_on_lightbulb | 159 / 159 | 100.0% |
| move_slider_left | 234 / 234 | 100.0% |
| rotate_red_block_left | 55 / 60 | 91.7% |
| push_pink_block_left | 59 / 76 | 77.6% |
| push_red_block_right | 28 / 71 | 39.4% |
| lift_pink_block_drawer | 11 / 11 | 100.0% |
| rotate_pink_block_right | 65 / 69 | 94.2% |
| unstack_block | 63 / 63 | 100.0% |
| lift_blue_block_slider | 114 / 124 | 91.9% |
| rotate_pink_block_left | 45 / 52 | 86.5% |
| push_pink_block_right | 23 / 63 | 36.5% |
| lift_red_block_drawer | 15 / 16 | 93.8% |
| push_blue_block_right | 19 / 66 | 28.8% |

## 实验记录 - 2026-08-05 (2)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_stageA2B1epoch/checkpoints/step-016316-epoch-02-loss=0.1870.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_stageA2B1epoch`
  - Step: 016316
  - Epoch: 02
  - Loss: 0.1870
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.901

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 93.1% |
| 2 | 85.8% |
| 3 | 78.2% |
| 4 | 69.8% |
| 5 | 63.2% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 74 / 75 | 98.7% |
| move_slider_right | 265 / 265 | 100.0% |
| lift_red_block_slider | 121 / 125 | 96.8% |
| turn_off_led | 157 / 157 | 100.0% |
| push_into_drawer | 89 / 111 | 80.2% |
| lift_blue_block_drawer | 17 / 17 | 100.0% |
| lift_pink_block_slider | 127 / 132 | 96.2% |
| place_in_slider | 226 / 339 | 66.7% |
| open_drawer | 338 / 341 | 99.1% |
| rotate_red_block_right | 73 / 73 | 100.0% |
| lift_red_block_table | 162 / 165 | 98.2% |
| lift_pink_block_table | 156 / 160 | 97.5% |
| move_slider_left | 240 / 241 | 99.6% |
| turn_on_lightbulb | 165 / 166 | 99.4% |
| rotate_blue_block_left | 66 / 67 | 98.5% |
| push_blue_block_left | 60 / 68 | 88.2% |
| close_drawer | 188 / 189 | 99.5% |
| turn_off_lightbulb | 134 / 134 | 100.0% |
| turn_on_led | 163 / 166 | 98.2% |
| stack_block | 180 / 189 | 95.2% |
| push_pink_block_right | 25 / 64 | 39.1% |
| push_red_block_left | 66 / 78 | 84.6% |
| lift_blue_block_table | 161 / 164 | 98.2% |
| place_in_drawer | 166 / 168 | 98.8% |
| rotate_red_block_left | 60 / 60 | 100.0% |
| push_pink_block_left | 58 / 74 | 78.4% |
| lift_pink_block_drawer | 11 / 13 | 84.6% |
| rotate_pink_block_right | 63 / 68 | 92.6% |
| unstack_block | 67 / 68 | 98.5% |
| lift_blue_block_slider | 116 / 124 | 93.5% |
| push_red_block_right | 26 / 71 | 36.6% |
| rotate_pink_block_left | 48 / 52 | 92.3% |
| lift_red_block_drawer | 13 / 15 | 86.7% |
| push_blue_block_right | 20 / 70 | 28.6% |

## 实验记录 - 2026-08-05 (3)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_stageA2B_unlabel/checkpoints/step-024474-epoch-03-loss=0.1676.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_stageA2B_unlabel`
  - Step: 024474
  - Epoch: 03
  - Loss: 0.1676
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.942

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 93.0% |
| 2 | 84.4% |
| 3 | 78.3% |
| 4 | 72.6% |
| 5 | 65.9% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 70 / 76 | 92.1% |
| move_slider_right | 270 / 271 | 99.6% |
| lift_red_block_slider | 118 / 124 | 95.2% |
| place_in_slider | 288 / 337 | 85.5% |
| turn_off_lightbulb | 135 / 135 | 100.0% |
| turn_off_led | 162 / 163 | 99.4% |
| push_into_drawer | 93 / 111 | 83.8% |
| lift_blue_block_drawer | 19 / 20 | 95.0% |
| close_drawer | 196 / 196 | 100.0% |
| lift_pink_block_slider | 123 / 129 | 95.3% |
| open_drawer | 336 / 337 | 99.7% |
| rotate_red_block_right | 61 / 74 | 82.4% |
| lift_red_block_table | 157 / 164 | 95.7% |
| lift_pink_block_table | 155 / 157 | 98.7% |
| turn_on_lightbulb | 168 / 168 | 100.0% |
| rotate_blue_block_left | 63 / 65 | 96.9% |
| push_blue_block_left | 64 / 69 | 92.8% |
| turn_on_led | 169 / 173 | 97.7% |
| stack_block | 164 / 184 | 89.1% |
| push_red_block_left | 67 / 78 | 85.9% |
| lift_blue_block_table | 160 / 164 | 97.6% |
| place_in_drawer | 162 / 167 | 97.0% |
| move_slider_left | 235 / 237 | 99.2% |
| rotate_red_block_left | 60 / 64 | 93.8% |
| push_pink_block_left | 67 / 76 | 88.2% |
| push_red_block_right | 24 / 71 | 33.8% |
| lift_pink_block_drawer | 14 / 14 | 100.0% |
| rotate_pink_block_right | 61 / 70 | 87.1% |
| unstack_block | 63 / 63 | 100.0% |
| lift_blue_block_slider | 113 / 123 | 91.9% |
| rotate_pink_block_left | 51 / 54 | 94.4% |
| push_pink_block_right | 23 / 65 | 35.4% |
| lift_red_block_drawer | 15 / 15 | 100.0% |
| push_blue_block_right | 16 / 69 | 23.2% |

## 实验记录 - 2026-08-05 (4)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_stageA2B_unlabel/checkpoints/step-016316-epoch-02-loss=0.1503.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_stageA2B_unlabel`
  - Step: 016316
  - Epoch: 02
  - Loss: 0.1503
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 4.002

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 93.0% |
| 2 | 85.4% |
| 3 | 80.2% |
| 4 | 74.2% |
| 5 | 67.4% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 72 / 77 | 93.5% |
| move_slider_right | 268 / 268 | 100.0% |
| lift_red_block_slider | 123 / 128 | 96.1% |
| place_in_slider | 288 / 341 | 84.5% |
| turn_off_lightbulb | 144 / 144 | 100.0% |
| turn_off_led | 159 / 159 | 100.0% |
| push_into_drawer | 90 / 106 | 84.9% |
| lift_blue_block_drawer | 18 / 18 | 100.0% |
| close_drawer | 208 / 208 | 100.0% |
| lift_pink_block_slider | 120 / 128 | 93.8% |
| open_drawer | 334 / 338 | 98.8% |
| rotate_red_block_right | 71 / 74 | 95.9% |
| lift_red_block_table | 160 / 163 | 98.2% |
| lift_pink_block_table | 159 / 162 | 98.1% |
| turn_on_lightbulb | 172 / 172 | 100.0% |
| rotate_blue_block_left | 62 / 68 | 91.2% |
| push_blue_block_left | 60 / 70 | 85.7% |
| turn_on_led | 169 / 175 | 96.6% |
| stack_block | 166 / 181 | 91.7% |
| push_pink_block_right | 26 / 65 | 40.0% |
| push_red_block_left | 64 / 77 | 83.1% |
| lift_blue_block_table | 158 / 162 | 97.5% |
| place_in_drawer | 169 / 170 | 99.4% |
| move_slider_left | 247 / 248 | 99.6% |
| rotate_red_block_left | 60 / 64 | 93.8% |
| push_pink_block_left | 66 / 75 | 88.0% |
| lift_pink_block_drawer | 12 / 14 | 85.7% |
| rotate_pink_block_right | 66 / 70 | 94.3% |
| unstack_block | 65 / 66 | 98.5% |
| lift_blue_block_slider | 118 / 125 | 94.4% |
| push_red_block_right | 22 / 71 | 31.0% |
| rotate_pink_block_left | 51 / 56 | 91.1% |
| lift_red_block_drawer | 16 / 16 | 100.0% |
| push_blue_block_right | 19 / 69 | 27.5% |

## 实验记录 - 2026-08-05 (5)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_stageA2B_unlabel/checkpoints/step-008158-epoch-01-loss=0.2301.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_stageA2B_unlabel`
  - Step: 008158
  - Epoch: 01
  - Loss: 0.2301
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.764

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 92.1% |
| 2 | 82.8% |
| 3 | 74.4% |
| 4 | 67.2% |
| 5 | 59.9% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| turn_off_led | 158 / 159 | 99.4% |
| push_into_drawer | 84 / 103 | 81.6% |
| lift_blue_block_drawer | 16 / 16 | 100.0% |
| lift_pink_block_slider | 120 / 127 | 94.5% |
| place_in_slider | 258 / 322 | 80.1% |
| open_drawer | 321 / 322 | 99.7% |
| rotate_red_block_right | 59 / 74 | 79.7% |
| lift_red_block_table | 157 / 159 | 98.7% |
| rotate_blue_block_right | 66 / 76 | 86.8% |
| lift_pink_block_table | 150 / 163 | 92.0% |
| turn_on_lightbulb | 165 / 165 | 100.0% |
| push_blue_block_left | 61 / 68 | 89.7% |
| close_drawer | 188 / 188 | 100.0% |
| lift_red_block_slider | 110 / 118 | 93.2% |
| turn_off_lightbulb | 137 / 138 | 99.3% |
| turn_on_led | 160 / 162 | 98.8% |
| push_pink_block_right | 27 / 64 | 42.2% |
| push_red_block_left | 65 / 75 | 86.7% |
| lift_blue_block_table | 153 / 157 | 97.5% |
| move_slider_right | 264 / 264 | 100.0% |
| rotate_blue_block_left | 54 / 66 | 81.8% |
| move_slider_left | 235 / 237 | 99.2% |
| rotate_red_block_left | 58 / 64 | 90.6% |
| stack_block | 129 / 179 | 72.1% |
| push_pink_block_left | 68 / 77 | 88.3% |
| lift_blue_block_slider | 106 / 117 | 90.6% |
| push_red_block_right | 30 / 71 | 42.3% |
| place_in_drawer | 163 / 165 | 98.8% |
| lift_pink_block_drawer | 11 / 12 | 91.7% |
| rotate_pink_block_right | 56 / 68 | 82.4% |
| unstack_block | 48 / 48 | 100.0% |
| rotate_pink_block_left | 48 / 55 | 87.3% |
| push_blue_block_right | 21 / 68 | 30.9% |
| lift_red_block_drawer | 18 / 18 | 100.0% |

## 实验记录 - 2026-08-05 (6)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_stageA2B1epoch_unlabel/checkpoints/step-024474-epoch-03-loss=0.2580.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_stageA2B1epoch_unlabel`
  - Step: 024474
  - Epoch: 03
  - Loss: 0.2580
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.876

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 91.5% |
| 2 | 83.9% |
| 3 | 76.9% |
| 4 | 70.6% |
| 5 | 64.7% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 66 / 78 | 84.6% |
| move_slider_right | 262 / 262 | 100.0% |
| lift_red_block_slider | 121 / 126 | 96.0% |
| place_in_slider | 288 / 331 | 87.0% |
| turn_off_lightbulb | 131 / 133 | 98.5% |
| turn_off_led | 160 / 161 | 99.4% |
| push_into_drawer | 88 / 105 | 83.8% |
| lift_blue_block_drawer | 19 / 19 | 100.0% |
| lift_pink_block_slider | 114 / 121 | 94.2% |
| open_drawer | 332 / 333 | 99.7% |
| rotate_red_block_right | 58 / 73 | 79.5% |
| lift_red_block_table | 160 / 162 | 98.8% |
| lift_pink_block_table | 154 / 160 | 96.2% |
| move_slider_left | 237 / 237 | 100.0% |
| turn_on_lightbulb | 168 / 169 | 99.4% |
| push_blue_block_left | 63 / 68 | 92.6% |
| close_drawer | 200 / 200 | 100.0% |
| turn_on_led | 168 / 172 | 97.7% |
| stack_block | 155 / 177 | 87.6% |
| push_red_block_left | 63 / 78 | 80.8% |
| lift_blue_block_table | 161 / 162 | 99.4% |
| rotate_blue_block_left | 60 / 66 | 90.9% |
| place_in_drawer | 155 / 159 | 97.5% |
| rotate_red_block_left | 57 / 64 | 89.1% |
| push_pink_block_left | 70 / 77 | 90.9% |
| lift_pink_block_drawer | 11 / 11 | 100.0% |
| unstack_block | 62 / 63 | 98.4% |
| lift_blue_block_slider | 102 / 118 | 86.4% |
| push_red_block_right | 28 / 71 | 39.4% |
| rotate_pink_block_left | 43 / 55 | 78.2% |
| rotate_pink_block_right | 60 / 70 | 85.7% |
| push_pink_block_right | 25 / 63 | 39.7% |
| lift_red_block_drawer | 14 / 14 | 100.0% |
| push_blue_block_right | 21 / 71 | 29.6% |

## 实验记录 - 2026-08-06

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_stageA2B1epoch_unlabel/checkpoints/step-016316-epoch-02-loss=0.1914.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_stageA2B1epoch_unlabel`
  - Step: 016316
  - Epoch: 02
  - Loss: 0.1914
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 4 卡（0-3），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 4.044

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 93.7% |
| 2 | 86.9% |
| 3 | 80.8% |
| 4 | 74.9% |
| 5 | 68.1% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 75 / 78 | 96.2% |
| move_slider_right | 268 / 268 | 100.0% |
| lift_red_block_slider | 114 / 125 | 91.2% |
| place_in_slider | 304 / 340 | 89.4% |
| turn_off_lightbulb | 147 / 147 | 100.0% |
| turn_off_led | 160 / 162 | 98.8% |
| push_into_drawer | 94 / 111 | 84.7% |
| lift_blue_block_drawer | 17 / 18 | 94.4% |
| lift_pink_block_slider | 127 / 132 | 96.2% |
| open_drawer | 342 / 346 | 98.8% |
| rotate_red_block_right | 61 / 73 | 83.6% |
| lift_red_block_table | 168 / 173 | 97.1% |
| lift_pink_block_table | 158 / 166 | 95.2% |
| move_slider_left | 252 / 252 | 100.0% |
| turn_on_lightbulb | 166 / 167 | 99.4% |
| rotate_blue_block_left | 60 / 66 | 90.9% |
| push_blue_block_left | 63 / 70 | 90.0% |
| close_drawer | 199 / 199 | 100.0% |
| turn_on_led | 166 / 172 | 96.5% |
| push_pink_block_right | 29 / 63 | 46.0% |
| push_red_block_left | 67 / 79 | 84.8% |
| lift_blue_block_table | 166 / 167 | 99.4% |
| place_in_drawer | 173 / 174 | 99.4% |
| rotate_red_block_left | 58 / 64 | 90.6% |
| stack_block | 170 / 189 | 89.9% |
| push_pink_block_left | 69 / 76 | 90.8% |
| push_red_block_right | 36 / 72 | 50.0% |
| lift_pink_block_drawer | 13 / 13 | 100.0% |
| rotate_pink_block_right | 59 / 70 | 84.3% |
| unstack_block | 69 / 69 | 100.0% |
| lift_blue_block_slider | 110 / 121 | 90.9% |
| rotate_pink_block_left | 42 / 56 | 75.0% |
| push_blue_block_right | 26 / 68 | 38.2% |
| lift_red_block_drawer | 16 / 17 | 94.1% |

## 实验记录 - 2026-08-06 (2)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_stageA2B1epoch_unlabel/checkpoints/step-008158-epoch-01-loss=0.3810.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_stageA2B1epoch_unlabel`
  - Step: 008158
  - Epoch: 01
  - Loss: 0.3810
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 4 卡（0-3），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.94

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 93.4% |
| 2 | 84.9% |
| 3 | 78.7% |
| 4 | 71.7% |
| 5 | 65.3% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 70 / 78 | 89.7% |
| move_slider_right | 274 / 274 | 100.0% |
| lift_red_block_slider | 114 / 125 | 91.2% |
| place_in_slider | 299 / 336 | 89.0% |
| turn_off_lightbulb | 137 / 140 | 97.9% |
| turn_off_led | 159 / 161 | 98.8% |
| push_into_drawer | 93 / 113 | 82.3% |
| lift_blue_block_drawer | 18 / 18 | 100.0% |
| close_drawer | 193 / 193 | 100.0% |
| lift_pink_block_slider | 118 / 128 | 92.2% |
| open_drawer | 326 / 334 | 97.6% |
| rotate_red_block_right | 53 / 74 | 71.6% |
| lift_red_block_table | 155 / 160 | 96.9% |
| lift_pink_block_table | 164 / 168 | 97.6% |
| turn_on_lightbulb | 173 / 174 | 99.4% |
| rotate_blue_block_left | 56 / 67 | 83.6% |
| push_blue_block_left | 64 / 66 | 97.0% |
| push_pink_block_right | 38 / 67 | 56.7% |
| push_red_block_left | 70 / 75 | 93.3% |
| lift_blue_block_table | 166 / 169 | 98.2% |
| place_in_drawer | 160 / 162 | 98.8% |
| move_slider_left | 236 / 238 | 99.2% |
| rotate_red_block_left | 55 / 65 | 84.6% |
| turn_on_led | 160 / 165 | 97.0% |
| stack_block | 154 / 183 | 84.2% |
| push_pink_block_left | 73 / 75 | 97.3% |
| lift_blue_block_slider | 110 / 124 | 88.7% |
| push_red_block_right | 33 / 72 | 45.8% |
| lift_pink_block_drawer | 13 / 15 | 86.7% |
| rotate_pink_block_right | 60 / 68 | 88.2% |
| unstack_block | 62 / 62 | 100.0% |
| rotate_pink_block_left | 39 / 54 | 72.2% |
| push_blue_block_right | 29 / 68 | 42.6% |
| lift_red_block_drawer | 16 / 16 | 100.0% |

## 实验记录 - 2026-08-06 (3)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_rerun_last12/checkpoints/step-008158-epoch-01-loss=0.3330.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_rerun_last12`
  - Step: 008158
  - Epoch: 01
  - Loss: 0.3330
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 4 卡（0-3），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 1.401

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 67.0% |
| 2 | 36.8% |
| 3 | 20.4% |
| 4 | 10.0% |
| 5 | 5.9% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| turn_off_led | 34 / 101 | 33.7% |
| lift_pink_block_slider | 35 / 70 | 50.0% |
| place_in_slider | 72 / 115 | 62.6% |
| open_drawer | 171 / 205 | 83.4% |
| rotate_red_block_right | 17 / 61 | 27.9% |
| rotate_blue_block_right | 19 / 58 | 32.8% |
| lift_pink_block_table | 52 / 75 | 69.3% |
| turn_on_lightbulb | 87 / 95 | 91.6% |
| rotate_blue_block_left | 13 / 52 | 25.0% |
| push_blue_block_left | 28 / 55 | 50.9% |
| close_drawer | 76 / 82 | 92.7% |
| lift_red_block_slider | 42 / 68 | 61.8% |
| turn_off_lightbulb | 62 / 83 | 74.7% |
| lift_blue_block_table | 31 / 71 | 43.7% |
| place_in_drawer | 52 / 62 | 83.9% |
| move_slider_right | 153 / 158 | 96.8% |
| move_slider_left | 126 / 137 | 92.0% |
| rotate_red_block_left | 15 / 45 | 33.3% |
| push_red_block_left | 29 / 58 | 50.0% |
| lift_red_block_table | 46 / 79 | 58.2% |
| turn_on_led | 65 / 102 | 63.7% |
| lift_blue_block_slider | 38 / 72 | 52.8% |
| rotate_pink_block_left | 15 / 40 | 37.5% |
| push_into_drawer | 25 / 52 | 48.1% |
| push_red_block_right | 17 / 55 | 30.9% |
| push_pink_block_left | 34 / 61 | 55.7% |
| stack_block | 6 / 53 | 11.3% |
| unstack_block | 4 / 4 | 100.0% |
| rotate_pink_block_right | 6 / 61 | 9.8% |
| push_pink_block_right | 14 / 45 | 31.1% |
| lift_red_block_drawer | 3 / 5 | 60.0% |
| push_blue_block_right | 13 / 55 | 23.6% |
| lift_blue_block_drawer | 1 / 6 | 16.7% |
| lift_pink_block_drawer | 0 / 1 | 0.0% |

## 实验记录 - 2026-08-07

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_rerun_last12/checkpoints/step-016316-epoch-02-loss=0.2495.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_rerun_last12`
  - Step: 016316
  - Epoch: 02
  - Loss: 0.2495
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 4 卡（0-3），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 2.289

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 78.9% |
| 2 | 57.3% |
| 3 | 41.4% |
| 4 | 30.1% |
| 5 | 21.2% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 26 / 65 | 40.0% |
| move_slider_right | 194 / 202 | 96.0% |
| turn_off_led | 104 / 112 | 92.9% |
| push_into_drawer | 47 / 66 | 71.2% |
| lift_blue_block_drawer | 11 / 12 | 91.7% |
| lift_pink_block_slider | 63 / 93 | 67.7% |
| place_in_slider | 143 / 221 | 64.7% |
| open_drawer | 179 / 268 | 66.8% |
| push_blue_block_left | 43 / 62 | 69.4% |
| close_drawer | 127 / 128 | 99.2% |
| lift_red_block_slider | 67 / 83 | 80.7% |
| push_pink_block_right | 30 / 50 | 60.0% |
| lift_pink_block_table | 90 / 112 | 80.4% |
| push_red_block_left | 46 / 62 | 74.2% |
| lift_blue_block_table | 98 / 113 | 86.7% |
| rotate_blue_block_left | 40 / 60 | 66.7% |
| place_in_drawer | 87 / 95 | 91.6% |
| turn_off_lightbulb | 88 / 97 | 90.7% |
| turn_on_lightbulb | 99 / 117 | 84.6% |
| move_slider_left | 153 / 177 | 86.4% |
| turn_on_led | 120 / 124 | 96.8% |
| stack_block | 38 / 119 | 31.9% |
| lift_blue_block_slider | 66 / 84 | 78.6% |
| push_red_block_right | 30 / 63 | 47.6% |
| lift_red_block_table | 104 / 117 | 88.9% |
| lift_pink_block_drawer | 4 / 5 | 80.0% |
| rotate_pink_block_right | 16 / 61 | 26.2% |
| rotate_pink_block_left | 26 / 46 | 56.5% |
| push_pink_block_left | 52 / 68 | 76.5% |
| rotate_red_block_right | 19 / 60 | 31.7% |
| push_blue_block_right | 26 / 59 | 44.1% |
| rotate_red_block_left | 37 / 54 | 68.5% |
| unstack_block | 11 / 12 | 91.7% |
| lift_red_block_drawer | 5 / 10 | 50.0% |

## 实验记录 - 2026-08-07 (2)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_rerun_last12/checkpoints/step-024474-epoch-03-loss=0.5615.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_rerun_last12`
  - Step: 024474
  - Epoch: 03
  - Loss: 0.5615
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 4 卡（0-3），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 2.502

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 82.8% |
| 2 | 62.4% |
| 3 | 46.3% |
| 4 | 34.4% |
| 5 | 24.3% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 45 / 66 | 68.2% |
| move_slider_right | 199 / 204 | 97.5% |
| turn_off_led | 113 / 115 | 98.3% |
| lift_pink_block_slider | 65 / 102 | 63.7% |
| place_in_slider | 169 / 238 | 71.0% |
| open_drawer | 215 / 270 | 79.6% |
| lift_pink_block_table | 98 / 112 | 87.5% |
| turn_on_lightbulb | 117 / 131 | 89.3% |
| push_blue_block_left | 47 / 63 | 74.6% |
| close_drawer | 136 / 139 | 97.8% |
| lift_red_block_slider | 75 / 100 | 75.0% |
| rotate_red_block_right | 26 / 67 | 38.8% |
| turn_on_led | 119 / 129 | 92.2% |
| push_pink_block_right | 24 / 55 | 43.6% |
| push_red_block_left | 40 / 69 | 58.0% |
| lift_blue_block_table | 110 / 120 | 91.7% |
| rotate_blue_block_left | 42 / 59 | 71.2% |
| place_in_drawer | 96 / 101 | 95.0% |
| turn_off_lightbulb | 87 / 103 | 84.5% |
| move_slider_left | 170 / 194 | 87.6% |
| push_pink_block_left | 50 / 71 | 70.4% |
| lift_red_block_table | 100 / 114 | 87.7% |
| lift_pink_block_drawer | 4 / 7 | 57.1% |
| stack_block | 44 / 127 | 34.6% |
| unstack_block | 22 / 23 | 95.7% |
| lift_blue_block_slider | 63 / 87 | 72.4% |
| push_red_block_right | 22 / 67 | 32.8% |
| push_into_drawer | 60 / 74 | 81.1% |
| rotate_pink_block_left | 38 / 48 | 79.2% |
| rotate_pink_block_right | 26 / 64 | 40.6% |
| lift_blue_block_drawer | 12 / 12 | 100.0% |
| rotate_red_block_left | 36 / 52 | 69.2% |
| push_blue_block_right | 22 / 63 | 34.9% |
| lift_red_block_drawer | 10 / 13 | 76.9% |

## 实验记录 - 2026-08-07 (3)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_rerun_last24/checkpoints/step-008158-epoch-01-loss=0.4012.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_rerun_last24`
  - Step: 008158
  - Epoch: 01
  - Loss: 0.4012
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 4 卡（0-3），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 2.961

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 83.8% |
| 2 | 68.5% |
| 3 | 56.4% |
| 4 | 47.7% |
| 5 | 39.7% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 56 / 74 | 75.7% |
| move_slider_right | 224 / 226 | 99.1% |
| lift_red_block_slider | 91 / 106 | 85.8% |
| place_in_slider | 219 / 278 | 78.8% |
| turn_off_lightbulb | 107 / 108 | 99.1% |
| turn_off_led | 121 / 126 | 96.0% |
| push_into_drawer | 70 / 90 | 77.8% |
| lift_blue_block_drawer | 15 / 16 | 93.8% |
| lift_pink_block_slider | 98 / 110 | 89.1% |
| open_drawer | 276 / 281 | 98.2% |
| lift_pink_block_table | 116 / 123 | 94.3% |
| turn_on_lightbulb | 135 / 136 | 99.3% |
| push_blue_block_left | 38 / 65 | 58.5% |
| close_drawer | 153 / 155 | 98.7% |
| push_pink_block_right | 21 / 57 | 36.8% |
| rotate_blue_block_left | 52 / 64 | 81.2% |
| lift_blue_block_table | 125 / 132 | 94.7% |
| place_in_drawer | 122 / 124 | 98.4% |
| move_slider_left | 183 / 190 | 96.3% |
| rotate_red_block_left | 52 / 63 | 82.5% |
| turn_on_led | 144 / 146 | 98.6% |
| push_red_block_left | 34 / 73 | 46.6% |
| lift_pink_block_drawer | 10 / 12 | 83.3% |
| stack_block | 68 / 139 | 48.9% |
| unstack_block | 29 / 30 | 96.7% |
| lift_blue_block_slider | 72 / 93 | 77.4% |
| lift_red_block_table | 126 / 136 | 92.6% |
| push_red_block_right | 24 / 68 | 35.3% |
| rotate_pink_block_left | 28 / 53 | 52.8% |
| rotate_pink_block_right | 39 / 68 | 57.4% |
| rotate_red_block_right | 47 / 71 | 66.2% |
| push_pink_block_left | 37 / 73 | 50.7% |
| lift_red_block_drawer | 10 / 11 | 90.9% |
| push_blue_block_right | 19 / 67 | 28.4% |

## 实验记录 - 2026-08-08

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_rerun_last24/checkpoints/step-016316-epoch-02-loss=0.3816.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_rerun_last24`
  - Step: 016316
  - Epoch: 02
  - Loss: 0.3816
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 4 卡（0-3），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.46

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 87.2% |
| 2 | 77.6% |
| 3 | 68.3% |
| 4 | 60.3% |
| 5 | 52.6% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 69 / 76 | 90.8% |
| move_slider_right | 246 / 247 | 99.6% |
| turn_off_led | 142 / 147 | 96.6% |
| push_into_drawer | 80 / 99 | 80.8% |
| lift_blue_block_drawer | 16 / 16 | 100.0% |
| lift_pink_block_slider | 113 / 120 | 94.2% |
| place_in_slider | 259 / 302 | 85.8% |
| open_drawer | 307 / 308 | 99.7% |
| rotate_red_block_right | 53 / 72 | 73.6% |
| lift_red_block_table | 140 / 145 | 96.6% |
| lift_pink_block_table | 128 / 141 | 90.8% |
| turn_on_lightbulb | 156 / 156 | 100.0% |
| rotate_blue_block_left | 62 / 66 | 93.9% |
| push_blue_block_left | 44 / 66 | 66.7% |
| close_drawer | 180 / 180 | 100.0% |
| lift_red_block_slider | 106 / 122 | 86.9% |
| turn_off_lightbulb | 119 / 119 | 100.0% |
| turn_on_led | 154 / 154 | 100.0% |
| push_red_block_left | 37 / 73 | 50.7% |
| lift_blue_block_table | 146 / 148 | 98.6% |
| place_in_drawer | 148 / 149 | 99.3% |
| move_slider_left | 220 / 224 | 98.2% |
| rotate_red_block_left | 52 / 61 | 85.2% |
| stack_block | 112 / 158 | 70.9% |
| push_pink_block_left | 38 / 73 | 52.1% |
| push_red_block_right | 23 / 70 | 32.9% |
| lift_pink_block_drawer | 8 / 11 | 72.7% |
| unstack_block | 45 / 45 | 100.0% |
| lift_blue_block_slider | 93 / 112 | 83.0% |
| rotate_pink_block_left | 47 / 55 | 85.5% |
| rotate_pink_block_right | 57 / 69 | 82.6% |
| push_pink_block_right | 29 / 65 | 44.6% |
| lift_red_block_drawer | 15 / 16 | 93.8% |
| push_blue_block_right | 16 / 69 | 23.2% |

## 实验记录 - 2026-08-08 (2)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_rerun_last24/checkpoints/step-024474-epoch-03-loss=0.1619.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_rerun_last24`
  - Step: 024474
  - Epoch: 03
  - Loss: 0.1619
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 4 卡（0-3），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.457

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 88.0% |
| 2 | 76.9% |
| 3 | 67.5% |
| 4 | 60.3% |
| 5 | 53.0% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 64 / 73 | 87.7% |
| move_slider_right | 253 / 253 | 100.0% |
| turn_off_led | 143 / 144 | 99.3% |
| push_into_drawer | 88 / 105 | 83.8% |
| lift_pink_block_slider | 111 / 122 | 91.0% |
| place_in_slider | 250 / 298 | 83.9% |
| open_drawer | 309 / 312 | 99.0% |
| rotate_red_block_right | 55 / 72 | 76.4% |
| lift_red_block_table | 138 / 142 | 97.2% |
| lift_pink_block_table | 135 / 141 | 95.7% |
| turn_on_lightbulb | 148 / 149 | 99.3% |
| push_blue_block_left | 44 / 68 | 64.7% |
| close_drawer | 168 / 168 | 100.0% |
| lift_red_block_slider | 108 / 119 | 90.8% |
| turn_off_lightbulb | 124 / 125 | 99.2% |
| turn_on_led | 166 / 166 | 100.0% |
| stack_block | 116 / 159 | 73.0% |
| push_red_block_left | 47 / 77 | 61.0% |
| lift_blue_block_table | 141 / 145 | 97.2% |
| rotate_blue_block_left | 64 / 67 | 95.5% |
| place_in_drawer | 148 / 150 | 98.7% |
| move_slider_left | 210 / 217 | 96.8% |
| rotate_red_block_left | 51 / 62 | 82.3% |
| lift_pink_block_drawer | 10 / 11 | 90.9% |
| rotate_pink_block_right | 57 / 67 | 85.1% |
| unstack_block | 45 / 47 | 95.7% |
| push_red_block_right | 20 / 70 | 28.6% |
| rotate_pink_block_left | 47 / 54 | 87.0% |
| push_pink_block_left | 40 / 76 | 52.6% |
| push_pink_block_right | 23 / 62 | 37.1% |
| lift_blue_block_drawer | 16 / 17 | 94.1% |
| lift_blue_block_slider | 89 / 109 | 81.7% |
| lift_red_block_drawer | 12 / 12 | 100.0% |
| push_blue_block_right | 17 / 68 | 25.0% |

## 实验记录 - 2026-08-09

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_rerun/checkpoints/step-024474-epoch-03-loss=0.3675.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_rerun`
  - Step: 024474
  - Epoch: 03
  - Loss: 0.3675
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.883

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 92.6% |
| 2 | 84.1% |
| 3 | 76.9% |
| 4 | 70.8% |
| 5 | 63.9% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 72 / 75 | 96.0% |
| move_slider_right | 267 / 267 | 100.0% |
| lift_red_block_slider | 119 / 123 | 96.7% |
| place_in_slider | 298 / 339 | 87.9% |
| turn_off_lightbulb | 137 / 138 | 99.3% |
| turn_off_led | 159 / 161 | 98.8% |
| push_into_drawer | 95 / 108 | 88.0% |
| lift_blue_block_drawer | 17 / 19 | 89.5% |
| close_drawer | 194 / 194 | 100.0% |
| lift_pink_block_slider | 126 / 132 | 95.5% |
| open_drawer | 331 / 331 | 100.0% |
| rotate_red_block_right | 71 / 75 | 94.7% |
| lift_red_block_table | 159 / 159 | 100.0% |
| lift_pink_block_table | 156 / 158 | 98.7% |
| turn_on_lightbulb | 169 / 169 | 100.0% |
| rotate_blue_block_left | 65 / 66 | 98.5% |
| turn_on_led | 168 / 173 | 97.1% |
| stack_block | 133 / 177 | 75.1% |
| push_red_block_left | 57 / 78 | 73.1% |
| lift_blue_block_table | 157 / 158 | 99.4% |
| place_in_drawer | 163 / 165 | 98.8% |
| move_slider_left | 238 / 241 | 98.8% |
| rotate_red_block_left | 63 / 65 | 96.9% |
| push_pink_block_left | 51 / 74 | 68.9% |
| lift_pink_block_drawer | 13 / 14 | 92.9% |
| rotate_pink_block_right | 63 / 71 | 88.7% |
| unstack_block | 51 / 51 | 100.0% |
| lift_blue_block_slider | 107 / 119 | 89.9% |
| push_red_block_right | 22 / 72 | 30.6% |
| rotate_pink_block_left | 52 / 55 | 94.5% |
| push_pink_block_right | 22 / 66 | 33.3% |
| push_blue_block_left | 58 / 66 | 87.9% |
| push_blue_block_right | 15 / 69 | 21.7% |
| lift_red_block_drawer | 15 / 16 | 93.8% |

## 实验记录 - 2026-08-09 (2)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_rerun/checkpoints/step-016316-epoch-02-loss=0.1740.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_rerun`
  - Step: 016316
  - Epoch: 02
  - Loss: 0.1740
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.85

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 91.4% |
| 2 | 82.9% |
| 3 | 76.4% |
| 4 | 70.0% |
| 5 | 64.3% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 71 / 76 | 93.4% |
| move_slider_right | 260 / 260 | 100.0% |
| lift_red_block_slider | 119 / 124 | 96.0% |
| place_in_slider | 283 / 334 | 84.7% |
| turn_off_lightbulb | 135 / 135 | 100.0% |
| turn_off_led | 154 / 154 | 100.0% |
| push_into_drawer | 93 / 106 | 87.7% |
| lift_blue_block_drawer | 18 / 19 | 94.7% |
| close_drawer | 188 / 188 | 100.0% |
| lift_pink_block_slider | 127 / 133 | 95.5% |
| open_drawer | 325 / 325 | 100.0% |
| rotate_red_block_right | 64 / 73 | 87.7% |
| lift_red_block_table | 155 / 156 | 99.4% |
| lift_pink_block_table | 150 / 154 | 97.4% |
| turn_on_lightbulb | 173 / 173 | 100.0% |
| rotate_blue_block_left | 65 / 68 | 95.6% |
| turn_on_led | 165 / 167 | 98.8% |
| push_red_block_left | 58 / 76 | 76.3% |
| lift_blue_block_table | 161 / 161 | 100.0% |
| place_in_drawer | 164 / 166 | 98.8% |
| move_slider_left | 244 / 245 | 99.6% |
| rotate_red_block_left | 57 / 63 | 90.5% |
| stack_block | 144 / 176 | 81.8% |
| push_pink_block_left | 60 / 76 | 78.9% |
| lift_pink_block_drawer | 13 / 13 | 100.0% |
| rotate_pink_block_right | 48 / 68 | 70.6% |
| unstack_block | 57 / 57 | 100.0% |
| lift_blue_block_slider | 108 / 115 | 93.9% |
| push_red_block_right | 22 / 72 | 30.6% |
| rotate_pink_block_left | 51 / 54 | 94.4% |
| push_blue_block_left | 56 / 67 | 83.6% |
| push_pink_block_right | 26 / 66 | 39.4% |
| push_blue_block_right | 19 / 70 | 27.1% |
| lift_red_block_drawer | 17 / 17 | 100.0% |

## 实验记录 - 2026-08-09 (3)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_rerun/checkpoints/step-008158-epoch-01-loss=0.3930.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_rerun`
  - Step: 008158
  - Epoch: 01
  - Loss: 0.3930
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.49

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 89.3% |
| 2 | 78.5% |
| 3 | 67.7% |
| 4 | 60.5% |
| 5 | 53.0% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 67 / 75 | 89.3% |
| move_slider_right | 253 / 253 | 100.0% |
| lift_red_block_slider | 112 / 119 | 94.1% |
| turn_off_led | 150 / 152 | 98.7% |
| push_into_drawer | 89 / 102 | 87.3% |
| lift_blue_block_drawer | 17 / 20 | 85.0% |
| place_in_slider | 248 / 309 | 80.3% |
| close_drawer | 169 / 169 | 100.0% |
| lift_pink_block_slider | 108 / 122 | 88.5% |
| open_drawer | 307 / 308 | 99.7% |
| rotate_red_block_right | 61 / 74 | 82.4% |
| lift_red_block_table | 146 / 152 | 96.1% |
| lift_pink_block_table | 134 / 143 | 93.7% |
| turn_on_lightbulb | 164 / 164 | 100.0% |
| rotate_blue_block_left | 64 / 66 | 97.0% |
| turn_on_led | 153 / 158 | 96.8% |
| push_red_block_left | 52 / 75 | 69.3% |
| lift_blue_block_table | 142 / 144 | 98.6% |
| place_in_drawer | 143 / 147 | 97.3% |
| turn_off_lightbulb | 123 / 123 | 100.0% |
| move_slider_left | 217 / 217 | 100.0% |
| rotate_red_block_left | 58 / 62 | 93.5% |
| push_pink_block_left | 46 / 74 | 62.2% |
| stack_block | 107 / 158 | 67.7% |
| push_red_block_right | 24 / 70 | 34.3% |
| lift_pink_block_drawer | 11 / 12 | 91.7% |
| rotate_pink_block_right | 58 / 68 | 85.3% |
| unstack_block | 41 / 42 | 97.6% |
| lift_blue_block_slider | 91 / 113 | 80.5% |
| rotate_pink_block_left | 46 / 52 | 88.5% |
| push_blue_block_left | 44 / 67 | 65.7% |
| push_pink_block_right | 19 / 66 | 28.8% |
| lift_red_block_drawer | 13 / 16 | 81.2% |
| push_blue_block_right | 13 / 68 | 19.1% |

## 实验记录 - 2026-08-10

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft/checkpoints/step-016316-epoch-02-loss=0.1471.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft`
  - Step: 016316
  - Epoch: 02
  - Loss: 0.1471
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 4.149

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 93.5% |
| 2 | 87.3% |
| 3 | 83.1% |
| 4 | 78.1% |
| 5 | 72.9% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 75 / 77 | 97.4% |
| move_slider_right | 271 / 271 | 100.0% |
| lift_red_block_slider | 125 / 129 | 96.9% |
| place_in_slider | 325 / 353 | 92.1% |
| turn_off_lightbulb | 147 / 148 | 99.3% |
| turn_off_led | 166 / 169 | 98.2% |
| push_into_drawer | 100 / 119 | 84.0% |
| lift_blue_block_drawer | 19 / 19 | 100.0% |
| close_drawer | 204 / 205 | 99.5% |
| lift_pink_block_slider | 130 / 133 | 97.7% |
| open_drawer | 343 / 344 | 99.7% |
| rotate_red_block_right | 72 / 74 | 97.3% |
| lift_red_block_table | 164 / 164 | 100.0% |
| lift_pink_block_table | 165 / 165 | 100.0% |
| move_slider_left | 247 / 247 | 100.0% |
| turn_on_lightbulb | 180 / 180 | 100.0% |
| rotate_blue_block_left | 68 / 68 | 100.0% |
| push_blue_block_left | 66 / 69 | 95.7% |
| turn_on_led | 172 / 179 | 96.1% |
| push_red_block_left | 76 / 78 | 97.4% |
| lift_blue_block_table | 164 / 164 | 100.0% |
| place_in_drawer | 177 / 178 | 99.4% |
| rotate_red_block_left | 63 / 64 | 98.4% |
| stack_block | 164 / 182 | 90.1% |
| push_pink_block_left | 60 / 77 | 77.9% |
| lift_blue_block_slider | 124 / 131 | 94.7% |
| lift_pink_block_drawer | 12 / 14 | 85.7% |
| rotate_pink_block_right | 70 / 71 | 98.6% |
| unstack_block | 65 / 66 | 98.5% |
| push_red_block_right | 22 / 71 | 31.0% |
| rotate_pink_block_left | 53 / 54 | 98.1% |
| push_pink_block_right | 26 / 68 | 38.2% |
| lift_red_block_drawer | 18 / 18 | 100.0% |
| push_blue_block_right | 16 / 71 | 22.5% |

## 实验记录 - 2026-08-10 (2)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft/checkpoints/step-024474-epoch-03-loss=0.3493.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft`
  - Step: 024474
  - Epoch: 03
  - Loss: 0.3493
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 4.053

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 93.0% |
| 2 | 85.7% |
| 3 | 80.8% |
| 4 | 75.7% |
| 5 | 70.1% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 73 / 76 | 96.1% |
| move_slider_right | 268 / 268 | 100.0% |
| lift_red_block_slider | 125 / 129 | 96.9% |
| place_in_slider | 301 / 348 | 86.5% |
| turn_off_lightbulb | 145 / 146 | 99.3% |
| turn_off_led | 165 / 165 | 100.0% |
| push_into_drawer | 93 / 110 | 84.5% |
| lift_blue_block_drawer | 17 / 18 | 94.4% |
| lift_pink_block_slider | 130 / 131 | 99.2% |
| open_drawer | 334 / 336 | 99.4% |
| rotate_red_block_right | 71 / 74 | 95.9% |
| lift_red_block_table | 167 / 168 | 99.4% |
| lift_pink_block_table | 157 / 165 | 95.2% |
| turn_on_lightbulb | 173 / 174 | 99.4% |
| rotate_blue_block_left | 68 / 68 | 100.0% |
| push_blue_block_left | 68 / 70 | 97.1% |
| close_drawer | 203 / 203 | 100.0% |
| turn_on_led | 173 / 174 | 99.4% |
| stack_block | 163 / 181 | 90.1% |
| push_pink_block_right | 30 / 66 | 45.5% |
| push_red_block_left | 67 / 78 | 85.9% |
| lift_blue_block_table | 169 / 169 | 100.0% |
| place_in_drawer | 173 / 174 | 99.4% |
| move_slider_left | 241 / 241 | 100.0% |
| rotate_red_block_left | 61 / 62 | 98.4% |
| push_pink_block_left | 55 / 75 | 73.3% |
| push_red_block_right | 25 / 72 | 34.7% |
| lift_pink_block_drawer | 12 / 14 | 85.7% |
| rotate_pink_block_right | 65 / 71 | 91.5% |
| unstack_block | 62 / 62 | 100.0% |
| lift_blue_block_slider | 113 / 120 | 94.2% |
| rotate_pink_block_left | 50 / 55 | 90.9% |
| lift_red_block_drawer | 19 / 20 | 95.0% |
| push_blue_block_right | 17 / 69 | 24.6% |

## 实验记录 - 2026-08-10 (3)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft/checkpoints/step-008158-epoch-01-loss=0.3132.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_latact_traj_fullft`
  - Step: 008158
  - Epoch: 01
  - Loss: 0.3132
- **unnorm-key**: `calvin_abc2d_oe_latact`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.934

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 93.6% |
| 2 | 85.9% |
| 3 | 78.4% |
| 4 | 71.3% |
| 5 | 64.2% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 66 / 73 | 90.4% |
| move_slider_right | 270 / 270 | 100.0% |
| lift_red_block_slider | 114 / 119 | 95.8% |
| place_in_slider | 292 / 352 | 83.0% |
| turn_off_lightbulb | 140 / 140 | 100.0% |
| turn_off_led | 162 / 162 | 100.0% |
| push_into_drawer | 91 / 110 | 82.7% |
| lift_blue_block_drawer | 17 / 18 | 94.4% |
| lift_pink_block_slider | 123 / 130 | 94.6% |
| lift_pink_block_table | 160 / 165 | 97.0% |
| open_drawer | 315 / 334 | 94.3% |
| turn_on_lightbulb | 176 / 176 | 100.0% |
| push_blue_block_left | 61 / 68 | 89.7% |
| close_drawer | 199 / 200 | 99.5% |
| rotate_red_block_right | 72 / 73 | 98.6% |
| turn_on_led | 163 / 167 | 97.6% |
| stack_block | 121 / 179 | 67.6% |
| push_pink_block_right | 33 / 66 | 50.0% |
| push_red_block_left | 67 / 74 | 90.5% |
| lift_blue_block_table | 165 / 165 | 100.0% |
| rotate_blue_block_left | 61 / 65 | 93.8% |
| place_in_drawer | 171 / 173 | 98.8% |
| move_slider_left | 242 / 242 | 100.0% |
| rotate_red_block_left | 60 / 62 | 96.8% |
| push_pink_block_left | 68 / 74 | 91.9% |
| lift_blue_block_slider | 123 / 130 | 94.6% |
| push_red_block_right | 28 / 71 | 39.4% |
| lift_red_block_table | 165 / 166 | 99.4% |
| lift_pink_block_drawer | 13 / 14 | 92.9% |
| rotate_pink_block_right | 60 / 70 | 85.7% |
| unstack_block | 47 / 47 | 100.0% |
| rotate_pink_block_left | 53 / 55 | 96.4% |
| push_blue_block_right | 22 / 67 | 32.8% |
| lift_red_block_drawer | 14 / 15 | 93.3% |


## 实验记录 - 2026-08-11

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_baseline_rerun_fullft_h800/checkpoints/step-025116-epoch-03-loss=0.1014.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_baseline_rerun_fullft_h800`
  - Step: 025116
  - Epoch: 03
  - Loss: 0.1014
- **unnorm-key**: `calvin_abc2d_oe_baseline`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.962

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 95.2% |
| 2 | 87.3% |
| 3 | 78.6% |
| 4 | 71.4% |
| 5 | 63.7% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 73 / 73 | 100.0% |
| move_slider_right | 271 / 271 | 100.0% |
| lift_red_block_slider | 122 / 128 | 95.3% |
| place_in_slider | 278 / 354 | 78.5% |
| turn_off_lightbulb | 136 / 136 | 100.0% |
| turn_off_led | 160 / 160 | 100.0% |
| push_into_drawer | 99 / 123 | 80.5% |
| lift_blue_block_drawer | 19 / 20 | 95.0% |
| lift_pink_block_slider | 128 / 134 | 95.5% |
| open_drawer | 328 / 329 | 99.7% |
| rotate_red_block_right | 69 / 73 | 94.5% |
| lift_red_block_table | 170 / 171 | 99.4% |
| lift_pink_block_table | 164 / 167 | 98.2% |
| turn_on_lightbulb | 175 / 176 | 99.4% |
| rotate_blue_block_left | 66 / 67 | 98.5% |
| push_blue_block_left | 66 / 67 | 98.5% |
| close_drawer | 194 / 194 | 100.0% |
| turn_on_led | 162 / 164 | 98.8% |
| stack_block | 96 / 191 | 50.3% |
| push_red_block_left | 71 / 76 | 93.4% |
| lift_blue_block_table | 175 / 175 | 100.0% |
| place_in_drawer | 183 / 184 | 99.5% |
| move_slider_left | 240 / 240 | 100.0% |
| rotate_red_block_left | 62 / 63 | 98.4% |
| push_pink_block_left | 66 / 72 | 91.7% |
| lift_blue_block_slider | 113 / 126 | 89.7% |
| push_red_block_right | 34 / 72 | 47.2% |
| lift_pink_block_drawer | 15 / 16 | 93.8% |
| rotate_pink_block_right | 64 / 67 | 95.5% |
| rotate_pink_block_left | 54 / 54 | 100.0% |
| push_blue_block_right | 30 / 69 | 43.5% |
| push_pink_block_right | 27 / 61 | 44.3% |
| lift_red_block_drawer | 17 / 17 | 100.0% |
| unstack_block | 35 / 35 | 100.0% |

## 实验记录 - 2026-08-11 (2)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_baseline_rerun_fullft_h800/checkpoints/step-016744-epoch-02-loss=0.1051.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_baseline_rerun_fullft_h800`
  - Step: 016744
  - Epoch: 02
  - Loss: 0.1051
- **unnorm-key**: `calvin_abc2d_oe_baseline`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.9

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 93.9% |
| 2 | 85.3% |
| 3 | 77.9% |
| 4 | 70.2% |
| 5 | 62.7% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 74 / 75 | 98.7% |
| move_slider_right | 272 / 273 | 99.6% |
| turn_off_led | 155 / 156 | 99.4% |
| push_into_drawer | 96 / 113 | 85.0% |
| lift_blue_block_drawer | 18 / 18 | 100.0% |
| lift_pink_block_slider | 131 / 134 | 97.8% |
| place_in_slider | 277 / 345 | 80.3% |
| open_drawer | 331 / 331 | 100.0% |
| rotate_red_block_right | 70 / 73 | 95.9% |
| lift_red_block_table | 163 / 163 | 100.0% |
| lift_pink_block_table | 158 / 164 | 96.3% |
| move_slider_left | 237 / 237 | 100.0% |
| turn_on_lightbulb | 169 / 170 | 99.4% |
| rotate_blue_block_left | 61 / 65 | 93.8% |
| push_blue_block_left | 54 / 66 | 81.8% |
| close_drawer | 192 / 192 | 100.0% |
| lift_red_block_slider | 121 / 132 | 91.7% |
| turn_off_lightbulb | 136 / 136 | 100.0% |
| push_red_block_left | 68 / 76 | 89.5% |
| lift_blue_block_table | 167 / 167 | 100.0% |
| place_in_drawer | 172 / 174 | 98.9% |
| rotate_red_block_left | 59 / 63 | 93.7% |
| turn_on_led | 164 / 168 | 97.6% |
| stack_block | 106 / 187 | 56.7% |
| push_pink_block_left | 68 / 74 | 91.9% |
| lift_blue_block_slider | 119 / 128 | 93.0% |
| push_red_block_right | 27 / 72 | 37.5% |
| lift_pink_block_drawer | 11 / 13 | 84.6% |
| rotate_pink_block_right | 66 / 69 | 95.7% |
| rotate_pink_block_left | 53 / 54 | 98.1% |
| unstack_block | 35 / 35 | 100.0% |
| push_blue_block_right | 27 / 69 | 39.1% |
| push_pink_block_right | 29 / 67 | 43.3% |
| lift_red_block_drawer | 14 / 14 | 100.0% |

## 实验记录 - 2026-08-11 (3)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_baseline_rerun_fullft_h800/checkpoints/step-008372-epoch-01-loss=0.0976.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_baseline_rerun_fullft_h800`
  - Step: 008372
  - Epoch: 01
  - Loss: 0.0976
- **unnorm-key**: `calvin_abc2d_oe_baseline`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.837

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 95.7% |
| 2 | 87.1% |
| 3 | 76.5% |
| 4 | 66.3% |
| 5 | 58.1% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 71 / 73 | 97.3% |
| move_slider_right | 276 / 276 | 100.0% |
| lift_red_block_slider | 117 / 123 | 95.1% |
| place_in_slider | 249 / 349 | 71.3% |
| turn_off_lightbulb | 131 / 133 | 98.5% |
| turn_off_led | 158 / 160 | 98.8% |
| push_into_drawer | 96 / 116 | 82.8% |
| lift_blue_block_drawer | 17 / 19 | 89.5% |
| lift_pink_block_slider | 124 / 135 | 91.9% |
| open_drawer | 323 / 326 | 99.1% |
| rotate_red_block_right | 69 / 72 | 95.8% |
| lift_red_block_table | 169 / 172 | 98.3% |
| lift_pink_block_table | 154 / 164 | 93.9% |
| turn_on_lightbulb | 166 / 170 | 97.6% |
| rotate_blue_block_left | 66 / 66 | 100.0% |
| push_blue_block_left | 63 / 63 | 100.0% |
| close_drawer | 190 / 190 | 100.0% |
| turn_on_led | 158 / 160 | 98.8% |
| push_red_block_left | 73 / 75 | 97.3% |
| lift_blue_block_table | 167 / 171 | 97.7% |
| place_in_drawer | 173 / 173 | 100.0% |
| move_slider_left | 230 / 233 | 98.7% |
| rotate_red_block_left | 57 / 62 | 91.9% |
| stack_block | 86 / 187 | 46.0% |
| push_pink_block_left | 69 / 74 | 93.2% |
| lift_blue_block_slider | 119 / 130 | 91.5% |
| push_red_block_right | 35 / 70 | 50.0% |
| lift_pink_block_drawer | 12 / 14 | 85.7% |
| rotate_pink_block_right | 62 / 66 | 93.9% |
| rotate_pink_block_left | 50 / 54 | 92.6% |
| unstack_block | 31 / 32 | 96.9% |
| push_pink_block_right | 30 / 62 | 48.4% |
| push_blue_block_right | 29 / 67 | 43.3% |
| lift_red_block_drawer | 17 / 19 | 89.5% |

## 实验记录 - 2026-08-12

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_baseline_rerun_fullft/checkpoints/step-025116-epoch-03-loss=0.1322.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_baseline_rerun_fullft`
  - Step: 025116
  - Epoch: 03
  - Loss: 0.1322
- **unnorm-key**: `calvin_abc2d_oe_baseline`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.825

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 94.1% |
| 2 | 85.3% |
| 3 | 76.3% |
| 4 | 67.3% |
| 5 | 59.5% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 75 / 75 | 100.0% |
| move_slider_right | 267 / 267 | 100.0% |
| lift_red_block_slider | 116 / 126 | 92.1% |
| place_in_slider | 245 / 349 | 70.2% |
| turn_off_lightbulb | 136 / 136 | 100.0% |
| turn_off_led | 159 / 160 | 99.4% |
| push_into_drawer | 94 / 111 | 84.7% |
| lift_blue_block_drawer | 19 / 20 | 95.0% |
| lift_pink_block_slider | 122 / 131 | 93.1% |
| open_drawer | 322 / 323 | 99.7% |
| rotate_red_block_right | 70 / 72 | 97.2% |
| lift_red_block_table | 167 / 170 | 98.2% |
| lift_pink_block_table | 162 / 167 | 97.0% |
| turn_on_lightbulb | 166 / 167 | 99.4% |
| rotate_blue_block_left | 67 / 68 | 98.5% |
| push_blue_block_left | 63 / 64 | 98.4% |
| close_drawer | 185 / 185 | 100.0% |
| turn_on_led | 158 / 159 | 99.4% |
| stack_block | 96 / 180 | 53.3% |
| push_red_block_left | 69 / 75 | 92.0% |
| lift_blue_block_table | 165 / 167 | 98.8% |
| place_in_drawer | 168 / 170 | 98.8% |
| move_slider_left | 232 / 233 | 99.6% |
| rotate_red_block_left | 61 / 62 | 98.4% |
| lift_blue_block_slider | 117 / 128 | 91.4% |
| lift_pink_block_drawer | 13 / 15 | 86.7% |
| rotate_pink_block_right | 62 / 68 | 91.2% |
| push_red_block_right | 28 / 70 | 40.0% |
| rotate_pink_block_left | 53 / 54 | 98.1% |
| unstack_block | 37 / 38 | 97.4% |
| push_pink_block_left | 65 / 75 | 86.7% |
| push_pink_block_right | 24 / 61 | 39.3% |
| push_blue_block_right | 25 / 66 | 37.9% |
| lift_red_block_drawer | 17 / 18 | 94.4% |

## 实验记录 - 2026-08-12 (2)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_baseline_rerun_fullft/checkpoints/step-016744-epoch-02-loss=0.1157.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_baseline_rerun_fullft`
  - Step: 016744
  - Epoch: 02
  - Loss: 0.1157
- **unnorm-key**: `calvin_abc2d_oe_baseline`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.78

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 94.1% |
| 2 | 84.8% |
| 3 | 75.5% |
| 4 | 66.0% |
| 5 | 57.6% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 69 / 73 | 94.5% |
| move_slider_right | 266 / 266 | 100.0% |
| lift_red_block_slider | 121 / 127 | 95.3% |
| place_in_slider | 246 / 348 | 70.7% |
| turn_off_lightbulb | 129 / 131 | 98.5% |
| turn_off_led | 158 / 159 | 99.4% |
| lift_pink_block_slider | 120 / 129 | 93.0% |
| open_drawer | 323 / 323 | 100.0% |
| rotate_red_block_right | 66 / 71 | 93.0% |
| lift_red_block_table | 165 / 166 | 99.4% |
| lift_pink_block_table | 161 / 164 | 98.2% |
| turn_on_lightbulb | 162 / 162 | 100.0% |
| rotate_blue_block_left | 64 / 64 | 100.0% |
| push_blue_block_left | 61 / 66 | 92.4% |
| close_drawer | 190 / 191 | 99.5% |
| turn_on_led | 159 / 161 | 98.8% |
| stack_block | 92 / 186 | 49.5% |
| push_red_block_left | 61 / 77 | 79.2% |
| lift_blue_block_table | 163 / 163 | 100.0% |
| place_in_drawer | 169 / 170 | 99.4% |
| move_slider_left | 230 / 231 | 99.6% |
| rotate_red_block_left | 62 / 62 | 100.0% |
| push_pink_block_left | 65 / 73 | 89.0% |
| lift_blue_block_slider | 112 / 123 | 91.1% |
| lift_pink_block_drawer | 13 / 14 | 92.9% |
| rotate_pink_block_right | 67 / 69 | 97.1% |
| push_red_block_right | 34 / 72 | 47.2% |
| push_into_drawer | 91 / 114 | 79.8% |
| rotate_pink_block_left | 53 / 54 | 98.1% |
| unstack_block | 30 / 33 | 90.9% |
| push_pink_block_right | 22 / 59 | 37.3% |
| lift_blue_block_drawer | 17 / 18 | 94.4% |
| push_blue_block_right | 21 / 66 | 31.8% |
| lift_red_block_drawer | 18 / 19 | 94.7% |

## 实验记录 - 2026-08-12 (3)

### 配置来源
`scripts/serve.sh`

### 实验配置
- **模型 checkpoint**: `/liujinxin/zhaowei/CogACT/logs/calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_baseline_rerun_fullft/checkpoints/step-008372-epoch-01-loss=0.1672.pt`
  - Run 名称: `calvin_abc2d_oe_h10_8b_layerwise_flow_statedrop0.8_baseline_rerun_fullft`
  - Step: 008372
  - Epoch: 01
  - Loss: 0.1672
- **unnorm-key**: `calvin_abc2d_oe_baseline`
- **future-action-window-size**: 9
- **推理服务**: `serve/flask_server.py`
- **GPU 部署**: 8 卡（0-7），每卡启动一个服务进程，端口从 9002 起递增

### 实验结果

**Results for Epoch -1:**

Average successful sequence length: 3.799

Success rates for i instructions in a row:

| i | SR |
|---|---|
| 1 | 96.9% |
| 2 | 87.7% |
| 3 | 76.8% |
| 4 | 64.2% |
| 5 | 54.3% |

**各任务成功率明细:**

| 任务 | 成功/总数 | SR |
|---|---|---|
| rotate_blue_block_right | 73 / 75 | 97.3% |
| move_slider_right | 273 / 273 | 100.0% |
| turn_off_led | 161 / 161 | 100.0% |
| lift_pink_block_slider | 120 / 132 | 90.9% |
| place_in_slider | 221 / 348 | 63.5% |
| open_drawer | 319 / 319 | 100.0% |
| rotate_red_block_right | 69 / 71 | 97.2% |
| lift_red_block_table | 172 / 177 | 97.2% |
| lift_pink_block_table | 164 / 170 | 96.5% |
| turn_on_lightbulb | 161 / 161 | 100.0% |
| rotate_blue_block_left | 65 / 65 | 100.0% |
| push_blue_block_left | 60 / 62 | 96.8% |
| close_drawer | 187 / 188 | 99.5% |
| lift_red_block_slider | 107 / 124 | 86.3% |
| turn_off_lightbulb | 130 / 136 | 95.6% |
| turn_on_led | 159 / 160 | 99.4% |
| push_pink_block_right | 36 / 61 | 59.0% |
| push_red_block_left | 68 / 73 | 93.2% |
| lift_blue_block_table | 180 / 183 | 98.4% |
| place_in_drawer | 170 / 174 | 97.7% |
| move_slider_left | 210 / 227 | 92.5% |
| rotate_red_block_left | 57 / 61 | 93.4% |
| push_pink_block_left | 70 / 75 | 93.3% |
| push_red_block_right | 45 / 68 | 66.2% |
| lift_pink_block_drawer | 14 / 15 | 93.3% |
| rotate_pink_block_right | 67 / 69 | 97.1% |
| lift_blue_block_slider | 104 / 128 | 81.2% |
| push_blue_block_right | 52 / 67 | 77.6% |
| push_into_drawer | 91 / 122 | 74.6% |
| rotate_pink_block_left | 48 / 51 | 94.1% |
| stack_block | 77 / 190 | 40.5% |
| unstack_block | 32 / 32 | 100.0% |
| lift_blue_block_drawer | 19 / 19 | 100.0% |
| lift_red_block_drawer | 18 / 19 | 94.7% |
