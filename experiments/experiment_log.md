# 实验测试记录

> 记录每次运行 `scripts/serve.sh` 时的实验配置与结果。

---

## 实验列表

| # | 日期 | 模型 checkpoint | unnorm_key | action_model_type | future_action_window_size | cfg_scale | DDIM steps | 结果 |
|---|------|----------------|------------|-------------------|--------------------------|-----------|------------|------|
| [Exp-001](#exp-001) | 2026-04-19 | step-016744-epoch-02-loss=0.0619 | calvin_abc2d_oe_baseline | DiT-B | 15 | 1.5 | 10 | Avg seq len 3.578 |
| [Exp-002](#exp-002) | 2026-04-19 | step-008372-epoch-01-loss=0.1021 | calvin_abc2d_oe_baseline | DiT-B | 15 | 1.5 | 10 | Avg seq len 3.495 |
| [Exp-003](#exp-003) | 2026-04-20 | step-016744-epoch-02-loss=0.0633 (calvin_abc2d_oe) | calvin_abc2d_oe | DiT-B | 15 | 1.5 | 10 | Avg seq len 3.613 |
| [Exp-004](#exp-004) | 2026-04-20 | step-008372-epoch-01-loss=0.0783 (calvin_abc2d_oe) | calvin_abc2d_oe | DiT-B | 15 | 1.5 | 10 | Avg seq len 3.505 |
| [Exp-005](#exp-005) | 2026-04-21 | step-016744-epoch-02-loss=0.0683 (calvin_abc2d_oe_baseline_h10) | calvin_abc2d_oe_baseline | DiT-B | 9 | 1.5 | 10 | Avg seq len 3.628 |
| [Exp-006](#exp-006) | 2026-04-21 | step-016744-epoch-02-loss=0.0623 (calvin_abc2d_oe_baseline_h10_8b) | calvin_abc2d_oe_baseline | DiT-B | 9 | 1.5 | 10 | Avg seq len 3.663 |
| [Exp-007](#exp-007) | 2026-04-21 | step-025116-epoch-03-loss=0.0484 (calvin_abc2d_oe_baseline_h10_8b) | calvin_abc2d_oe_baseline | DiT-B | 9 | 1.5 | 10 | Avg seq len 3.626 |
| [Exp-008](#exp-008) | 2026-04-21 | step-008372-epoch-01-loss=0.1109 (calvin_abc2d_oe_baseline_h10_8b) | calvin_abc2d_oe_baseline | DiT-B | 9 | 1.5 | 10 | Avg seq len 3.271 |
| [Exp-009](#exp-009) | 2026-04-21 | step-016744-epoch-02-loss=0.0627 (calvin_abc2d_oe_h10_8b) | calvin_abc2d_oe | DiT-B | 9 | 1.5 | 10 | Avg seq len 3.52 |
| [Exp-010](#exp-010) | 2026-04-24 | step-025116-epoch-03-loss=0.0413 (calvin_abc2d_oe_h10_8b) | calvin_abc2d_oe | DiT-B | 9 | 1.5 | 10 | Avg seq len 3.721 |
| [Exp-011](#exp-011) | 2026-04-24 | step-008372-epoch-01-loss=0.1151 (calvin_abc2d_oe_h10_8b) | calvin_abc2d_oe | DiT-B | 9 | 1.5 | 10 | Avg seq len 3.001 |
| [Exp-012](#exp-012) | 2026-04-27 | step-016744-epoch-02-loss=0.0585 (calvin_abc2d_oe_h10_layerwise) | calvin_abc2d_oe | DiT-B | 9 | 1.5 | 10 | Avg seq len 3.526 |
| [Exp-013](#exp-013) | 2026-04-28 | step-025116-epoch-03-loss=0.0451 (calvin_abc2d_oe_h10_layerwise) | calvin_abc2d_oe | DiT-B | 9 | 1.5 | 10 | Avg seq len 3.68 |

---

## Exp-001

**日期：** 2026-04-19

### serve.sh 配置

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# 每张 GPU 启动一个 flask server，端口从 9002 依次递增
# GPU 0 → port 9002, GPU 1 → port 9003, ..., GPU 7 → port 9009
```

### flask_server.py 参数

| 参数 | 值 | 说明 |
|------|----|------|
| `--model-path` | `/zhaowei/workspace/CogACT/logs/calvin_abc2d_oe_baseline/checkpoints/step-016744-epoch-02-loss=0.0619.pt` | 模型 checkpoint 路径 |
| `--load-for-training` | `False`（未启用） | 是否以训练模式加载模型 |
| `--action-model-type` | `DiT-B` | 动作模型类型 |
| `--future-action-window-size` | `15` | 预测未来动作窗口大小（输出 16 步，index 0~15） |
| `--port` | `9002`（GPU 0），依次 +1 | Flask 服务端口 |
| `--debug` | `False` | 是否保存调试图片 |

### 推理参数（硬编码于 flask_server.py）

| 参数 | 值 |
|------|----|
| `unnorm_key` | `calvin_abc2d_oe_baseline` |
| `cfg_scale` | `1.5` |
| `use_ddim` | `True` |
| `num_ddim_steps` | `10` |

### 实验结果

**Average successful sequence length:** 3.578

**Success rates for i instructions in a row:**

| 连续指令数 i | Success Rate |
|--------------|--------------|
| 1 | 91.7% |
| 2 | 80.5% |
| 3 | 70.5% |
| 4 | 62.3% |
| 5 | 52.8% |

**Per-task success rates:**

| 任务 | 成功 / 总数 | Success Rate |
|------|-------------|--------------|
| `rotate_blue_block_right` | 63 / 70 | 90.0% |
| `move_slider_right` | 260 / 260 | 100.0% |
| `lift_red_block_slider` | 103 / 120 | 85.8% |
| `place_in_slider` | 244 / 315 | 77.5% |
| `turn_off_lightbulb` | 128 / 129 | 99.2% |
| `turn_off_led` | 148 / 149 | 99.3% |
| `push_into_drawer` | 88 / 104 | 84.6% |
| `lift_blue_block_drawer` | 17 / 17 | 100.0% |
| `lift_pink_block_slider` | 112 / 124 | 90.3% |
| `open_drawer` | 304 / 305 | 99.7% |
| `rotate_red_block_right` | 63 / 72 | 87.5% |
| `lift_red_block_table` | 155 / 158 | 98.1% |
| `lift_pink_block_table` | 137 / 146 | 93.8% |
| `turn_on_lightbulb` | 164 / 164 | 100.0% |
| `rotate_blue_block_left` | 62 / 64 | 96.9% |
| `push_blue_block_left` | 57 / 66 | 86.4% |
| `close_drawer` | 182 / 183 | 99.5% |
| `turn_on_led` | 154 / 158 | 97.5% |
| `lift_blue_block_table` | 154 / 154 | 100.0% |
| `place_in_drawer` | 157 / 160 | 98.1% |
| `move_slider_left` | 215 / 233 | 92.3% |
| `rotate_red_block_left` | 61 / 63 | 96.8% |
| `stack_block` | 98 / 170 | 57.6% |
| `push_pink_block_left` | 49 / 71 | 69.0% |
| `push_red_block_right` | 28 / 70 | 40.0% |
| `lift_pink_block_drawer` | 12 / 12 | 100.0% |
| `rotate_pink_block_right` | 55 / 69 | 79.7% |
| `unstack_block` | 46 / 47 | 97.9% |
| `lift_blue_block_slider` | 103 / 119 | 86.6% |
| `rotate_pink_block_left` | 48 / 55 | 87.3% |
| `push_pink_block_right` | 28 / 62 | 45.2% |
| `push_red_block_left` | 48 / 76 | 63.2% |
| `push_blue_block_right` | 20 / 69 | 29.0% |
| `lift_red_block_drawer` | 15 / 16 | 93.8% |

---

## Exp-002

**日期：** 2026-04-19

### serve.sh 配置

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# 每张 GPU 启动一个 flask server，端口从 9002 依次递增
# GPU 0 → port 9002, GPU 1 → port 9003, ..., GPU 7 → port 9009
```

### flask_server.py 参数

| 参数 | 值 | 说明 |
|------|----|------|
| `--model-path` | `/zhaowei/workspace/CogACT/logs/calvin_abc2d_oe_baseline/checkpoints/step-008372-epoch-01-loss=0.1021.pt` | 模型 checkpoint 路径（epoch-01，较 Exp-001 更早期） |
| `--load-for-training` | `False`（未启用） | 是否以训练模式加载模型 |
| `--action-model-type` | `DiT-B` | 动作模型类型 |
| `--future-action-window-size` | `15` | 预测未来动作窗口大小（输出 16 步，index 0~15） |
| `--port` | `9002`（GPU 0），依次 +1 | Flask 服务端口 |
| `--debug` | `False` | 是否保存调试图片 |

### 推理参数（硬编码于 flask_server.py）

| 参数 | 值 |
|------|----|
| `unnorm_key` | `calvin_abc2d_oe_baseline` |
| `cfg_scale` | `1.5` |
| `use_ddim` | `True` |
| `num_ddim_steps` | `10` |

### 与 Exp-001 的差异

| 项目 | Exp-001 | Exp-002 |
|------|---------|---------|
| checkpoint | step-016744-epoch-02-loss=0.0619 | step-008372-epoch-01-loss=0.1021 |
| 训练步数 | 16744 | 8372 |
| epoch | 2 | 1 |
| loss | 0.0619 | 0.1021 |

### 实验结果

**Average successful sequence length:** 3.495

**Success rates for i instructions in a row:**

| 连续指令数 i | Success Rate |
|--------------|--------------|
| 1 | 90.8% |
| 2 | 79.3% |
| 3 | 68.9% |
| 4 | 59.9% |
| 5 | 50.6% |

**Per-task success rates:**

| 任务 | 成功 / 总数 | Success Rate |
|------|-------------|--------------|
| `rotate_blue_block_right` | 64 / 73 | 87.7% |
| `move_slider_right` | 258 / 259 | 99.6% |
| `turn_off_led` | 144 / 144 | 100.0% |
| `push_into_drawer` | 90 / 107 | 84.1% |
| `lift_blue_block_drawer` | 14 / 15 | 93.3% |
| `place_in_slider` | 242 / 310 | 78.1% |
| `close_drawer` | 176 / 177 | 99.4% |
| `lift_pink_block_slider` | 105 / 125 | 84.0% |
| `open_drawer` | 308 / 311 | 99.0% |
| `lift_pink_block_table` | 150 / 155 | 96.8% |
| `turn_on_lightbulb` | 157 / 158 | 99.4% |
| `rotate_blue_block_left` | 58 / 64 | 90.6% |
| `push_blue_block_left` | 58 / 65 | 89.2% |
| `lift_red_block_slider` | 96 / 122 | 78.7% |
| `turn_off_lightbulb` | 123 / 125 | 98.4% |
| `rotate_red_block_right` | 62 / 75 | 82.7% |
| `turn_on_led` | 159 / 161 | 98.8% |
| `push_red_block_left` | 50 / 75 | 66.7% |
| `lift_blue_block_table` | 138 / 142 | 97.2% |
| `place_in_drawer` | 143 / 144 | 99.3% |
| `move_slider_left` | 210 / 229 | 91.7% |
| `rotate_red_block_left` | 59 / 63 | 93.7% |
| `stack_block` | 94 / 165 | 57.0% |
| `push_pink_block_left` | 66 / 77 | 85.7% |
| `lift_blue_block_slider` | 100 / 122 | 82.0% |
| `lift_red_block_table` | 143 / 147 | 97.3% |
| `lift_pink_block_drawer` | 14 / 14 | 100.0% |
| `rotate_pink_block_right` | 49 / 66 | 74.2% |
| `unstack_block` | 38 / 38 | 100.0% |
| `push_red_block_right` | 20 / 70 | 28.6% |
| `rotate_pink_block_left` | 50 / 53 | 94.3% |
| `push_pink_block_right` | 29 / 60 | 48.3% |
| `lift_red_block_drawer` | 15 / 15 | 100.0% |
| `push_blue_block_right` | 13 / 63 | 20.6% |

---

## Exp-003

**日期：** 2026-04-20

### serve.sh 配置

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# 每张 GPU 启动一个 flask server，端口从 9002 依次递增
# GPU 0 → port 9002, GPU 1 → port 9003, ..., GPU 7 → port 9009
```

### flask_server.py 参数

| 参数 | 值 | 说明 |
|------|----|------|
| `--model-path` | `/zhaowei/workspace/CogACT/logs/calvin_abc2d_oe/checkpoints/step-016744-epoch-02-loss=0.0633.pt` | 模型 checkpoint 路径（新日志目录 calvin_abc2d_oe） |
| `--unnorm-key` | `calvin_abc2d_oe` | 动作归一化统计键（显式传入，区别于前两次实验） |
| `--load-for-training` | `False`（未启用） | 是否以训练模式加载模型 |
| `--action-model-type` | `DiT-B` | 动作模型类型 |
| `--future-action-window-size` | `15` | 预测未来动作窗口大小（输出 16 步，index 0~15） |
| `--port` | `9002`（GPU 0），依次 +1 | Flask 服务端口 |
| `--debug` | `False` | 是否保存调试图片 |

### 推理参数（硬编码于 flask_server.py）

| 参数 | 值 |
|------|----|
| `unnorm_key` | `calvin_abc2d_oe`（由 `--unnorm-key` 传入） |
| `cfg_scale` | `1.5` |
| `use_ddim` | `True` |
| `num_ddim_steps` | `10` |

### 与 Exp-001 的差异

| 项目 | Exp-001 | Exp-003 |
|------|---------|---------|
| 日志目录 | `calvin_abc2d_oe_baseline` | `calvin_abc2d_oe` |
| checkpoint | step-016744-epoch-02-loss=0.0619 | step-016744-epoch-02-loss=0.0633 |
| 训练步数 | 16744 | 16744 |
| epoch | 2 | 2 |
| loss | 0.0619 | 0.0633 |
| unnorm_key | `calvin_abc2d_oe_baseline` | `calvin_abc2d_oe` |

### 实验结果

**Average successful sequence length:** 3.613

**Success rates for i instructions in a row:**

| 连续指令数 i | Success Rate |
|--------------|--------------|
| 1 | 91.3% |
| 2 | 81.1% |
| 3 | 70.7% |
| 4 | 62.9% |
| 5 | 55.3% |

**Per-task success rates:**

| 任务 | 成功 / 总数 | Success Rate |
|------|-------------|--------------|
| `rotate_blue_block_right` | 61 / 74 | 82.4% |
| `move_slider_right` | 264 / 268 | 98.5% |
| `turn_off_led` | 153 / 154 | 99.4% |
| `push_into_drawer` | 83 / 100 | 83.0% |
| `lift_blue_block_drawer` | 14 / 15 | 93.3% |
| `lift_pink_block_slider` | 100 / 125 | 80.0% |
| `place_in_slider` | 242 / 301 | 80.4% |
| `open_drawer` | 315 / 329 | 95.7% |
| `rotate_red_block_right` | 67 / 73 | 91.8% |
| `lift_red_block_table` | 150 / 158 | 94.9% |
| `lift_pink_block_table` | 145 / 149 | 97.3% |
| `turn_on_lightbulb` | 158 / 160 | 98.8% |
| `rotate_blue_block_left` | 58 / 65 | 89.2% |
| `push_blue_block_left` | 49 / 67 | 73.1% |
| `close_drawer` | 176 / 178 | 98.9% |
| `lift_red_block_slider` | 90 / 119 | 75.6% |
| `turn_off_lightbulb` | 135 / 139 | 97.1% |
| `turn_on_led` | 154 / 156 | 98.7% |
| `stack_block` | 143 / 163 | 87.7% |
| `push_red_block_left` | 55 / 73 | 75.3% |
| `lift_blue_block_table` | 151 / 156 | 96.8% |
| `place_in_drawer` | 142 / 145 | 97.9% |
| `move_slider_left` | 225 / 230 | 97.8% |
| `rotate_red_block_left` | 55 / 65 | 84.6% |
| `push_pink_block_left` | 60 / 73 | 82.2% |
| `lift_blue_block_slider` | 95 / 122 | 77.9% |
| `push_red_block_right` | 31 / 69 | 44.9% |
| `unstack_block` | 57 / 59 | 96.6% |
| `rotate_pink_block_left` | 50 / 54 | 92.6% |
| `rotate_pink_block_right` | 55 / 70 | 78.6% |
| `lift_pink_block_drawer` | 11 / 11 | 100.0% |
| `push_pink_block_right` | 28 / 62 | 45.2% |
| `push_blue_block_right` | 31 / 66 | 47.0% |
| `lift_red_block_drawer` | 10 / 12 | 83.3% |

---

## Exp-004

**日期：** 2026-04-20

### serve.sh 配置

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# 每张 GPU 启动一个 flask server，端口从 9002 依次递增
# GPU 0 → port 9002, GPU 1 → port 9003, ..., GPU 7 → port 9009
```

### flask_server.py 参数

| 参数 | 值 | 说明 |
|------|----|------|
| `--model-path` | `/zhaowei/workspace/CogACT/logs/calvin_abc2d_oe/checkpoints/step-008372-epoch-01-loss=0.0783.pt` | 模型 checkpoint 路径（epoch-01，较 Exp-003 更早期） |
| `--unnorm-key` | `calvin_abc2d_oe` | 动作归一化统计键 |
| `--load-for-training` | `False`（未启用） | 是否以训练模式加载模型 |
| `--action-model-type` | `DiT-B` | 动作模型类型 |
| `--future-action-window-size` | `15` | 预测未来动作窗口大小（输出 16 步，index 0~15） |
| `--port` | `9002`（GPU 0），依次 +1 | Flask 服务端口 |
| `--debug` | `False` | 是否保存调试图片 |

### 推理参数（硬编码于 flask_server.py）

| 参数 | 值 |
|------|----|
| `unnorm_key` | `calvin_abc2d_oe`（由 `--unnorm-key` 传入） |
| `cfg_scale` | `1.5` |
| `use_ddim` | `True` |
| `num_ddim_steps` | `10` |

### 与 Exp-003 的差异

| 项目 | Exp-003 | Exp-004 |
|------|---------|---------|
| checkpoint | step-016744-epoch-02-loss=0.0633 | step-008372-epoch-01-loss=0.0783 |
| 训练步数 | 16744 | 8372 |
| epoch | 2 | 1 |
| loss | 0.0633 | 0.0783 |

### 实验结果

**Average successful sequence length:** 3.505

**Success rates for i instructions in a row:**

| 连续指令数 i | Success Rate |
|--------------|--------------|
| 1 | 90.5% |
| 2 | 79.1% |
| 3 | 69.0% |
| 4 | 60.2% |
| 5 | 51.7% |

**Per-task success rates:**

| 任务 | 成功 / 总数 | Success Rate |
|------|-------------|--------------|
| `rotate_blue_block_right` | 63 / 69 | 91.3% |
| `move_slider_right` | 251 / 254 | 98.8% |
| `lift_red_block_slider` | 82 / 118 | 69.5% |
| `place_in_slider` | 237 / 299 | 79.3% |
| `turn_off_lightbulb` | 123 / 133 | 92.5% |
| `turn_off_led` | 148 / 149 | 99.3% |
| `push_into_drawer` | 82 / 100 | 82.0% |
| `lift_blue_block_drawer` | 16 / 18 | 88.9% |
| `lift_pink_block_table` | 140 / 143 | 97.9% |
| `open_drawer` | 296 / 321 | 92.2% |
| `turn_on_lightbulb` | 165 / 165 | 100.0% |
| `rotate_blue_block_left` | 55 / 63 | 87.3% |
| `push_blue_block_left` | 46 / 65 | 70.8% |
| `close_drawer` | 179 / 179 | 100.0% |
| `rotate_red_block_right` | 59 / 69 | 85.5% |
| `turn_on_led` | 160 / 161 | 99.4% |
| `push_red_block_left` | 61 / 78 | 78.2% |
| `lift_blue_block_table` | 148 / 150 | 98.7% |
| `place_in_drawer` | 143 / 147 | 97.3% |
| `lift_pink_block_slider` | 100 / 125 | 80.0% |
| `move_slider_left` | 197 / 215 | 91.6% |
| `rotate_red_block_left` | 57 / 62 | 91.9% |
| `stack_block` | 127 / 162 | 78.4% |
| `push_pink_block_left` | 63 / 72 | 87.5% |
| `lift_blue_block_slider` | 97 / 119 | 81.5% |
| `lift_red_block_table` | 146 / 151 | 96.7% |
| `lift_pink_block_drawer` | 10 / 11 | 90.9% |
| `rotate_pink_block_right` | 53 / 66 | 80.3% |
| `unstack_block` | 53 / 53 | 100.0% |
| `push_red_block_right` | 31 / 69 | 44.9% |
| `rotate_pink_block_left` | 44 / 54 | 81.5% |
| `push_pink_block_right` | 28 / 63 | 44.4% |
| `push_blue_block_right` | 30 / 68 | 44.1% |
| `lift_red_block_drawer` | 15 / 17 | 88.2% |

---

## Exp-005

**日期：** 2026-04-21

### serve.sh 配置

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# 每张 GPU 启动一个 flask server，端口从 9002 依次递增
# GPU 0 → port 9002, GPU 1 → port 9003, ..., GPU 7 → port 9009
```

### flask_server.py 参数

| 参数 | 值 | 说明 |
|------|----|------|
| `--model-path` | `/zhaowei/workspace/CogACT/logs/calvin_abc2d_oe_baseline_h10/checkpoints/step-016744-epoch-02-loss=0.0683.pt` | 模型 checkpoint 路径（新日志目录 calvin_abc2d_oe_baseline_h10） |
| `--unnorm-key` | `calvin_abc2d_oe_baseline` | 动作归一化统计键 |
| `--load-for-training` | `False`（未启用） | 是否以训练模式加载模型 |
| `--action-model-type` | `DiT-B` | 动作模型类型 |
| `--future-action-window-size` | `9` | 预测未来动作窗口大小（输出 10 步，index 0~9） |
| `--port` | `9002`（GPU 0），依次 +1 | Flask 服务端口 |
| `--debug` | `False` | 是否保存调试图片 |

### 推理参数（硬编码于 flask_server.py）

| 参数 | 值 |
|------|----|
| `unnorm_key` | `calvin_abc2d_oe_baseline`（由 `--unnorm-key` 传入） |
| `cfg_scale` | `1.5` |
| `use_ddim` | `True` |
| `num_ddim_steps` | `10` |

### 与 Exp-001 的差异

| 项目 | Exp-001 | Exp-005 |
|------|---------|---------|
| 日志目录 | `calvin_abc2d_oe_baseline` | `calvin_abc2d_oe_baseline_h10` |
| checkpoint | step-016744-epoch-02-loss=0.0619 | step-016744-epoch-02-loss=0.0683 |
| loss | 0.0619 | 0.0683 |
| unnorm_key | `calvin_abc2d_oe_baseline`（硬编码） | `calvin_abc2d_oe_baseline`（显式传入） |
| future_action_window_size | 15 | 9 |

### 实验结果

**Average successful sequence length:** 3.628

**Success rates for i instructions in a row:**

| 连续指令数 i | Success Rate |
|--------------|--------------|
| 1 | 92.6% |
| 2 | 82.8% |
| 3 | 71.1% |
| 4 | 62.2% |
| 5 | 54.1% |

**Per-task success rates:**

| 任务 | 成功 / 总数 | Success Rate |
|------|-------------|--------------|
| `rotate_blue_block_right` | 72 / 73 | 98.6% |
| `move_slider_right` | 247 / 248 | 99.6% |
| `lift_red_block_slider` | 112 / 121 | 92.6% |
| `place_in_slider` | 201 / 331 | 60.7% |
| `turn_off_lightbulb` | 130 / 131 | 99.2% |
| `turn_off_led` | 143 / 144 | 99.3% |
| `push_into_drawer` | 87 / 103 | 84.5% |
| `lift_blue_block_drawer` | 16 / 17 | 94.1% |
| `lift_pink_block_slider` | 113 / 125 | 90.4% |
| `open_drawer` | 316 / 316 | 100.0% |
| `rotate_red_block_right` | 68 / 71 | 95.8% |
| `lift_red_block_table` | 153 / 156 | 98.1% |
| `lift_pink_block_table` | 148 / 149 | 99.3% |
| `move_slider_left` | 221 / 225 | 98.2% |
| `turn_on_lightbulb` | 158 / 159 | 99.4% |
| `rotate_blue_block_left` | 63 / 67 | 94.0% |
| `push_blue_block_left` | 49 / 68 | 72.1% |
| `close_drawer` | 187 / 189 | 98.9% |
| `turn_on_led` | 156 / 159 | 98.1% |
| `stack_block` | 134 / 175 | 76.6% |
| `push_red_block_left` | 47 / 78 | 60.3% |
| `lift_blue_block_table` | 163 / 165 | 98.8% |
| `place_in_drawer` | 157 / 158 | 99.4% |
| `rotate_red_block_left` | 59 / 60 | 98.3% |
| `push_red_block_right` | 29 / 69 | 42.0% |
| `push_pink_block_left` | 49 / 73 | 67.1% |
| `lift_pink_block_drawer` | 12 / 12 | 100.0% |
| `rotate_pink_block_right` | 58 / 70 | 82.9% |
| `unstack_block` | 54 / 56 | 96.4% |
| `lift_blue_block_slider` | 102 / 114 | 89.5% |
| `rotate_pink_block_left` | 53 / 55 | 96.4% |
| `push_pink_block_right` | 31 / 65 | 47.7% |
| `push_blue_block_right` | 23 / 68 | 33.8% |
| `lift_red_block_drawer` | 17 / 17 | 100.0% |

---

## Exp-006

**日期：** 2026-04-21

### serve.sh 配置

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# 每张 GPU 启动一个 flask server，端口从 9002 依次递增
# GPU 0 → port 9002, GPU 1 → port 9003, ..., GPU 7 → port 9009
```

### flask_server.py 参数

| 参数 | 值 | 说明 |
|------|----|------|
| `--model-path` | `/zhaowei/workspace/CogACT/logs/calvin_abc2d_oe_baseline_h10_8b/checkpoints/step-016744-epoch-02-loss=0.0623.pt` | 模型 checkpoint 路径（8B 模型，目录 calvin_abc2d_oe_baseline_h10_8b） |
| `--unnorm-key` | `calvin_abc2d_oe_baseline` | 动作归一化统计键 |
| `--load-for-training` | `False`（未启用） | 是否以训练模式加载模型 |
| `--action-model-type` | `DiT-B` | 动作模型类型 |
| `--future-action-window-size` | `9` | 预测未来动作窗口大小（输出 10 步，index 0~9） |
| `--port` | `9002`（GPU 0），依次 +1 | Flask 服务端口 |
| `--debug` | `False` | 是否保存调试图片 |

### 推理参数（硬编码于 flask_server.py）

| 参数 | 值 |
|------|----|
| `unnorm_key` | `calvin_abc2d_oe_baseline`（由 `--unnorm-key` 传入） |
| `cfg_scale` | `1.5` |
| `use_ddim` | `True` |
| `num_ddim_steps` | `10` |

### 与 Exp-005 的差异

| 项目 | Exp-005 | Exp-006 |
|------|---------|---------|
| 日志目录 | `calvin_abc2d_oe_baseline_h10` | `calvin_abc2d_oe_baseline_h10_8b` |
| checkpoint | step-016744-epoch-02-loss=0.0683 | step-016744-epoch-02-loss=0.0623 |
| loss | 0.0683 | 0.0623 |

### 实验结果

**Average successful sequence length:** 3.663

**Success rates for i instructions in a row:**

| 连续指令数 i | Success Rate |
|--------------|--------------|
| 1 | 91.9% |
| 2 | 81.6% |
| 3 | 72.3% |
| 4 | 64.6% |
| 5 | 55.9% |

**Per-task success rates:**

| 任务 | 成功 / 总数 | Success Rate |
|------|-------------|--------------|
| `rotate_blue_block_right` | 75 / 78 | 96.2% |
| `move_slider_right` | 262 / 264 | 99.2% |
| `lift_red_block_slider` | 95 / 115 | 82.6% |
| `place_in_slider` | 246 / 320 | 76.9% |
| `turn_off_lightbulb` | 132 / 134 | 98.5% |
| `turn_off_led` | 151 / 151 | 100.0% |
| `push_into_drawer` | 91 / 110 | 82.7% |
| `lift_blue_block_drawer` | 17 / 17 | 100.0% |
| `close_drawer` | 176 / 176 | 100.0% |
| `lift_pink_block_slider` | 114 / 130 | 87.7% |
| `open_drawer` | 323 / 324 | 99.7% |
| `lift_pink_block_table` | 147 / 152 | 96.7% |
| `move_slider_left` | 230 / 230 | 100.0% |
| `turn_on_lightbulb` | 167 / 168 | 99.4% |
| `rotate_blue_block_left` | 63 / 64 | 98.4% |
| `push_blue_block_left` | 51 / 63 | 81.0% |
| `rotate_red_block_right` | 64 / 72 | 88.9% |
| `turn_on_led` | 162 / 165 | 98.2% |
| `stack_block` | 124 / 175 | 70.9% |
| `lift_blue_block_table` | 154 / 157 | 98.1% |
| `place_in_drawer` | 145 / 152 | 95.4% |
| `rotate_red_block_left` | 59 / 59 | 100.0% |
| `push_pink_block_left` | 63 / 75 | 84.0% |
| `push_red_block_left` | 60 / 76 | 78.9% |
| `lift_blue_block_slider` | 97 / 113 | 85.8% |
| `push_red_block_right` | 27 / 71 | 38.0% |
| `lift_red_block_table` | 152 / 155 | 98.1% |
| `lift_pink_block_drawer` | 14 / 15 | 93.3% |
| `rotate_pink_block_right` | 38 / 70 | 54.3% |
| `unstack_block` | 51 / 51 | 100.0% |
| `rotate_pink_block_left` | 52 / 55 | 94.5% |
| `push_pink_block_right` | 25 / 63 | 39.7% |
| `lift_red_block_drawer` | 14 / 15 | 93.3% |
| `push_blue_block_right` | 22 / 69 | 31.9% |

---

## Exp-007

**日期：** 2026-04-21

### serve.sh 配置

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# 每张 GPU 启动一个 flask server，端口从 9002 依次递增
# GPU 0 → port 9002, GPU 1 → port 9003, ..., GPU 7 → port 9009
```

### flask_server.py 参数

| 参数 | 值 | 说明 |
|------|----|------|
| `--model-path` | `/zhaowei/workspace/CogACT/logs/calvin_abc2d_oe_baseline_h10_8b/checkpoints/step-025116-epoch-03-loss=0.0484.pt` | 模型 checkpoint 路径（epoch-03，较 Exp-006 训练更充分） |
| `--unnorm-key` | `calvin_abc2d_oe_baseline` | 动作归一化统计键 |
| `--load-for-training` | `False`（未启用） | 是否以训练模式加载模型 |
| `--action-model-type` | `DiT-B` | 动作模型类型 |
| `--future-action-window-size` | `9` | 预测未来动作窗口大小（输出 10 步，index 0~9） |
| `--port` | `9002`（GPU 0），依次 +1 | Flask 服务端口 |
| `--debug` | `False` | 是否保存调试图片 |

### 推理参数（硬编码于 flask_server.py）

| 参数 | 值 |
|------|----|
| `unnorm_key` | `calvin_abc2d_oe_baseline`（由 `--unnorm-key` 传入） |
| `cfg_scale` | `1.5` |
| `use_ddim` | `True` |
| `num_ddim_steps` | `10` |

### 与 Exp-006 的差异

| 项目 | Exp-006 | Exp-007 |
|------|---------|---------|
| checkpoint | step-016744-epoch-02-loss=0.0623 | step-025116-epoch-03-loss=0.0484 |
| 训练步数 | 16744 | 25116 |
| epoch | 2 | 3 |
| loss | 0.0623 | 0.0484 |

### 实验结果

**Average successful sequence length:** 3.626

**Success rates for i instructions in a row:**

| 连续指令数 i | Success Rate |
|--------------|--------------|
| 1 | 92.0% |
| 2 | 81.4% |
| 3 | 70.7% |
| 4 | 63.2% |
| 5 | 55.3% |

**Per-task success rates:**

| 任务 | 成功 / 总数 | Success Rate |
|------|-------------|--------------|
| `rotate_blue_block_right` | 73 / 74 | 98.6% |
| `move_slider_right` | 254 / 254 | 100.0% |
| `turn_off_led` | 147 / 148 | 99.3% |
| `push_into_drawer` | 89 / 107 | 83.2% |
| `lift_blue_block_drawer` | 18 / 18 | 100.0% |
| `lift_pink_block_slider` | 113 / 128 | 88.3% |
| `place_in_slider` | 232 / 318 | 73.0% |
| `open_drawer` | 327 / 327 | 100.0% |
| `rotate_red_block_right` | 65 / 73 | 89.0% |
| `lift_red_block_table` | 153 / 156 | 98.1% |
| `lift_pink_block_table` | 145 / 148 | 98.0% |
| `turn_on_lightbulb` | 160 / 160 | 100.0% |
| `rotate_blue_block_left` | 65 / 65 | 100.0% |
| `push_blue_block_left` | 54 / 65 | 83.1% |
| `close_drawer` | 172 / 172 | 100.0% |
| `lift_red_block_slider` | 101 / 117 | 86.3% |
| `turn_off_lightbulb` | 132 / 134 | 98.5% |
| `turn_on_led` | 161 / 164 | 98.2% |
| `stack_block` | 125 / 172 | 72.7% |
| `lift_blue_block_table` | 157 / 159 | 98.7% |
| `place_in_drawer` | 151 / 155 | 97.4% |
| `move_slider_left` | 221 / 229 | 96.5% |
| `rotate_red_block_left` | 63 / 63 | 100.0% |
| `push_pink_block_left` | 61 / 73 | 83.6% |
| `push_red_block_left` | 52 / 77 | 67.5% |
| `lift_pink_block_drawer` | 14 / 14 | 100.0% |
| `rotate_pink_block_right` | 33 / 66 | 50.0% |
| `unstack_block` | 48 / 48 | 100.0% |
| `lift_blue_block_slider` | 97 / 116 | 83.6% |
| `push_red_block_right` | 24 / 71 | 33.8% |
| `rotate_pink_block_left` | 52 / 54 | 96.3% |
| `push_pink_block_right` | 24 / 63 | 38.1% |
| `push_blue_block_right` | 26 / 68 | 38.2% |
| `lift_red_block_drawer` | 17 / 17 | 100.0% |

---

## Exp-008

**日期：** 2026-04-21

### serve.sh 配置

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# 每张 GPU 启动一个 flask server，端口从 9002 依次递增
# GPU 0 → port 9002, GPU 1 → port 9003, ..., GPU 7 → port 9009
```

### flask_server.py 参数

| 参数 | 值 | 说明 |
|------|----|------|
| `--model-path` | `/zhaowei/workspace/CogACT/logs/calvin_abc2d_oe_baseline_h10_8b/checkpoints/step-008372-epoch-01-loss=0.1109.pt` | 模型 checkpoint 路径（epoch-01，同目录下最早期 checkpoint） |
| `--unnorm-key` | `calvin_abc2d_oe_baseline` | 动作归一化统计键 |
| `--load-for-training` | `False`（未启用） | 是否以训练模式加载模型 |
| `--action-model-type` | `DiT-B` | 动作模型类型 |
| `--future-action-window-size` | `9` | 预测未来动作窗口大小（输出 10 步，index 0~9） |
| `--port` | `9002`（GPU 0），依次 +1 | Flask 服务端口 |
| `--debug` | `False` | 是否保存调试图片 |

### 推理参数（硬编码于 flask_server.py）

| 参数 | 值 |
|------|----|
| `unnorm_key` | `calvin_abc2d_oe_baseline`（由 `--unnorm-key` 传入） |
| `cfg_scale` | `1.5` |
| `use_ddim` | `True` |
| `num_ddim_steps` | `10` |

### 与 Exp-006 的差异

| 项目 | Exp-006 | Exp-008 |
|------|---------|---------|
| checkpoint | step-016744-epoch-02-loss=0.0623 | step-008372-epoch-01-loss=0.1109 |
| 训练步数 | 16744 | 8372 |
| epoch | 2 | 1 |
| loss | 0.0623 | 0.1109 |

### 实验结果

**Average successful sequence length:** 3.271

**Success rates for i instructions in a row:**

| 连续指令数 i | Success Rate |
|--------------|--------------|
| 1 | 89.5% |
| 2 | 76.1% |
| 3 | 63.6% |
| 4 | 54.4% |
| 5 | 43.5% |

**Per-task success rates:**

| 任务 | 成功 / 总数 | Success Rate |
|------|-------------|--------------|
| `rotate_blue_block_right` | 63 / 74 | 85.1% |
| `move_slider_right` | 249 / 250 | 99.6% |
| `turn_off_led` | 137 / 138 | 99.3% |
| `push_into_drawer` | 83 / 103 | 80.6% |
| `lift_blue_block_drawer` | 14 / 14 | 100.0% |
| `place_in_slider` | 210 / 292 | 71.9% |
| `close_drawer` | 166 / 166 | 100.0% |
| `lift_pink_block_slider` | 90 / 120 | 75.0% |
| `open_drawer` | 299 / 300 | 99.7% |
| `rotate_red_block_right` | 56 / 71 | 78.9% |
| `lift_red_block_table` | 143 / 151 | 94.7% |
| `lift_pink_block_table` | 137 / 144 | 95.1% |
| `turn_on_lightbulb` | 157 / 158 | 99.4% |
| `push_blue_block_left` | 46 / 64 | 71.9% |
| `lift_red_block_slider` | 87 / 116 | 75.0% |
| `turn_off_lightbulb` | 117 / 121 | 96.7% |
| `turn_on_led` | 147 / 149 | 98.7% |
| `stack_block` | 76 / 145 | 52.4% |
| `push_pink_block_right` | 29 / 63 | 46.0% |
| `push_red_block_left` | 43 / 73 | 58.9% |
| `lift_blue_block_table` | 146 / 148 | 98.6% |
| `rotate_blue_block_left` | 59 / 64 | 92.2% |
| `place_in_drawer` | 129 / 141 | 91.5% |
| `move_slider_left` | 212 / 212 | 100.0% |
| `rotate_red_block_left` | 54 / 61 | 88.5% |
| `push_pink_block_left` | 53 / 72 | 73.6% |
| `push_red_block_right` | 32 / 69 | 46.4% |
| `lift_pink_block_drawer` | 12 / 12 | 100.0% |
| `rotate_pink_block_right` | 33 / 68 | 48.5% |
| `lift_blue_block_slider` | 76 / 111 | 68.5% |
| `rotate_pink_block_left` | 46 / 54 | 85.2% |
| `unstack_block` | 32 / 33 | 97.0% |
| `push_blue_block_right` | 28 / 67 | 41.8% |
| `lift_red_block_drawer` | 10 / 12 | 83.3% |

---

## Exp-009

**日期：** 2026-04-21

### serve.sh 配置

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# 每张 GPU 启动一个 flask server，端口从 9002 依次递增
# GPU 0 → port 9002, GPU 1 → port 9003, ..., GPU 7 → port 9009
```

### flask_server.py 参数

| 参数 | 值 | 说明 |
|------|----|------|
| `--model-path` | `/zhaowei/workspace/CogACT/logs/calvin_abc2d_oe_h10_8b/checkpoints/step-016744-epoch-02-loss=0.0627.pt` | 模型 checkpoint 路径（目录 calvin_abc2d_oe_h10_8b，非 baseline） |
| `--unnorm-key` | `calvin_abc2d_oe` | 动作归一化统计键（非 baseline） |
| `--load-for-training` | `False`（未启用） | 是否以训练模式加载模型 |
| `--action-model-type` | `DiT-B` | 动作模型类型 |
| `--future-action-window-size` | `9` | 预测未来动作窗口大小（输出 10 步，index 0~9） |
| `--port` | `9002`（GPU 0），依次 +1 | Flask 服务端口 |
| `--debug` | `False` | 是否保存调试图片 |

### 推理参数（硬编码于 flask_server.py）

| 参数 | 值 |
|------|----|
| `unnorm_key` | `calvin_abc2d_oe`（由 `--unnorm-key` 传入） |
| `cfg_scale` | `1.5` |
| `use_ddim` | `True` |
| `num_ddim_steps` | `10` |

### 与 Exp-006 的差异

| 项目 | Exp-006 | Exp-009 |
|------|---------|---------|
| 日志目录 | `calvin_abc2d_oe_baseline_h10_8b` | `calvin_abc2d_oe_h10_8b` |
| checkpoint | step-016744-epoch-02-loss=0.0623 | step-016744-epoch-02-loss=0.0627 |
| unnorm_key | `calvin_abc2d_oe_baseline` | `calvin_abc2d_oe` |
| 训练步数 | 16744 | 16744 |
| epoch | 2 | 2 |

### 实验结果

**Average successful sequence length:** 3.52

**Success rates for i instructions in a row:**

| 连续指令数 i | Success Rate |
|--------------|--------------|
| 1 | 89.6% |
| 2 | 78.7% |
| 3 | 68.7% |
| 4 | 61.2% |
| 5 | 53.8% |

**Per-task success rates:**

| 任务 | 成功 / 总数 | Success Rate |
|------|-------------|--------------|
| `rotate_blue_block_right` | 62 / 76 | 81.6% |
| `move_slider_right` | 245 / 246 | 99.6% |
| `lift_red_block_slider` | 108 / 114 | 94.7% |
| `place_in_slider` | 230 / 307 | 74.9% |
| `turn_off_lightbulb` | 123 / 125 | 98.4% |
| `turn_off_led` | 151 / 151 | 100.0% |
| `push_into_drawer` | 85 / 105 | 81.0% |
| `lift_blue_block_drawer` | 18 / 19 | 94.7% |
| `lift_pink_block_slider` | 112 / 123 | 91.1% |
| `open_drawer` | 315 / 317 | 99.4% |
| `rotate_red_block_right` | 52 / 74 | 70.3% |
| `lift_red_block_table` | 140 / 144 | 97.2% |
| `lift_pink_block_table` | 133 / 140 | 95.0% |
| `push_blue_block_left` | 51 / 67 | 76.1% |
| `close_drawer` | 168 / 169 | 99.4% |
| `turn_on_led` | 160 / 162 | 98.8% |
| `stack_block` | 132 / 162 | 81.5% |
| `push_red_block_left` | 53 / 78 | 67.9% |
| `lift_blue_block_table` | 150 / 152 | 98.7% |
| `turn_on_lightbulb` | 155 / 158 | 98.1% |
| `rotate_blue_block_left` | 66 / 66 | 100.0% |
| `place_in_drawer` | 146 / 147 | 99.3% |
| `move_slider_left` | 219 / 224 | 97.8% |
| `push_red_block_right` | 26 / 72 | 36.1% |
| `lift_pink_block_drawer` | 10 / 12 | 83.3% |
| `rotate_pink_block_right` | 25 / 68 | 36.8% |
| `unstack_block` | 59 / 59 | 100.0% |
| `lift_blue_block_slider` | 101 / 109 | 92.7% |
| `rotate_pink_block_left` | 52 / 56 | 92.9% |
| `push_pink_block_left` | 51 / 74 | 68.9% |
| `push_pink_block_right` | 29 / 64 | 45.3% |
| `push_blue_block_right` | 21 / 67 | 31.3% |
| `rotate_red_block_left` | 60 / 62 | 96.8% |
| `lift_red_block_drawer` | 12 / 13 | 92.3% |

---

---

## Exp-010

**日期：** 2026-04-24

### serve.sh 配置

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# 每张 GPU 启动一个 flask server，端口从 9002 依次递增
# GPU 0 → port 9002, GPU 1 → port 9003, ..., GPU 7 → port 9009
```

### flask_server.py 参数

| 参数 | 值 | 说明 |
|------|----|------|
| `--model-path` | `/zhaowei/workspace/CogACT/logs/calvin_abc2d_oe_h10_8b/checkpoints/step-025116-epoch-03-loss=0.0413.pt` | 模型 checkpoint 路径（epoch-03，较 Exp-009 训练更充分） |
| `--unnorm-key` | `calvin_abc2d_oe` | 动作归一化统计键 |
| `--action-model-type` | `DiT-B` | 动作模型类型（默认值） |
| `--future-action-window-size` | `9` | 预测未来动作窗口大小（输出 10 步，index 0~9） |
| `--port` | `9002`（GPU 0），依次 +1 | Flask 服务端口 |

### 推理参数（硬编码于 flask_server.py）

| 参数 | 值 |
|------|----|
| `unnorm_key` | `calvin_abc2d_oe` |
| `cfg_scale` | `1.5` |
| `use_ddim` | `True` |
| `num_ddim_steps` | `10` |

### 与 Exp-009 的差异

| 项目 | Exp-009 | Exp-010 |
|------|---------|---------|
| checkpoint | step-016744-epoch-02-loss=0.0627 | step-025116-epoch-03-loss=0.0413 |
| 训练步数 | 16744 | 25116 |
| epoch | 2 | 3 |
| loss | 0.0627 | 0.0413 |

### 实验结果

**Average successful sequence length:** 3.721

**Success rates for i instructions in a row:**

| 连续指令数 i | Success Rate |
|--------------|--------------|
| 1 | 91.4% |
| 2 | 82.5% |
| 3 | 73.7% |
| 4 | 66.3% |
| 5 | 58.2% |

**Per-task success rates:**

| 任务 | 成功 / 总数 | Success Rate |
|------|-------------|--------------|
| `rotate_blue_block_right` | 67 / 77 | 87.0% |
| `move_slider_right` | 255 / 255 | 100.0% |
| `lift_red_block_slider` | 112 / 121 | 92.6% |
| `place_in_slider` | 259 / 328 | 79.0% |
| `turn_off_lightbulb` | 135 / 136 | 99.3% |
| `turn_off_led` | 155 / 156 | 99.4% |
| `push_into_drawer` | 86 / 104 | 82.7% |
| `lift_blue_block_drawer` | 19 / 19 | 100.0% |
| `lift_pink_block_slider` | 101 / 124 | 81.5% |
| `open_drawer` | 316 / 319 | 99.1% |
| `rotate_red_block_right` | 57 / 74 | 77.0% |
| `lift_red_block_table` | 152 / 154 | 98.7% |
| `lift_pink_block_table` | 143 / 149 | 96.0% |
| `turn_on_lightbulb` | 170 / 170 | 100.0% |
| `rotate_blue_block_left` | 66 / 66 | 100.0% |
| `push_blue_block_left` | 57 / 68 | 83.8% |
| `close_drawer` | 185 / 187 | 98.9% |
| `turn_on_led` | 165 / 168 | 98.2% |
| `push_pink_block_right` | 28 / 65 | 43.1% |
| `push_red_block_left` | 58 / 79 | 73.4% |
| `lift_blue_block_table` | 164 / 164 | 100.0% |
| `place_in_drawer` | 157 / 159 | 98.7% |
| `move_slider_left` | 230 / 232 | 99.1% |
| `rotate_red_block_left` | 62 / 64 | 96.9% |
| `stack_block` | 132 / 165 | 80.0% |
| `push_pink_block_left` | 56 / 75 | 74.7% |
| `lift_pink_block_drawer` | 11 / 11 | 100.0% |
| `rotate_pink_block_right` | 47 / 68 | 69.1% |
| `lift_blue_block_slider` | 105 / 113 | 92.9% |
| `unstack_block` | 62 / 62 | 100.0% |
| `push_red_block_right` | 25 / 70 | 35.7% |
| `rotate_pink_block_left` | 49 / 56 | 87.5% |
| `push_blue_block_right` | 23 / 69 | 33.3% |
| `lift_red_block_drawer` | 12 / 12 | 100.0% |

---

## Exp-011

**日期：** 2026-04-24

### serve.sh 配置

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# 每张 GPU 启动一个 flask server，端口从 9002 依次递增
# GPU 0 → port 9002, GPU 1 → port 9003, ..., GPU 7 → port 9009
```

### flask_server.py 参数

| 参数 | 值 | 说明 |
|------|----|------|
| `--model-path` | `/zhaowei/workspace/CogACT/logs/calvin_abc2d_oe_h10_8b/checkpoints/step-008372-epoch-01-loss=0.1151.pt` | 模型 checkpoint 路径（epoch-01，同目录下最早期 checkpoint） |
| `--unnorm-key` | `calvin_abc2d_oe` | 动作归一化统计键 |
| `--action-model-type` | `DiT-B` | 动作模型类型（默认值） |
| `--future-action-window-size` | `9` | 预测未来动作窗口大小（输出 10 步，index 0~9） |
| `--port` | `9002`（GPU 0），依次 +1 | Flask 服务端口 |

### 推理参数（硬编码于 flask_server.py）

| 参数 | 值 |
|------|----|
| `unnorm_key` | `calvin_abc2d_oe` |
| `cfg_scale` | `1.5` |
| `use_ddim` | `True` |
| `num_ddim_steps` | `10` |

### 与 Exp-010 的差异

| 项目 | Exp-010 | Exp-011 |
|------|---------|---------|
| checkpoint | step-025116-epoch-03-loss=0.0413 | step-008372-epoch-01-loss=0.1151 |
| 训练步数 | 25116 | 8372 |
| epoch | 3 | 1 |
| loss | 0.0413 | 0.1151 |

### 实验结果

**Average successful sequence length:** 3.001

**Success rates for i instructions in a row:**

| 连续指令数 i | Success Rate |
|--------------|--------------|
| 1 | 85.0% |
| 2 | 70.5% |
| 3 | 57.5% |
| 4 | 47.9% |
| 5 | 39.2% |

**Per-task success rates:**

| 任务 | 成功 / 总数 | Success Rate |
|------|-------------|--------------|
| `rotate_blue_block_right` | 59 / 73 | 80.8% |
| `move_slider_right` | 249 / 251 | 99.2% |
| `turn_off_led` | 137 / 138 | 99.3% |
| `push_into_drawer` | 77 / 88 | 87.5% |
| `lift_blue_block_drawer` | 17 / 17 | 100.0% |
| `open_drawer` | 283 / 285 | 99.3% |
| `lift_pink_block_table` | 120 / 121 | 99.2% |
| `place_in_slider` | 190 / 268 | 70.9% |
| `turn_on_lightbulb` | 140 / 140 | 100.0% |
| `rotate_blue_block_left` | 61 / 64 | 95.3% |
| `rotate_red_block_right` | 52 / 63 | 82.5% |
| `turn_on_led` | 143 / 143 | 100.0% |
| `push_pink_block_right` | 29 / 61 | 47.5% |
| `close_drawer` | 156 / 158 | 98.7% |
| `lift_blue_block_table` | 125 / 128 | 97.7% |
| `place_in_drawer` | 131 / 136 | 96.3% |
| `turn_off_lightbulb` | 106 / 107 | 99.1% |
| `lift_pink_block_slider` | 91 / 114 | 79.8% |
| `move_slider_left` | 121 / 205 | 59.0% |
| `rotate_red_block_left` | 56 / 57 | 98.2% |
| `lift_red_block_slider` | 80 / 107 | 74.8% |
| `stack_block` | 91 / 139 | 65.5% |
| `lift_red_block_table` | 128 / 134 | 95.5% |
| `lift_pink_block_drawer` | 7 / 7 | 100.0% |
| `rotate_pink_block_right` | 38 / 62 | 61.3% |
| `unstack_block` | 44 / 45 | 97.8% |
| `lift_blue_block_slider` | 82 / 99 | 82.8% |
| `push_red_block_right` | 25 / 69 | 36.2% |
| `rotate_pink_block_left` | 42 / 50 | 84.0% |
| `push_pink_block_left` | 31 / 70 | 44.3% |
| `push_red_block_left` | 32 / 71 | 45.1% |
| `push_blue_block_left` | 29 / 62 | 46.8% |
| `lift_red_block_drawer` | 11 / 14 | 78.6% |
| `push_blue_block_right` | 18 / 63 | 28.6% |

---

## Exp-012

**日期：** 2026-04-27

### serve.sh 配置

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# 每张 GPU 启动一个 flask server，端口从 9002 依次递增
# GPU 0 → port 9002, GPU 1 → port 9003, ..., GPU 7 → port 9009
```

### flask_server.py 参数

| 参数 | 值 | 说明 |
|------|----|------|
| `--model-path` | `/zhaowei/workspace/CogACT/logs/calvin_abc2d_oe_h10_layerwise/checkpoints/step-016744-epoch-02-loss=0.0585.pt` | 模型 checkpoint 路径（layerwise 架构，目录 calvin_abc2d_oe_h10_layerwise） |
| `--unnorm-key` | `calvin_abc2d_oe` | 动作归一化统计键 |
| `--future-action-window-size` | `9` | 预测未来动作窗口大小（输出 10 步，index 0~9） |
| `--port` | `9002`（GPU 0），依次 +1 | Flask 服务端口 |

### 推理参数（硬编码于 flask_server.py）

| 参数 | 值 |
|------|----|
| `unnorm_key` | `calvin_abc2d_oe` |
| `cfg_scale` | `1.5` |
| `use_ddim` | `True` |
| `num_ddim_steps` | `10` |

### 与 Exp-010 的差异

| 项目 | Exp-010 | Exp-012 |
|------|---------|---------|
| 日志目录 | `calvin_abc2d_oe_h10_8b` | `calvin_abc2d_oe_h10_layerwise` |
| checkpoint | step-025116-epoch-03-loss=0.0413 | step-016744-epoch-02-loss=0.0585 |
| 训练步数 | 25116 | 16744 |
| epoch | 3 | 2 |
| loss | 0.0413 | 0.0585 |
| 架构 | 8B 标准 | layerwise |

### 实验结果

**Average successful sequence length:** 3.526

**Success rates for i instructions in a row:**

| 连续指令数 i | Success Rate |
|--------------|--------------|
| 1 | 90.3% |
| 2 | 79.2% |
| 3 | 69.3% |
| 4 | 61.1% |
| 5 | 52.7% |

**Per-task success rates:**

| 任务 | 成功 / 总数 | Success Rate |
|------|-------------|--------------|
| `rotate_blue_block_right` | 69 / 72 | 95.8% |
| `move_slider_right` | 257 / 257 | 100.0% |
| `lift_red_block_slider` | 111 / 120 | 92.5% |
| `place_in_slider` | 239 / 323 | 74.0% |
| `turn_off_lightbulb` | 126 / 128 | 98.4% |
| `turn_off_led` | 149 / 151 | 98.7% |
| `lift_pink_block_slider` | 114 / 124 | 91.9% |
| `open_drawer` | 311 / 311 | 100.0% |
| `rotate_red_block_right` | 68 / 73 | 93.2% |
| `lift_red_block_table` | 148 / 148 | 100.0% |
| `lift_pink_block_table` | 144 / 150 | 96.0% |
| `turn_on_lightbulb` | 158 / 158 | 100.0% |
| `rotate_blue_block_left` | 62 / 62 | 100.0% |
| `turn_on_led` | 156 / 159 | 98.1% |
| `push_red_block_left` | 47 / 75 | 62.7% |
| `lift_blue_block_table` | 147 / 148 | 99.3% |
| `place_in_drawer` | 153 / 155 | 98.7% |
| `move_slider_left` | 206 / 215 | 95.8% |
| `rotate_red_block_left` | 61 / 61 | 100.0% |
| `close_drawer` | 180 / 180 | 100.0% |
| `stack_block` | 95 / 167 | 56.9% |
| `lift_pink_block_drawer` | 12 / 12 | 100.0% |
| `rotate_pink_block_right` | 61 / 69 | 88.4% |
| `lift_blue_block_slider` | 108 / 121 | 89.3% |
| `unstack_block` | 32 / 32 | 100.0% |
| `push_red_block_right` | 21 / 71 | 29.6% |
| `push_into_drawer` | 84 / 102 | 82.4% |
| `rotate_pink_block_left` | 50 / 54 | 92.6% |
| `push_pink_block_left` | 46 / 73 | 63.0% |
| `push_blue_block_left` | 43 / 67 | 64.2% |
| `push_pink_block_right` | 23 / 61 | 37.7% |
| `lift_red_block_drawer` | 13 / 13 | 100.0% |
| `lift_blue_block_drawer` | 16 / 16 | 100.0% |
| `push_blue_block_right` | 16 / 71 | 22.5% |

---

## Exp-013

**日期：** 2026-04-28

### serve.sh 配置

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# 每张 GPU 启动一个 flask server，端口从 9002 依次递增
# GPU 0 → port 9002, GPU 1 → port 9003, ..., GPU 7 → port 9009
```

### flask_server.py 参数

| 参数 | 值 | 说明 |
|------|----|------|
| `--model-path` | `/zhaowei/workspace/CogACT/logs/calvin_abc2d_oe_h10_layerwise/checkpoints/step-025116-epoch-03-loss=0.0451.pt` | 模型 checkpoint 路径（layerwise 架构，epoch-03） |
| `--unnorm-key` | `calvin_abc2d_oe` | 动作归一化统计键 |
| `--future-action-window-size` | `9` | 预测未来动作窗口大小（输出 10 步，index 0~9） |
| `--port` | `9002`（GPU 0），依次 +1 | Flask 服务端口 |

### 推理参数（硬编码于 flask_server.py）

| 参数 | 值 |
|------|----|
| `unnorm_key` | `calvin_abc2d_oe` |
| `cfg_scale` | `1.5` |
| `use_ddim` | `True` |
| `num_ddim_steps` | `10` |

### 与 Exp-012 的差异

| 项目 | Exp-012 | Exp-013 |
|------|---------|---------|
| checkpoint | step-016744-epoch-02-loss=0.0585 | step-025116-epoch-03-loss=0.0451 |
| 训练步数 | 16744 | 25116 |
| epoch | 2 | 3 |
| loss | 0.0585 | 0.0451 |

### 实验结果

**Average successful sequence length:** 3.68

**Success rates for i instructions in a row:**

| 连续指令数 i | Success Rate |
|--------------|--------------|
| 1 | 91.1% |
| 2 | 81.7% |
| 3 | 73.0% |
| 4 | 64.9% |
| 5 | 57.3% |

**Per-task success rates:**

| 任务 | 成功 / 总数 | Success Rate |
|------|-------------|--------------|
| `rotate_blue_block_right` | 72 / 75 | 96.0% |
| `move_slider_right` | 260 / 261 | 99.6% |
| `lift_red_block_slider` | 115 / 121 | 95.0% |
| `place_in_slider` | 266 / 337 | 78.9% |
| `turn_off_lightbulb` | 131 / 132 | 99.2% |
| `turn_off_led` | 147 / 148 | 99.3% |
| `push_into_drawer` | 91 / 109 | 83.5% |
| `lift_blue_block_drawer` | 20 / 20 | 100.0% |
| `close_drawer` | 185 / 186 | 99.5% |
| `lift_pink_block_slider` | 114 / 123 | 92.7% |
| `open_drawer` | 322 / 322 | 100.0% |
| `rotate_red_block_right` | 69 / 74 | 93.2% |
| `lift_red_block_table` | 156 / 159 | 98.1% |
| `lift_pink_block_table` | 162 / 166 | 97.6% |
| `turn_on_lightbulb` | 158 / 159 | 99.4% |
| `rotate_blue_block_left` | 65 / 65 | 100.0% |
| `turn_on_led` | 163 / 164 | 99.4% |
| `push_pink_block_right` | 25 / 61 | 41.0% |
| `push_red_block_left` | 49 / 74 | 66.2% |
| `lift_blue_block_table` | 151 / 152 | 99.3% |
| `place_in_drawer` | 160 / 161 | 99.4% |
| `move_slider_left` | 210 / 226 | 92.9% |
| `rotate_red_block_left` | 62 / 62 | 100.0% |
| `stack_block` | 99 / 165 | 60.0% |
| `lift_pink_block_drawer` | 10 / 11 | 90.9% |
| `rotate_pink_block_right` | 60 / 67 | 89.6% |
| `lift_blue_block_slider` | 103 / 114 | 90.4% |
| `unstack_block` | 38 / 39 | 97.4% |
| `push_red_block_right` | 32 / 71 | 45.1% |
| `rotate_pink_block_left` | 52 / 55 | 94.5% |
| `push_pink_block_left` | 51 / 74 | 68.9% |
| `push_blue_block_left` | 43 / 66 | 65.2% |
| `lift_red_block_drawer` | 15 / 16 | 93.8% |
| `push_blue_block_right` | 24 / 72 | 33.3% |

<!-- 新增实验请复制下方模板 -->

<!--
## Exp-XXX

**日期：** YYYY-MM-DD

### serve.sh 配置

```bash
CUDA_VISIBLE_DEVICES="..."
```

### flask_server.py 参数

| 参数 | 值 | 说明 |
|------|----|------|
| `--model-path` | | |
| `--action-model-type` | | |
| `--future-action-window-size` | | |

### 推理参数

| 参数 | 值 |
|------|----|
| `unnorm_key` | |
| `cfg_scale` | |
| `use_ddim` | |
| `num_ddim_steps` | |

### 实验结果

> （待补充）
-->
