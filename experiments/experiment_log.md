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
