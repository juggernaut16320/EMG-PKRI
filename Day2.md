### Day2 要完成的代码（结构 + 容器友好）

> **环境准备**：开始前请确保已激活 conda 环境：`conda activate emgpkri`

```text
project/
│
├── data/
│   ├── train.jsonl
│   ├── dev.jsonl
│   ├── test.jsonl
│
├── checkpoints/
│   └── baseline-lora/
│       ├── adapter_model.bin
│       ├── adapter_config.json
│       ├── training_args.bin
│       └── ...
│
├── configs/
│   └── config.yaml          # model_name_or_path / LoRA 超参 / 训练参数等
│
├── scripts/
│   └── baseline_train.py
│
├── tests/
│   └── test_baseline_train.py
│
├── requirements.txt
└── .env.example
```

### `baseline_train.py`（基线模型训练工具）

**功能：**
使用 Qwen 模型 + LoRA 微调训练二分类基线模型（敏感/非敏感）。

**关键特性：**

- **模型选择**：使用 Qwen1.5-1.7B 或 Qwen1.5-0.5B（可在 config.yaml 配置）
- **LoRA 微调**：只训练 LoRA 适配器，不训练全量参数
- **任务类型**：二分类（敏感/非敏感），不训练子标签头
- **监控指标**：在 dev 集上监控 loss 和 F1，保存最优 checkpoint
- **增量训练支持**：支持从已有 checkpoint 继续训练

**使用方式：**

```bash
# 基本使用（从 config.yaml 读取所有参数）
python scripts/baseline_train.py

# 指定训练和验证文件
python scripts/baseline_train.py --train-file train.jsonl --dev-file dev.jsonl

# 从已有 checkpoint 继续训练（增量训练）
python scripts/baseline_train.py --resume-from-checkpoint checkpoints/baseline-lora/

# 自定义输出目录
python scripts/baseline_train.py --output-dir checkpoints/my-baseline/

# 覆盖配置中的模型路径
python scripts/baseline_train.py --model-name-or-path Qwen/Qwen1.5-0.5B
```

**输入格式：**

**位置：** `data/train.jsonl`, `data/dev.jsonl`

**格式：** JSONL（每行一个 JSON 对象）

**必需字段：**
- `id`: 样本 ID（如 `s0`, `s1`）
- `text`: 文本内容
- `coarse_label`: 标签（0=非敏感, 1=敏感）

**示例：**
```json
{"id": "s0", "text": "这是一条测试文本", "coarse_label": 0}
{"id": "s1", "text": "这是另一条测试文本", "coarse_label": 1}
```

**输出格式：**

**位置：** `checkpoints/baseline-lora/`

**文件结构：**
```
checkpoints/baseline-lora/
├── adapter_model.bin          # LoRA 权重
├── adapter_config.json        # LoRA 配置
├── training_args.bin          # 训练参数
├── tokenizer_config.json      # Tokenizer 配置
├── special_tokens_map.json
└── ...
```

**配置参数（config.yaml）：**

```yaml
# 模型配置
model:
  name_or_path: "Qwen/Qwen1.5-1.7B"  # 或 Qwen/Qwen1.5-0.5B
  max_length: 512                      # 最大序列长度

# LoRA 配置
lora:
  r: 8                                 # LoRA rank
  alpha: 16                            # LoRA alpha
  dropout: 0.1                         # LoRA dropout
  target_modules: ["q_proj", "v_proj"] # 目标模块（可配置）

# 训练配置
training:
  train_file: "train.jsonl"           # 训练集文件（相对于 data_dir）
  dev_file: "dev.jsonl"                # 验证集文件（相对于 data_dir）
  output_dir: "checkpoints/baseline-lora"  # 输出目录
  num_epochs: 3                        # 训练轮数
  per_device_train_batch_size: 8       # 训练批次大小
  per_device_eval_batch_size: 16       # 验证批次大小
  learning_rate: 2e-4                  # 学习率
  warmup_steps: 100                    # Warmup 步数
  logging_steps: 50                    # 日志记录步数
  eval_steps: 200                      # 评估步数
  save_steps: 500                      # 保存 checkpoint 步数
  save_strategy: "steps"               # 保存策略
  evaluation_strategy: "steps"        # 评估策略
  load_best_model_at_end: true        # 训练结束时加载最优模型
  metric_for_best_model: "f1"          # 最优模型指标（f1/loss/accuracy）
  greater_is_better: true              # F1 越大越好
  save_total_limit: 3                  # 最多保存的 checkpoint 数量
  seed: 42                             # 随机种子
```

**增量训练说明：**

- **新数据加入后：**
  1. 重新生成新的 `train/dev/test`（通过 `dataset_split.py`）
  2. 重新运行 `baseline_train.py`：
     - **方式1（增量微调）**：使用 `--resume-from-checkpoint` 从旧 checkpoint 继续训练
       ```bash
       python scripts/baseline_train.py --resume-from-checkpoint checkpoints/baseline-lora/
       ```
     - **方式2（完整重训）**：使用旧 checkpoint 作为 warm start，在新数据上完整重训
       ```bash
       python scripts/baseline_train.py --model-name-or-path checkpoints/baseline-lora/
       ```
  3. 两种方式都会很快，因为 LoRA 参数量小

- **后续依赖：**
  - Day3–Day12 都只依赖最新的 baseline 概率输出 `p(c|x)`
  - 只要 baseline checkpoint 更新，后续脚本无需改代码

**工程/容器化约定：**

- 所有路径从 `config.yaml` 或 CLI 参数读取：
  - `data_dir`（默认 `./data`）
  - `output_dir`（默认 `./checkpoints/baseline-lora`）
  - `train_file` / `dev_file`（相对于 `data_dir`）

- 模型路径：
  - 支持 HuggingFace 模型 ID（如 `Qwen/Qwen1.5-1.7B`）
  - 支持本地路径（如 `checkpoints/baseline-lora/`）

- 输出目录：
  - 统一使用 `checkpoints/baseline-lora/` 目录
  - 方便云端和本地统一使用

**训练监控：**

- 训练过程中会输出：
  - Loss（训练集和验证集）
  - F1 分数（验证集）
  - Accuracy（验证集）
- 自动保存最优 checkpoint（基于验证集 F1）
- 训练日志保存在 `checkpoints/baseline-lora/training.log`

**注意事项：**

- **只训练二分类**：不训练子标签头，只关注敏感/非敏感二分类
- **LoRA 参数**：默认只训练 `q_proj` 和 `v_proj`，可根据需要调整
- **内存优化**：如果显存不足，可以：
  - 减小 `per_device_train_batch_size`
  - 使用梯度累积（`gradient_accumulation_steps`）
  - 使用更小的模型（如 Qwen1.5-0.5B）

