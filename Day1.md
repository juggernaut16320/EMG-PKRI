### Day1 要完成的代码（结构 + 容器友好）

> **环境准备**：开始前请确保已激活 conda 环境：`conda activate emgpkri`

```text
project/
│
├── data/
│   ├── dataset_raw.jsonl
│   ├── cleaned_raw.jsonl
│   ├── with_coarse_label.jsonl
│   ├── with_coarse_and_subtypes.jsonl
│   ├── train.jsonl
│   ├── dev.jsonl
│   ├── test.jsonl
│   ├── hard_eval_set.jsonl
│
├── configs/
│   └── config.yaml          # data_dir / backends / 阈值 等
│
├── scripts/
│   ├── txt_to_jsonl.py
│   ├── llm_labeler.py
│   ├── data_cleaner.py
│   ├── coarse_label.py
│   ├── subtype_assign.py
│   ├── dataset_split.py
│   ├── hardset_maker.py
│
├── tests/
│   ├── test_txt_to_jsonl.py
│   ├── test_data_cleaner.py
│   ├── test_coarse_label.py
│   ├── test_subtype_assign.py
│   ├── test_dataset_split.py
│   ├── test_hardset_maker.py
│
├── requirements.txt
└── .env.example
```

### 0. `txt_to_jsonl.py`（TXT 转 JSONL 工具）

**功能：**
将 `unprocessed` 目录下的多个 txt 文件转换为统一的 JSONL 格式。

**关键特性：**

- 从 `unprocessed` 目录读取所有 `.txt` 文件
- 使用全局序号（从 `s0` 开始，`s+num` 格式）为每条记录生成 ID
- 如果输出 JSONL 已存在，自动检测最大 ID 并继续编号（增量处理）
- 处理完成后自动删除已处理的 txt 文件
- 性能优化：大文件时使用临时文件策略，避免追加大文件的性能问题

**使用方式：**

```bash
# 基本使用
python scripts/txt_to_jsonl.py --unprocessed-dir data/unprocessed --output data/dataset_raw.jsonl

# 自定义起始 ID（默认从 s0 开始）
python scripts/txt_to_jsonl.py --start-id 0

# 强制使用临时文件策略（适用于大文件）
python scripts/txt_to_jsonl.py --use-temp
```

**输出格式：**

每条记录包含：
- `id`: 格式为 `s0`, `s1`, `s2` 等
- `text`: txt 文件的文本内容

**增量处理：**

- 如果 `dataset_raw.jsonl` 已存在，工具会自动检测最大 ID（如 `s99`），新数据将从 `s100` 开始编号
- 支持无缝续接，无需手动合并

------

### 1. `llm_labeler.py`（LLM 打标接口）

**功能：**
 封装「调用 Gemini 模型打标签」逻辑，后面主标签和子标签都用它。

**关键容器化/增量点：**

- 从 `config.yaml` 读取：

  - `batch_size`、`max_retries` 等参数。

- 从环境变量读取：

  - `GEMINI_API_KEY`（必需）。

- 模型锁定：

  - 当前仅使用 `gemma-3-27b-it` 模型（30 RPM，14.4K RPD，15K TPM，适合大批量打标）。

- 定义统一调用接口：

  ```python
  def call_llm_backend(prompt: str) -> str:
      # 使用 gemini-2.5-flash-live 调用 API
      ...
  ```

- `run_label_task(...)` 支持：

  - `batch_size`（默认 10，支持批量处理多条数据打包传给大模型）；
  - `request_interval`（默认 2.5 秒，确保不超过 RPM 限制）；
  - 从指定 input/output 路径处理 jsonl（增量数据时重复调用即可）；
  - 批量处理失败时自动回退到逐条处理。

其余思路与原描述一致：`LabelTask` + `prompt_template` + `parse_fn`。

------

### 2. `data_cleaner.py`

**功能：**
 把原始 `dataset_raw.jsonl` 清洗成 `cleaned_raw.jsonl`。

**小改点：**

- 从 `config.yaml` 或命令行读取：

  ```python
  data_dir = cfg["data_dir"]  # 默认 "./data"
  input_path = os.path.join(data_dir, "dataset_raw.jsonl")
  output_path = os.path.join(data_dir, "cleaned_raw.jsonl")
  ```

- 不写死路径，这样云端挂载 `/app/data` 也只需改 config。

**增量数据：**

- 如果有 `dataset_new_raw.jsonl`：
  - 可以设计参数 `--input dataset_new_raw.jsonl --output cleaned_new.jsonl`；
  - 清洗逻辑不变，方便独立处理新数据。

------

### 3. `coarse_label.py`

**功能：**
 利用 LLM 给每条样本打主标签：`label = 1`（敏感）或 `0`（非敏感）。

**小改点（容器 & 增量）：**

- 输入输出路径从 `config.yaml` / CLI 获取；
- 使用 `llm_labeler.run_label_task(...)`，自动使用 `gemma-3-27b-it` 模型。
- 支持批量处理（默认 batch_size=10），大幅减少 API 调用次数。
- 对新数据：
  - 允许传入 `--input cleaned_new.jsonl --output with_coarse_label_new.jsonl`。

其余逻辑同你原来的设计。

------

### 4. `subtype_assign.py`

**功能：**
 给敏感样本添加 `subtypes` 多标签字段。

**容器/增量改动：**

- 同样通过 `config.yaml` 和 CLI 决定：
  - 输入是全量还是仅新数据；
  - 输出文件名。
- 可以支持一个简单开关：
  - `--only-sensitive`：只对 `coarse_label=1` 的数据调用 LLM 打子标签，然后合并回原始数据。

其它逻辑保持不变。

------

### 5. `dataset_split.py`

**功能：**
 将打好标签的数据按 8/1/1 划分 train/dev/test（stratify=coarse_label）。

**关键特性：**

- **完全重新划分策略**：每次运行都会重新划分所有数据（旧数据+新数据合并后重新划分）
- **固定随机种子**：使用固定种子（默认42，可在 config.yaml 配置），确保可复现
- **分层划分**：按 `coarse_label` 分层，保证 train/dev/test 中敏感/非敏感比例一致
- **增量数据影响**：新数据加入后，原来的 train/dev/test 中的样本会重新分配（这是正常且推荐的做法）

**使用方式：**

```bash
# 基本使用（默认从 dataset_with_coarse.jsonl 或 with_coarse_and_subtypes.jsonl 读取）
python scripts/dataset_split.py

# 指定输入文件
python scripts/dataset_split.py --input dataset_with_coarse.jsonl

# 自定义输出路径
python scripts/dataset_split.py --input dataset_with_coarse.jsonl --train-output train.jsonl --dev-output dev.jsonl --test-output test.jsonl

# 自定义随机种子
python scripts/dataset_split.py --random-seed 123
```

**输出格式：**

- `train.jsonl`：训练集（默认80%）
- `dev.jsonl`：验证集（默认10%）
- `test.jsonl`：测试集（默认10%）

**增量友好：**

- 只依赖一个"合并后的全量数据文件"，比如：
  - `with_coarse_and_subtypes_all.jsonl` 或 `dataset_with_coarse.jsonl`；
- 每次有增量数据，只要你先合并，再跑这一脚本，就可以得到新的 train/dev/test。
- **注意**：使用完全重新划分策略，原来的 train/dev/test 中的样本会重新分配，这是正常且推荐的做法，因为：
  - 保证分层比例一致
  - 新数据均匀分布
  - 便于增量训练（从旧 checkpoint 继续训练）

------

### 6. `hardset_maker.py`

**功能：**
 基于 teacher–student 分歧构造困难子集。

**容器/增量点：**

- 使用 `gemma-3-27b` 模型进行打标；
- 路径都走 `data_dir`；
- 每次重新训练 baseline 或增加数据后，只需：
  - 用最新的 dev/test + 最新 baseline 重新跑一次这个脚本，
  - hard_eval_set 就更新了，后面 Day3/Day9/Day11 自然用最新的数据。