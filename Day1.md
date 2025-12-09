### Day1 要完成的代码（结构 + 容器友好）

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
│   ├── llm_labeler.py
│   ├── data_cleaner.py
│   ├── coarse_label.py
│   ├── subtype_assign.py
│   ├── dataset_split.py
│   ├── hardset_maker.py
│
├── tests/
│   ├── test_data_cleaner.py
│   ├── test_coarse_label.py
│   ├── test_subtype_assign.py
│   ├── test_dataset_split.py
│   ├── test_hardset_maker.py
│
├── requirements.txt
└── .env.example
```

### 0. `llm_labeler.py`（通用 LLM 打标接口）

**功能：**
 封装「调用大模型打标签」逻辑，后面主标签和子标签都用它。

**关键容器化/增量点：**

- 从 `config.yaml` 读取：

  - 默认 backend 列表（如 `teacher_backends`、`student_backend`）；
  - `batch_size`、`max_retries` 等参数。

- 从环境变量读取：

  - `OPENAI_API_KEY`、`GEMINI_API_KEY` 等。

- 定义统一后端调用接口：

  ```python
  def call_llm_backend(backend: str, prompt: str) -> str:
      # 根据 backend = "gpt-4o" / "gemini-pro" 选择不同 API
      ...
  ```

- `run_label_task(...)` 支持：

  - `batch_size`（本地 1，云端可以 10）；
  - 从指定 input/output 路径处理 jsonl（增量数据时重复调用即可）。

其余思路与原描述一致：`LabelTask` + `prompt_template` + `parse_fn`。

------

### 1. `data_cleaner.py`

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

### 2. `coarse_label.py`

**功能：**
 利用 LLM 给每条样本打主标签：`label = 1`（敏感）或 `0`（非敏感）。

**小改点（容器 & 增量）：**

- 输入输出路径从 `config.yaml` / CLI 获取；
- 使用 `llm_labeler.run_label_task(...)`，支持：
  - 指定 `backends=cfg["teacher_backends"]`；
  - 未来想用多模型共识时，只改配置，不改脚本。
- 对新数据：
  - 允许传入 `--input cleaned_new.jsonl --output with_coarse_label_new.jsonl`。

其余逻辑同你原来的设计。

------

### 3. `subtype_assign.py`

**功能：**
 给敏感样本添加 `subtypes` 多标签字段。

**容器/增量改动：**

- 同样通过 `config.yaml` 和 CLI 决定：
  - 输入是全量还是仅新数据；
  - 输出文件名。
- 可以支持一个简单开关：
  - `--only_sensitive`：只对 `label=1` 的数据调用 LLM。

其它逻辑保持不变。

------

### 4. `dataset_split.py`

**功能：**
 将打好标签的数据按 8/1/1 划分 train/dev/test（stratify=label）。

**增量友好：**

- 只依赖一个“合并后的全量数据文件”，比如：
  - `with_coarse_and_subtypes_all.jsonl`；
- 每次有增量数据，只要你先合并，再跑这一脚本，就可以得到新的 train/dev/test。

------

### 5. `hardset_maker.py`

**功能：**
 基于 teacher–student 分歧构造困难子集。

**容器/增量点：**

- teacher / student backend 从 `config.yaml` 读取；
- 路径都走 `data_dir`；
- 每次重新训练 baseline 或增加数据后，只需：
  - 用最新的 dev/test + 最新 baseline 重新跑一次这个脚本，
  - hard_eval_set 就更新了，后面 Day3/Day9/Day11 自然用最新的数据。