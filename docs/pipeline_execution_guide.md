# 完整链路运行指南

本文档提供 Day1 中 0-4 步骤的完整运行命令和数据格式说明。

## 前置准备

### 1. 环境配置

```bash
# 激活 conda 环境
conda activate emgpkri

# 加载 API Key（Windows PowerShell）
$env:GEMINI_API_KEY = (Get-Content .env | Select-String -Pattern "GEMINI_API_KEY").ToString().Split('=')[1]

# 加载 API Key（Linux/Mac）
export GEMINI_API_KEY=$(grep GEMINI_API_KEY .env | cut -d '=' -f2)
```

### 2. 准备数据

将待处理的 txt 文件放入 `data/unprocessed/` 目录。

---

## 步骤 0: txt_to_jsonl.py

### 命令

```bash
python scripts/txt_to_jsonl.py --unprocessed-dir data/unprocessed --output data/dataset_raw.jsonl
```

### 输入格式

**位置：** `data/unprocessed/*.txt`

**格式：** 纯文本文件，每行一个文本内容

**示例：**
```
这是第一条文本内容
这是第二条文本内容
```

### 输出格式

**位置：** `data/dataset_raw.jsonl`

**格式：** JSONL（每行一个 JSON 对象）

**字段：**
- `id`: 字符串，格式为 `s0`, `s1`, `s2` 等
- `text`: 字符串，txt 文件的文本内容

**示例：**
```json
{"id": "s0", "text": "这是第一条文本内容"}
{"id": "s1", "text": "这是第二条文本内容"}
```

### 关键特性

- 自动检测最大 ID，支持增量处理
- 处理完成后自动删除已处理的 txt 文件
- ID 从 `s0` 开始，自动递增

---

## 步骤 1: data_cleaner.py

### 命令

```bash
python scripts/data_cleaner.py
```

### 输入格式

**位置：** `data/dataset_raw.jsonl`（默认）

**格式：** JSONL

**字段：**
- `id`: 字符串
- `text`: 字符串，原始文本

**示例：**
```json
{"id": "s0", "text": "访问 https://example.com 联系 @user 获取信息"}
```

### 输出格式

**位置：** `data/cleaned_raw.jsonl`（默认）

**格式：** JSONL

**字段：**
- `id`: 字符串（保留）
- `text`: 字符串，清洗后的文本

**处理内容：**
- URL → `[URL]`
- @提及 → `[MENTION]`
- 转发内容提取
- 去重（基于文本哈希）

**示例：**
```json
{"id": "s0", "text": "访问 [URL] 联系 [MENTION] 获取信息"}
```

### 关键特性

- 支持自定义输入输出路径：`--input` 和 `--output`
- 自动去重
- 保留原始 ID

---

## 步骤 2: coarse_label.py

### 命令

```bash
python scripts/coarse_label.py
```

### 输入格式

**位置：** `data/cleaned_raw.jsonl`（默认）

**格式：** JSONL

**字段：**
- `id`: 字符串
- `text`: 字符串，清洗后的文本

**示例：**
```json
{"id": "s0", "text": "访问 [URL] 联系 [MENTION] 获取信息"}
```

### 输出格式

**位置：** `data/with_coarse_label.jsonl`（默认）

**格式：** JSONL

**字段：**
- `id`: 字符串（保留）
- `text`: 字符串（保留）
- `coarse_label`: 整数，`0` 表示非敏感，`1` 表示敏感
- `coarse_response`: 字符串，LLM 原始响应

**示例：**
```json
{
  "id": "s0",
  "text": "访问 [URL] 联系 [MENTION] 获取信息",
  "coarse_label": 1,
  "coarse_response": "1"
}
```

### 关键特性

- 支持批量处理（默认 batch_size=10）
- 自动处理速率限制（默认请求间隔 2.5 秒）
- 批量处理失败时自动回退到逐条处理
- 支持增量处理（跳过已存在的记录）

### 配置参数

- `batch_size`: 批量处理大小（默认 10）
- `request_interval`: 请求间隔（秒，默认 2.5）
- `max_retries`: 最大重试次数（默认 3）

---

## 步骤 3: subtype_assign.py

### 命令

```bash
python scripts/subtype_assign.py --only-sensitive
```

### 输入格式

**位置：** `data/with_coarse_label.jsonl`（默认）

**格式：** JSONL

**字段：**
- `id`: 字符串
- `text`: 字符串
- `coarse_label`: 整数（0 或 1）
- `coarse_response`: 字符串

**示例：**
```json
{
  "id": "s0",
  "text": "访问 [URL] 联系 [MENTION] 获取信息",
  "coarse_label": 1,
  "coarse_response": "1"
}
```

### 输出格式

**位置：** `data/with_coarse_and_subtypes.jsonl`（默认）

**格式：** JSONL

**字段：**
- `id`: 字符串（保留）
- `text`: 字符串（保留）
- `coarse_label`: 整数（保留）
- `coarse_response`: 字符串（保留）
- `subtype_label`: 列表，子标签列表（仅敏感样本有）
- `subtype_response`: 字符串，LLM 原始响应（仅敏感样本有）

**子标签类型：**
- `porn`: 色情内容
- `politics`: 涉政内容
- `abuse`: 辱骂/攻击性内容
- `other`: 其他敏感内容

**示例（敏感样本）：**
```json
{
  "id": "s0",
  "text": "访问 [URL] 联系 [MENTION] 获取信息",
  "coarse_label": 1,
  "coarse_response": "1",
  "subtype_label": ["porn", "abuse"],
  "subtype_response": "[\"porn\", \"abuse\"]"
}
```

**示例（非敏感样本）：**
```json
{
  "id": "s1",
  "text": "这是一段正常文本",
  "coarse_label": 0,
  "coarse_response": "0",
  "subtype_label": [],
  "subtype_response": ""
}
```

### 关键特性

- `--only-sensitive`: 只对敏感样本（coarse_label=1）调用 LLM 打子标签
- 支持批量处理（默认 batch_size=10）
- 自动合并子标签回原始数据
- 非敏感样本的 `subtype_label` 为空列表 `[]`

### 配置参数

- `--only-sensitive`: 只处理敏感样本（推荐）
- `batch_size`: 批量处理大小（默认 10）
- `request_interval`: 请求间隔（秒，默认 2.5）

---

## 完整链路示例

### 一次性运行所有步骤

```bash
# 步骤 0: TXT 转 JSONL
python scripts/txt_to_jsonl.py --unprocessed-dir data/unprocessed --output data/dataset_raw.jsonl

# 步骤 1: 数据清洗
python scripts/data_cleaner.py

# 步骤 2: 粗粒度标签打标
python scripts/coarse_label.py

# 步骤 3: 子标签打标
python scripts/subtype_assign.py --only-sensitive
```

### 预期输出

**最终文件：** `data/with_coarse_and_subtypes.jsonl`

**包含字段：**
- `id`: 唯一标识符
- `text`: 清洗后的文本
- `coarse_label`: 粗粒度标签（0/1）
- `coarse_response`: LLM 原始响应
- `subtype_label`: 子标签列表（仅敏感样本）
- `subtype_response`: LLM 原始响应（仅敏感样本）

---

## 数据流转图

```
unprocessed/*.txt
    ↓ [txt_to_jsonl.py]
dataset_raw.jsonl (id, text)
    ↓ [data_cleaner.py]
cleaned_raw.jsonl (id, text)  # URL→[URL], @→[MENTION]
    ↓ [coarse_label.py]
with_coarse_label.jsonl (id, text, coarse_label, coarse_response)
    ↓ [subtype_assign.py --only-sensitive]
with_coarse_and_subtypes.jsonl (id, text, coarse_label, coarse_response, subtype_label, subtype_response)
```

---

## 常见问题

### 1. API 速率限制

如果遇到速率限制错误，可以：
- 增加 `request_interval`（默认 2.5 秒）
- 减小 `batch_size`（默认 10）

### 2. 批量处理失败

批量处理失败时会自动回退到逐条处理，不影响数据完整性。

### 3. 增量处理

所有脚本都支持增量处理：
- `txt_to_jsonl.py`: 自动检测最大 ID 并续接
- `coarse_label.py`: 默认跳过已存在的记录
- `subtype_assign.py`: 默认跳过已存在的记录

### 4. 自定义路径

所有脚本都支持自定义输入输出路径：

```bash
# 示例：自定义输入输出
python scripts/data_cleaner.py --input data/custom_input.jsonl --output data/custom_output.jsonl
python scripts/coarse_label.py --input data/custom_input.jsonl --output data/custom_output.jsonl
python scripts/subtype_assign.py --input data/custom_input.jsonl --output data/custom_output.jsonl --only-sensitive
```

---

## 性能说明

### 批量处理优势

- **API 调用次数减少 90%**：10 条数据 = 1 次调用（原来 10 次）
- **处理速度提升约 10 倍**：1000 条数据约需 4.2 分钟（原来 42 分钟）

### 速率限制

- **RPM**: 30 请求/分钟
- **TPM**: 15,000 tokens/分钟
- **RPD**: 14,400 请求/天
- **默认请求间隔**: 2.5 秒（确保不超过 RPM 限制）

---

## 配置说明

所有配置在 `configs/config.yaml` 中：

```yaml
llm:
  batch_size: 10          # 批量处理大小
  max_retries: 3          # 最大重试次数
  request_interval: 2.5   # 请求间隔（秒）
```

可以通过命令行参数覆盖：

```bash
python scripts/coarse_label.py --batch-size 5 --max-retries 5
```

