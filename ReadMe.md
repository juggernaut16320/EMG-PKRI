# 待解决问题（工程视角）

**工程约定**：

1. **Conda 环境 + requirements**
   - 用 `conda create -n emgpkri python=3.10` 建环境；
   - **每次开发前务必激活环境**：`conda activate emgpkri`
   - 所有依赖写在 `requirements.txt`，本地 & 云端统一。
2. **代码容器化友好**
   - 所有脚本：
     - 只用 **相对路径**；
     - 统一从 `config.yaml` / 命令行参数读 `data_dir`、模型名等；
     - API Key 用环境变量（不写死在代码里）。
   - 这样以后写 Dockerfile / 上阿里云，只需要：
     - `pip install -r requirements.txt`
     - 挂载 `/app/data`
     - 执行脚本命令即可。
3. **Gemini API 调用**
   - 在 `.env` / 云平台环境变量里配置：
     - `GEMINI_API_KEY`（必需）
   - `llm_labeler.py` 里统一封装 `call_llm_backend`，当前锁定使用 `gemma-3-27b` 模型。
4. **阿里云部署路径打通**
   - 仓库结构、`requirements.txt`、`config.yaml`、相对路径 → 直接适配「自定义镜像 + 命令行」风格的部署（例如：阿里云 ECI / 容器服务）。

下面所有 Day1–12 的设计，都默认满足以上约定。

------

# 论文核心思想

> **利用模型不确定性 (u) 驱动的单调门控机制（EMG），在“主模型后验 (p(c\mid x))”和“知识后验 (q(c\mid z))”之间动态加权：不确定性高时多信知识，不确定性低时多信模型；同时通过 PKRI 对知识边做“可信度建模”，把外部敏感词 / alias 变成“可校准的概率后验”，在端侧实现一致、可控、可解释的融合，从而降低高置信错误，提高敏感文本识别的可靠性。**

> **两周实验主线 = 数据 ➝ coarse baseline ➝ 不确定性 u ➝ EMG ➝ 轻量 PKRI ➝ EMG+PKRI 融合形成最小闭环**，后续再在这个框架上扩子标签、扩数据。

# 全局工程约定（容器化 & 增量友好）

项目根目录建议固定为：

```text
project/
│
├── data/                  # 本地 & 云端挂载点
├── scripts/               # 所有 *.py
├── tests/                 # 单元测试
├── configs/
│   └── config.yaml        # 路径、模型名、后端等统一配置
├── requirements.txt
└── .env.example           # API key 示例（不提交真实密钥）
```

## 快速开始

```bash
# 1. 激活 conda 环境（必须）
conda activate emgpkri

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量（复制 .env.example 为 .env 并填入 API Key）
cp .env.example .env

# 4. 运行测试
python -m pytest tests/ -v
```

通用约定：

- **路径统一：**

  - 所有脚本从 `config.yaml` 或 CLI 参数读取 `data_dir` / `output_dir`：

    ```python
    # 伪代码
    cfg = yaml.safe_load(open("configs/config.yaml"))
    data_dir = args.data_dir or cfg["data_dir"]
    input_path = os.path.join(data_dir, "dataset_raw.jsonl")
    ```

- **API 不写死：**

  - 使用 `os.getenv("GEMINI_API_KEY")` 等方式；
  - `.env.example` 给出变量名，实际 `.env` 不进 git。

- **增量数据场景：**

  - 所有脚本只依赖「输入文件」而不是某个一次性的中间状态；
  - 只要把「旧数据 + 新数据」合并为一个新的 `with_*.jsonl`，就可以**重跑对应 Day 的脚本**，不改代码。

------

# Day 1–12（加入容器化 & 增量训练友好说明）

------

## Day 1｜数据清洗 + 主/子标签 + 划分 + 困难子集

**目标：** 固定任务定义和数据形态，产出后续所有实验统一使用的数据基座。

**工作内容：**

- **TXT 转 JSONL**：将 `unprocessed` 目录下的多个 txt 文件转换为 JSONL 格式，使用 `s0`, `s1`, `s2` 等格式的 ID
- 清洗 `dataset_raw.jsonl`：去重、空文本、乱码、极短文本、明显噪声。
- **主任务标签（coarse）：敏感 / 非敏感**
  - 使用 `gemma-3-27b` 模型打 `label ∈ {0,1}`（non_sensitive / sensitive）。
- **子标签（subtypes，多标签）：**
  - 体系：`["porn", "politics", "abuse", "other"]`（可多选）。
  - 只作为**附加信息**和后续知识/分析使用，**不进 Day2 训练 loss**。
- 按 8/1/1 做 `train/dev/test` 划分（stratify=coarse label）。
- 基于模型预测分歧构造 `hard_eval_set`（500–2000 条）：
  - 使用 `gemma-3-27b` 模型进行打标；
  - 抽"高置信但可能错误的样本"等难例。

**工程/容器化约定：**

- 所有脚本从 `configs/config.yaml` 读取：

  - `data_dir`（默认 `./data`，云端挂载 `/app/data` 也只改 config）；
  - `batch_size`、`max_retries` 等参数。

- LLM 模型：

  - 当前锁定使用 `gemma-3-27b-it`（30 RPM，14.4K RPD，15K TPM，适合大批量打标）。
  - 支持批量处理（默认 batch_size=10），将多条数据打包传给大模型，大幅减少 API 调用次数。
  - 自动处理速率限制（默认请求间隔 2.5 秒）。

- 所有输入输出文件路径写成：

  ```python
  input_path = os.path.join(data_dir, "dataset_raw.jsonl")
  ```

- API key 全部由环境变量提供（不写进代码 / config）。

**增量数据说明：**

- 后续新增数据时，只需：
  1. 将新的 txt 文件放入 `data/unprocessed` 目录，运行 `txt_to_jsonl.py` 自动追加到 `dataset_raw.jsonl`（ID 自动续接）
  2. 对新 raw 数据再跑一遍 `data_cleaner.py`、`coarse_label.py`、`subtype_assign.py`；
  3. 把旧 `with_coarse_and_subtypes.jsonl` 与新数据合并；
  4. 重新跑 `dataset_split.py` & `hardset_maker.py`。
- 代码逻辑不用改，**Day1 全部脚本设计成"对输入文件无状态处理"**，方便反复重跑。

------

## Day 2｜训练主模型 coarse baseline（Qwen + LoRA）

**目标：** 拿到一个稳定的二分类基线模型。

**工作内容：**

- 使用 `train.jsonl` 上的 coarse label（敏感 / 非敏感）做 LoRA 微调：
  - 模型：Qwen1.7B；
  - 任务：只做二分类，不训练子类头。
- 在 dev 上监控 loss / F1，保存最优 checkpoint。

**新增工程说明：**

- 增加脚本：`scripts/baseline_train.py`，满足：
  - 从 `config.yaml` 读取：
    - `model_name_or_path`（如 `Qwen/Qwen1.5-1.8B`）；
    - LoRA 超参（r、alpha、lr、batch_size、epochs）；
    - `train_file/dev_file`。
  - 支持参数 `--resume_from_checkpoint`：
    - 未来新增数据时，可以在旧 LoRA 基础上增量再训练几轮；
    - 或者用旧 LoRA 作为 warm start，在“旧+新数据合并”上完整重训。
- 输出目录约定：
  - `checkpoints/baseline-lora/` 下保存：
    - `adapter_model.bin`（LoRA 权重）
    - `config.json`、`training_args.bin` 等
  - 方便云端和本地统一使用。

**增量训练说明：**

- 新数据加入后，只需要：
  1. 重新生成新的 `train/dev/test`；
  2. 重新跑 `baseline_train.py`：
     - 要么 `--resume_from_checkpoint` 增量微调；
     - 要么重新训练（warm start，仍然很快）。
- 后续 Day3–Day12 都只依赖最新的 baseline 概率输出 `p(c|x)`，无需改代码。

------

## Day 3｜基线评估 + 高置信错误分析

**目标：** 了解 baseline 的错误分布，特别是“高置信错误”。

**工作内容：**

- 在 `test.jsonl` 上评估 baseline：Accuracy / F1 / confusion matrix。
- 在 `hard_eval_set.jsonl` 上单独评估，比较 hard vs 普通样本表现。
- 选出一批高置信错误样本（如预测概率>0.8但错了），存成样本库。

**工程说明：**

- 新增脚本：`scripts/eval_baseline.py`：
  - 从 `config.yaml` 读取：
    - `baseline_checkpoint_path`
    - `data_dir` 和 `test_file` / `hard_file`
  - 输出：
    - `metrics_baseline.json`
    - `high_conf_error_samples.jsonl`

**增量说明：**

- 每次 baseline 更新或数据扩展后：
  - 重新跑 `eval_baseline.py` 即可获得最新错误分布；
  - 其它代码不改。

------

## Day 4｜不确定性指标 u 构建 + 分桶分析

**目标：** 证明不确定性 u 与错误率有相关性，为 EMG 提供依据。

**工作内容：**

- 对 dev/test 的每条样本，计算：
  - `u = 1 - max_c pθ(c|x)`。
- 按 u 分桶（10 桶），统计：
  - 样本数、错误率、平均置信度等。

**工程说明：**

- 新增脚本：`scripts/uncertainty_analysis.py`：
  - 输入：baseline 在 dev/test 上的预测结果（含 logits/probs）。
  - 输出：
    - `uncertainty_buckets.csv`
    - `u_vs_error.png`
- 为未来扩展预留简单接口：
  - 支持 `--metric u_max` / `u_entropy` / `u_margin`，方便后续扩展方向 1。

**增量说明：**

- 新 baseline、或新数据时：
  - 只需重新生成预测结果 → 再跑一次本脚本即可。

------

## Day 5｜校准分析：Reliability Diagram + ECE

**目标：** 量化 baseline 的“过度自信”问题，为门控/校准提供动机。

**工作内容：**

- 在 dev/test 上绘制 Reliability Diagram；
- 计算 ECE/MCE；
- 写出简要小结。

**工程说明：**

- 在 `uncertainty_analysis.py` 中或单独脚本 `calibration_analysis.py`：
  - 输入：`p(c|x)` 和真实标签；
  - 输出：
    - `reliability_diagram.png`
    - `ece_results.json`

**增量说明：**

- 一旦 baseline 更新，只要重新算 `p(c|x)`，校准分析脚本直接复用。

------

## Day 6｜构造规则版知识后验 q₀(c|z)（coarse 二分类）

**目标：** 生成一个启发式知识后验 q₀，为 EMG 提供“知识通道”的第一版信号。

**工作内容：**

- 基于色情/涉政/辱骂等词表、alias、正则规则构造知识信号：
  - 输出 coarse 二分类：[p_non_sensitive, p_sensitive]；
  - 内部可按 porn/politics/abuse 设计不同强度权重。
- 对 train/dev/test/hardset 全部生成 q₀ 后验。

**工程说明：**

- 新增脚本：`scripts/q0_builder.py`：
  - 输入：语料 + 子标签 / 词表文件（在 `configs/lexicons/` 下）；
  - 输出：
    - `q0_train.jsonl`
    - `q0_dev.jsonl`
    - `q0_test.jsonl`
    - `q0_hard.jsonl`
- 轻量扩展预留：
  - 词频/PMI、词密度、embedding cosine 的计算可以在这里顺手实现。

**增量说明：**

- 当你扩展词表或数据时：
  - 只需重跑 `q0_builder.py` 即可；
  - 不影响 EMG/PKRI 代码结构。

------

## Day 7｜EMG 分桶 α 搜索

**目标：** 找到“在不同不确定性水平下，模型 vs 知识”的最佳融合权重 α。

**工作内容：**

- 对每个 u bucket，枚举 α ∈ {0, 0.25, 0.5, 0.75, 1}；
- 在 dev 集上算 NLL / F1，选出每个 bucket 的 α*。

**工程说明：**

- 新增脚本：`scripts/emg_bucket_search.py`：
  - 输入：
    - baseline 的 `p(c|x)`；
    - q₀ 的 `q(c|z)`；
    - u 分桶信息。
  - 输出：
    - `bucket_alpha_star.csv`
- 预留扩展：
  - 支持 `--bucket_type`（等宽 / 等量）；
  - 支持更细粒度 α 网格（0.1 步长）。

**增量说明：**

- baseline 或 q₀ 更新后：
  - 重新跑一次 `emg_bucket_search.py`，得到新的 α*，即可支持新分布。

------

## Day 8｜PAV 保序回归生成单调 α(u)

**目标：** 从离散 α* 拟合单调递增 α(u) 门控函数。

**工作内容：**

- 用 PAV 对 (u_bucket, α*) 做保序回归；
- 输出 α(u) 查表 / 函数。

**工程说明：**

- 新增脚本：`scripts/emg_fit_alpha_u.py`：
  - 输入：`bucket_alpha_star.csv`；
  - 输出：
    - `alpha_u_lut.json`（查表）
    - `alpha_u_curve.png`

**增量说明：**

- 每次数据分布变化 / baseline 更新导致 α* 变化时：
  - 只需重跑这个脚本；
  - EMG 推理逻辑本身不需要改。

------

## Day 9｜EMG 效果验证（p vs 固定融合 vs EMG）

**目标：** 验证 EMG 优于 baseline 与固定 α 融合。

**工作内容：**

- 在 test & hardset 上对比：
  1. baseline；
  2. 固定 α 融合；
  3. EMG（α(u) 自适应）。

**工程说明：**

- 新增脚本：`scripts/eval_emg.py`：
  - 输入：
    - baseline `p(c|x)`；
    - q₀；
    - α(u) 查表；
  - 输出：
    - `metrics_emg.json`
    - 对比表格 / 图（F1/NLL/ECE）。

**增量说明：**

- baseline 或 q₀ 更新，只需完成：
  - 重新算 p / q₀ → 重跑 `emg_bucket_search.py`、`emg_fit_alpha_u.py` → 重跑 `eval_emg.py`。

------

## Day 10｜轻量 PKRI（知识可信度建模）构建

**目标：** 让 q₀ 升级为带可信度建模的 q_PKRI（仍为 coarse 二分类）。

**工作内容：**

- 构建轻量特征（词表命中、embedding cosine、子标签匹配等）；
- 用 LR / LightGBM 训练 PKRI 模型；
- 输出 q_PKRI(c|z)。

**工程说明：**

- 新增脚本：
  - `scripts/pkri_feature_builder.py`
  - `scripts/pkri_train.py`
- 输出：
  - `pkri_features_train.csv` / `pkri_model.pkl`
  - `qpkri_train/dev/test/hard.jsonl`

**增量说明：**

- 数据扩展时：
  - 重跑 feature builder；
  - 用同一脚本 `pkri_train.py` 重训 LR/LightGBM 即可。

------

## Day 11｜EMG + PKRI 联合融合试验

**目标：** 检验用 q_PKRI 替代 q₀ 是否带来收益。

**工作内容：**

- 使用同一 α(u)，对比：
  1. baseline；
  2. EMG + q₀；
  3. EMG + q_PKRI。

**工程说明：**

- 在 `eval_emg.py` 中增加一个模式：
  - `--knowledge_source q0` 或 `--knowledge_source qpkri`；
- 输出：
  - `metrics_emg_q0.json`
  - `metrics_emg_qpkri.json`
  - 汇总对比表。

**增量说明：**

- 一旦 PKRI 重新训练：
  - 直接用新的 q_PKRI 重新跑 `eval_emg.py` 即可，无需调整其它逻辑。

------

## Day 12｜可视化 + 初步结论总结

**目标：** 把前 11 天所有关键结果整理成图表 & 文字小结。

**工作内容：**

- 自动化生成 3–5 张核心图；
- 输出若干对比表；
- 写一份 `initial_experiment_summary.md`。

**工程说明：**

- 新增脚本：`scripts/generate_reports.py`：
  - 汇总各天输出的 metrics / 图表 / CSV；
  - 自动生成 md 摘要骨架（后续再人工润色）。

**增量说明：**

- 每次你扩展数据或新增实验，只要保证各子脚本重新跑完，再执行一次 `generate_reports.py`，就能得到一份新的“完整实验闭环报告”。

------

# 工作量扩展方向

1. **不确定性建模：**
   - `uncertainty_analysis.py` 支持多种 u 指标（max/entropy/margin），方便切换对比。
2. **知识后验 q₀：**
   - `q0_builder.py` 内部可以逐步加入词密度、PMI、embedding cosine 等特征逻辑，而不改变接口。
3. **EMG：**
   - `emg_bucket_search.py` / `emg_fit_alpha_u.py` 支持分桶策略、网格搜索粒度的配置。
4. **PKRI：**
   - `pkri_feature_builder.py` / `pkri_train.py` 可逐步添加新特征或从 LR 换到 LightGBM，而不影响下游脚本。

