### Day8 要完成的代码（PAV 保序回归生成单调 α(u)）

> **环境准备**：开始前请确保已激活 conda 环境：`conda activate emgpkri`（或激活 venv：`source venv/bin/activate`）

```text
project/
│
├── output/
│   ├── bucket_alpha_star.csv    # Day7 输出（输入）
│   ├── alpha_u_lut.json         # 输出：α(u) 查表
│   └── alpha_u_curve.png        # 输出：α(u) 曲线图
│
├── configs/
│   └── config.yaml
│
├── scripts/
│   └── emg_fit_alpha_u.py
│
└── requirements.txt
```

### `emg_fit_alpha_u.py`（PAV 保序回归生成单调 α(u)）

**功能：**
- 从离散的 α* 拟合单调递减的 α(u) 函数
- 使用 PAV（Pool Adjacent Violators）保序回归算法
- 生成 α(u) 查表（JSON格式）
- 绘制 α(u) 曲线图

**关键特性：**

- **PAV 保序回归**：
  - 确保 α(u) 是单调递减的（不确定性越高，α越小，越信任知识）
  - 如果输入数据违反单调性，PAV会将其调整为单调的
  - 使用 scipy.stats.isotonic_regression（如果可用），否则使用自定义实现

- **α(u) 物理意义**：
  - u 越大（不确定性高）→ α 越小 → 越信任知识 q₀
  - u 越小（不确定性低）→ α 越大 → 越信任模型 p(c|x)

- **查表生成**：
  - 默认生成 100 个均匀分布的 u 值
  - 使用线性插值计算对应的 α 值
  - 确保 α 在 [0, 1] 范围内

**使用方式：**

```bash
# 基本使用（从 config.yaml 读取所有参数）
python scripts/emg_fit_alpha_u.py

# 指定输入文件
python scripts/emg_fit_alpha_u.py \
    --input-file output/bucket_alpha_star.csv

# 指定输出目录
python scripts/emg_fit_alpha_u.py \
    --output-dir output

# 自定义查表点数
python scripts/emg_fit_alpha_u.py \
    --lut-points 200

# 强制使用自定义实现（不使用 scipy）
python scripts/emg_fit_alpha_u.py \
    --no-use-scipy
```

**输入格式：**

**位置：** `output/bucket_alpha_star.csv`（Day7 输出）

**必需字段：**
- `bucket_id`: bucket ID
- `u_mean`: 不确定性均值
- `alpha_star`: 最优 α* 值

**格式：**
```csv
bucket_id,u_min,u_max,u_mean,n_samples,alpha_star,f1_at_alpha_star,nll_at_alpha_star,...
0,0.0,0.1,0.05,7825,0.75,0.9123,0.2345,...
1,0.1,0.2,0.15,758,0.6,0.87,0.28,...
...
```

**输出格式：**

**1. 查表文件：** `output/alpha_u_lut.json`

**格式：**
```json
{
  "u": [0.05, 0.0571, 0.0641, ...],
  "alpha": [0.75, 0.7394, 0.7288, ...]
}
```

**2. 曲线图：** `output/alpha_u_curve.png`

- 显示原始数据点（蓝色散点）
- 显示 PAV 拟合曲线（红色线）
- 包含说明文字

**配置参数（config.yaml）：**

```yaml
# 输出目录配置
output_dir: "./output"
```

**增量分析说明：**

- **Day7 输出更新后：**
  - 重新运行 `emg_fit_alpha_u.py` 即可获得新的 α(u) 函数
  - 无需修改代码

**工程/容器化约定：**

- 所有路径从 `config.yaml` 或 CLI 参数读取
- 支持 scipy 和自定义实现两种方式
- 输出格式统一（JSON + PNG）

**注意事项：**

- **单调性保证**：
  - PAV 确保输出的 α(u) 是单调递减的
  - 如果输入数据不符合单调递减，PAV 会将其调整为单调的

- **查表使用**：
  - Day9 的 EMG 评估会使用这个查表
  - 对于给定的 u，使用线性插值找到对应的 α

- **与 Day7 的关系：**
  - Day7 输出离散的 α*（每个 bucket 一个值）
  - Day8 将其拟合为连续的单调递减函数 α(u)
  - 为 Day9 的 EMG 评估提供门控函数

---

## Day8 实现清单

### 核心功能实现

- [x] **emg_fit_alpha_u.py**
  - [x] PAV 保序回归算法（自定义实现）
  - [x] 支持 scipy.stats.isotonic_regression（可选）
  - [x] 单调递减 α(u) 拟合
  - [x] 查表生成（JSON格式）
  - [x] 曲线图绘制

### 测试文件

- [x] **test_emg_fit_alpha_u.py**
  - [x] 测试 PAV 回归算法
  - [x] 测试 α(u) 拟合逻辑
  - [x] 测试查表生成
  - [x] 测试边界情况
  - [x] 端到端集成测试

### 文档和示例

- [x] **Day8.md 文档**（当前文档）

---

## PAV 保序回归核心算法

### 算法原理

PAV（Pool Adjacent Violators）保序回归是一种非参数回归方法，用于拟合单调函数。

**对于单调递减的情况：**
1. 对 -α 做单调递增回归
2. 结果取负得到单调递减的 α

**算法步骤：**
1. 按 u 值排序数据
2. 对 -α 应用 PAV 算法（合并违反单调性的相邻组）
3. 取负得到最终的 α(u)

### 查表生成

使用线性插值在 u 的整个范围内生成均匀分布的查表点。

---

## 预期输出示例

### 查表示例

```json
{
  "u": [0.05, 0.0571, 0.0641, ...],
  "alpha": [0.75, 0.7394, 0.7288, ...]
}
```

**解读：**
- u=0.05（低不确定性）→ α=0.75（较信任模型）
- u=0.75（高不确定性）→ α=0.10（较信任知识）

### 曲线图

图表显示：
- 原始 α* 数据点（每个 bucket）
- PAV 拟合的单调递减曲线
- 符合 EMG 理论预期

---

## 与 Day7 的对比

| 特性 | Day7（α 搜索） | Day8（PAV 拟合） |
|------|--------------|----------------|
| **输入** | dev 集 + q₀ + 不确定性分桶 | bucket_alpha_star.csv |
| **输出** | 离散的 α*（每个bucket一个） | 连续的 α(u) 函数 |
| **用途** | 找到每个 bucket 的最优 α | 生成平滑的门控函数 |
| **关系** | 为 Day8 提供数据 | 为 Day9 提供查表 |

---

**最后更新**：2025-12-14  
**状态**：✅ 已完成

**完成情况总结：**
- ✅ 核心功能已实现：`emg_fit_alpha_u.py` 脚本已完成
- ✅ PAV 保序回归算法已实现（支持 scipy 和自定义实现）
- ✅ 查表生成和曲线图绘制功能正常
- ✅ 所有单元测试通过（11个测试用例）
- ✅ 模拟数据测试通过，输出格式正确

