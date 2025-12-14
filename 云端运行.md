# Day9 云端运行命令

## 1. 激活环境

```bash
cd /mnt/workspace/EMG-PKRI
source venv/bin/activate
```

## 2. Git拉取最新代码

```bash
cd /mnt/workspace/EMG-PKRI
git pull origin main
```

## 3. 检查前置文件

使用 Python 命令检查前置文件是否存在：

```bash
python3 << 'EOF'
import os

print("检查 Day9 前置文件...")
print("=" * 60)

files_to_check = [
    ("output/test_with_uncertainty.jsonl", "Baseline 预测结果（Day4输出）"),
    ("data/q0_test.jsonl", "q₀ 后验（Day6输出）"),
    ("output/alpha_u_lut.json", "α(u) 查表（Day8输出）")
]

all_exist = True
for file_path, description in files_to_check:
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        with open(file_path, 'r') as f:
            if file_path.endswith('.jsonl'):
                count = sum(1 for _ in f)
                print(f"✓ {file_path} 存在 ({description})")
                print(f"  大小: {size/1024:.2f} KB, 行数: {count}")
            else:
                import json
                data = json.load(f)
                if 'u' in data and 'alpha' in data:
                    print(f"✓ {file_path} 存在 ({description})")
                    print(f"  大小: {size/1024:.2f} KB, 查表点数: {len(data['u'])}")
                else:
                    print(f"✓ {file_path} 存在 ({description})")
                    print(f"  大小: {size/1024:.2f} KB")
    else:
        print(f"✗ {file_path} 不存在 ({description})")
        all_exist = False

print("=" * 60)
if all_exist:
    print("✓ 所有前置文件都已就绪，可以运行 Day9")
else:
    print("✗ 部分前置文件缺失，请先运行前置任务")
EOF
```

或者使用简单的 bash 命令检查：

```bash
# 检查文件是否存在
echo "检查前置文件..."
echo "test_with_uncertainty.jsonl: $([ -f output/test_with_uncertainty.jsonl ] && echo '✓ 存在' || echo '✗ 不存在')"
echo "q0_test.jsonl: $([ -f data/q0_test.jsonl ] && echo '✓ 存在' || echo '✗ 不存在')"
echo "alpha_u_lut.json: $([ -f output/alpha_u_lut.json ] && echo '✓ 存在' || echo '✗ 不存在')"

# 显示文件大小和行数
if [ -f output/test_with_uncertainty.jsonl ]; then
    echo "test_with_uncertainty.jsonl: $(wc -l < output/test_with_uncertainty.jsonl) 行, $(du -h output/test_with_uncertainty.jsonl | cut -f1)"
fi
if [ -f data/q0_test.jsonl ]; then
    echo "q0_test.jsonl: $(wc -l < data/q0_test.jsonl) 行, $(du -h data/q0_test.jsonl | cut -f1)"
fi
if [ -f output/alpha_u_lut.json ]; then
    echo "alpha_u_lut.json: $(du -h output/alpha_u_lut.json | cut -f1)"
fi
```

## 4. 运行 Day9

### 基本运行（使用默认配置）

```bash
cd /mnt/workspace/EMG-PKRI
source venv/bin/activate
python scripts/eval_emg.py
```

### 指定文件路径运行

```bash
cd /mnt/workspace/EMG-PKRI
source venv/bin/activate
python scripts/eval_emg.py \
    --baseline-file output/test_with_uncertainty.jsonl \
    --q0-file data/q0_test.jsonl \
    --alpha-lut-file output/alpha_u_lut.json \
    --output-dir output
```

### 指定固定 α 值运行

```bash
python scripts/eval_emg.py --fixed-alpha 0.5
```

### 同时评估困难集（如果存在）

```bash
python scripts/eval_emg.py \
    --baseline-file output/test_with_uncertainty.jsonl \
    --q0-file data/q0_test.jsonl \
    --alpha-lut-file output/alpha_u_lut.json \
    --hard-file output/hard_with_uncertainty.jsonl \
    --hard-q0-file data/q0_hard.jsonl \
    --output-dir output
```

## 5. 一键运行命令（组合）

```bash
cd /mnt/workspace/EMG-PKRI && \
source venv/bin/activate && \
git pull origin main && \
python3 << 'EOF'
import os
files = [
    "output/test_with_uncertainty.jsonl",
    "data/q0_test.jsonl", 
    "output/alpha_u_lut.json"
]
missing = [f for f in files if not os.path.exists(f)]
if missing:
    print(f"✗ 缺失文件: {missing}")
    exit(1)
else:
    print("✓ 所有前置文件就绪")
EOF
python scripts/eval_emg.py --output-dir output
```

## 6. 验证输出

运行完成后，检查输出文件：

```bash
# 检查输出文件
ls -lh output/metrics_emg.json
ls -lh output/emg_comparison_table.csv
ls -lh output/emg_comparison_charts.png

# 查看指标摘要
python3 << 'EOF'
import json
with open('output/metrics_emg.json', 'r') as f:
    metrics = json.load(f)
    
if 'test_set' in metrics:
    print("Test Set 指标:")
    for method, m in metrics['test_set'].items():
        print(f"  {method}: F1={m['f1']:.4f}, NLL={m['nll']:.4f}, ECE={m['ece']:.4f}")
        
if 'comparison' in metrics:
    print("\n对比分析:")
    if 'emg_vs_baseline' in metrics['comparison']:
        comp = metrics['comparison']['emg_vs_baseline']
        print(f"  EMG vs Baseline: F1提升 {comp['f1_improvement_percent']:+.2f}%, NLL降低 {comp['nll_reduction_percent']:+.2f}%")
EOF
```

## 注意事项

1. **确保前置任务完成**：
   - Day4: 生成 `output/test_with_uncertainty.jsonl`
   - Day6: 生成 `data/q0_test.jsonl`
   - Day8: 生成 `output/alpha_u_lut.json`

2. **如果文件不存在**：
   - `test_with_uncertainty.jsonl`: 运行 `uncertainty_analysis.py --test-file test.jsonl`
   - `q0_test.jsonl`: 运行 `q0_builder.py --datasets test`
   - `alpha_u_lut.json`: 运行 Day8 的 `emg_fit_alpha_u.py`

3. **运行时间**：评估脚本运行很快（通常 < 1 分钟），因为不需要模型推理，只是数据融合和指标计算。

