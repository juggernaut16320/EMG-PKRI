# 本地运行Day8和Day9指南

## 📋 前置文件列表

### Day8需要
- `output/bucket_alpha_star.csv`（Day7的输出）

### Day9需要
- `output/test_with_uncertainty.jsonl`（Baseline预测结果）
- `data/q0_test.jsonl`（q0后验）
- `output/alpha_u_lut.json`（Day8的输出，Day8会生成）

---

## 🔽 步骤1：云端提交文件到Git

在云端执行：

```bash
cd /mnt/workspace/EMG-PKRI
chmod +x 云端提交Day8Day9前置文件.sh
./云端提交Day8Day9前置文件.sh
```

或者手动提交：

```bash
cd /mnt/workspace/EMG-PKRI
git add output/bucket_alpha_star.csv
git add output/test_with_uncertainty.jsonl
git add data/q0_test.jsonl
git commit -m "添加Day8和Day9前置文件（供本地运行）"
git push origin main
```

---

## 🔽 步骤2：本地拉取文件

在本地执行：

```powershell
# Windows PowerShell
cd C:\develop
git pull origin main
```

---

## ✅ 步骤3：本地环境准备

### 检查依赖

```powershell
# 检查Python和依赖
python -c "import numpy, pandas, matplotlib, yaml, json; print('✓ 所有依赖已安装')"
```

如果缺少依赖：

```powershell
pip install numpy pandas matplotlib pyyaml scikit-learn
```

---

## 🚀 步骤4：本地运行Day8

### Windows PowerShell

```powershell
# 确保output目录存在
New-Item -ItemType Directory -Force -Path output | Out-Null

# 运行Day8
python scripts/emg_fit_alpha_u.py `
    --input-file output/bucket_alpha_star.csv `
    --output-dir output
```

### 验证Day8输出

```powershell
# 检查输出文件
Test-Path output/alpha_u_lut.json
Test-Path output/alpha_u_curve.png
```

---

## 🚀 步骤5：本地运行Day9

### Windows PowerShell

```powershell
# 运行Day9
python scripts/eval_emg.py `
    --baseline-file output/test_with_uncertainty.jsonl `
    --q0-file data/q0_test.jsonl `
    --alpha-lut-file output/alpha_u_lut.json `
    --use-consistency-gating `
    --output-dir output
```

### 验证Day9输出

```powershell
# 检查输出文件
Test-Path output/metrics_emg_q0.json
Test-Path output/emg_comparison_charts_q0.png
Test-Path output/emg_comparison_table_q0.csv
```

---

## 📤 步骤6：上传结果回云端（可选）

如果需要在云端继续使用结果，可以提交回Git：

```powershell
# 添加Day8输出
git add output/alpha_u_lut.json
git add output/alpha_u_curve.png

# 添加Day9输出
git add output/metrics_emg_q0.json
git add output/emg_comparison_charts_q0.png
git add output/emg_comparison_table_q0.csv

# 提交
git commit -m "本地运行Day8和Day9结果"
git push origin main
```

---

## 📝 一键运行脚本（PowerShell）

创建 `run_day8_day9_local.ps1`：

```powershell
# 确保在项目根目录
cd C:\develop

# 拉取最新代码
git pull origin main

# 确保目录存在
New-Item -ItemType Directory -Force -Path output | Out-Null
New-Item -ItemType Directory -Force -Path data | Out-Null

# 运行Day8
Write-Host "运行Day8..." -ForegroundColor Yellow
python scripts/emg_fit_alpha_u.py --input-file output/bucket_alpha_star.csv --output-dir output

if ($LASTEXITCODE -ne 0) {
    Write-Host "Day8失败" -ForegroundColor Red
    exit 1
}

# 运行Day9
Write-Host "运行Day9..." -ForegroundColor Yellow
python scripts/eval_emg.py `
    --baseline-file output/test_with_uncertainty.jsonl `
    --q0-file data/q0_test.jsonl `
    --alpha-lut-file output/alpha_u_lut.json `
    --use-consistency-gating `
    --output-dir output

if ($LASTEXITCODE -ne 0) {
    Write-Host "Day9失败" -ForegroundColor Red
    exit 1
}

Write-Host "完成！" -ForegroundColor Green
```

---

## ⚠️ 注意事项

1. **文件大小**：
   - `test_with_uncertainty.jsonl` 可能较大（~4-5MB）
   - `q0_test.jsonl` 可能较大（~1-2MB）
   - 确保Git仓库可以容纳这些文件

2. **如果文件太大**：
   - 可以考虑使用Git LFS
   - 或者只提交小文件，大文件使用其他方式传输

3. **确保云端文件存在**：
   - 在提交前检查文件是否真的存在
   - Day7需要先完成，才能有 `bucket_alpha_star.csv`

---

## ✅ 快速检查清单

- [ ] 云端：Day7已完成，生成 `bucket_alpha_star.csv`
- [ ] 云端：Day4已完成，生成 `test_with_uncertainty.jsonl`
- [ ] 云端：Day6已完成，生成 `q0_test.jsonl`
- [ ] 云端：提交文件到Git
- [ ] 本地：拉取最新代码
- [ ] 本地：安装依赖（numpy, pandas, matplotlib, pyyaml）
- [ ] 本地：运行Day8
- [ ] 本地：运行Day9
- [ ] 本地：检查输出文件
- [ ] （可选）本地：提交结果回Git

