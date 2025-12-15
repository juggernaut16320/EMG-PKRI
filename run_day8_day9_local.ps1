# 本地运行Day8和Day9脚本
# 最后更新：2025-12-15

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "本地运行 Day8 和 Day9" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# 步骤1：拉取最新代码
Write-Host "步骤1/3: 拉取最新代码..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray
git pull origin main
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Git拉取失败" -ForegroundColor Red
    exit 1
}
Write-Host "✓ 代码拉取完成" -ForegroundColor Green
Write-Host ""

# 检查必需文件
Write-Host "检查必需文件..." -ForegroundColor Yellow
$requiredFiles = @(
    "output/bucket_alpha_star.csv",
    "output/test_with_uncertainty.jsonl",
    "data/q0_test.jsonl"
)

$allFilesExist = $true
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $file 不存在" -ForegroundColor Red
        $allFilesExist = $false
    }
}

if (-not $allFilesExist) {
    Write-Host ""
    Write-Host "✗ 缺少必需文件，请先确保已拉取最新代码" -ForegroundColor Red
    exit 1
}

Write-Host ""

# 确保目录存在
New-Item -ItemType Directory -Force -Path output | Out-Null
New-Item -ItemType Directory -Force -Path data | Out-Null

# 步骤2：运行Day8
Write-Host "步骤2/3: 运行Day8（PAV拟合）..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray
python scripts/emg_fit_alpha_u.py `
    --input-file output/bucket_alpha_star.csv `
    --output-dir output

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Day8 运行失败" -ForegroundColor Red
    exit 1
}

# 验证Day8输出
if (-not (Test-Path "output/alpha_u_lut.json")) {
    Write-Host "✗ Day8 输出文件缺失: output/alpha_u_lut.json" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Day8 运行完成" -ForegroundColor Green
Write-Host ""

# 步骤3：运行Day9
Write-Host "步骤3/3: 运行Day9（EMG评估）..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray
python scripts/eval_emg.py `
    --baseline-file output/test_with_uncertainty.jsonl `
    --q0-file data/q0_test.jsonl `
    --alpha-lut-file output/alpha_u_lut.json `
    --use-consistency-gating `
    --output-dir output

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Day9 运行失败" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Day9 运行完成" -ForegroundColor Green
Write-Host ""

# 验证输出文件
Write-Host "验证输出文件..." -ForegroundColor Yellow
$outputFiles = @(
    "output/alpha_u_lut.json",
    "output/alpha_u_curve.png",
    "output/metrics_emg_q0.json",
    "output/emg_comparison_charts_q0.png",
    "output/emg_comparison_table_q0.csv"
)

foreach ($file in $outputFiles) {
    if (Test-Path $file) {
        Write-Host "  ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ $file 不存在（可能正常，部分文件可选）" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "✓ Day8 和 Day9 运行完成！" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "查看结果：" -ForegroundColor Yellow
Write-Host "  - Day8结果: output/alpha_u_lut.json, output/alpha_u_curve.png" -ForegroundColor Gray
Write-Host "  - Day9结果: output/metrics_emg_q0.json, output/emg_comparison_charts_q0.png" -ForegroundColor Gray
Write-Host ""

