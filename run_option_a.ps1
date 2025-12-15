# 选项A执行脚本：应用最优参数并验证EMG效果（PowerShell版本）
# 最后更新：2025-12-15

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "执行选项A：应用最优参数并验证EMG效果" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# 步骤1：重新生成q0（使用最优参数）
Write-Host "步骤1/5: 重新生成q0（使用最优参数）..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray
python scripts/q0_builder.py --datasets train dev test
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ q0生成失败" -ForegroundColor Red
    exit 1
}
Write-Host "✓ q0重新生成完成" -ForegroundColor Green
Write-Host ""

# 步骤2：评估q0质量（验证优化效果）
Write-Host "步骤2/5: 评估q0质量（验证优化效果）..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray
python scripts/eval_q0.py `
    --q0-file data/q0_dev.jsonl `
    --baseline-file output/dev_with_uncertainty.jsonl
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ q0质量评估失败" -ForegroundColor Red
    exit 1
}
Write-Host "✓ q0质量评估完成" -ForegroundColor Green
Write-Host ""

# 步骤3：重新运行Day7（α*搜索）
Write-Host "步骤3/5: 重新运行Day7（α*搜索）..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray
python scripts/emg_bucket_search.py `
    --dev-file output/dev_with_uncertainty.jsonl `
    --q0-file data/q0_dev.jsonl `
    --uncertainty-file output/dev_uncertainty_buckets.csv `
    --output-file output/bucket_alpha_star.csv
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Day7 α*搜索失败" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Day7 α*搜索完成" -ForegroundColor Green
Write-Host ""

# 步骤4：重新运行Day8（α(u)拟合）
Write-Host "步骤4/5: 重新运行Day8（α(u)拟合）..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray
python scripts/emg_fit_alpha_u.py `
    --input-file output/bucket_alpha_star.csv `
    --output-dir output
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Day8 α(u)拟合失败" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Day8 α(u)拟合完成" -ForegroundColor Green
Write-Host ""

# 步骤5：重新运行Day9（EMG评估）
Write-Host "步骤5/5: 重新运行Day9（EMG评估）..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray
python scripts/eval_emg.py `
    --baseline-file output/test_with_uncertainty.jsonl `
    --q0-file data/q0_test.jsonl `
    --alpha-lut-file output/alpha_u_lut.json `
    --use-consistency-gating `
    --output-dir output
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Day9 EMG评估失败" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Day9 EMG评估完成" -ForegroundColor Green
Write-Host ""

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "✓ 选项A执行完成！" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "查看结果：" -ForegroundColor Yellow
Write-Host "  - q0质量: 查看步骤2的输出"
Write-Host "  - Day7结果: output/bucket_alpha_star.csv"
Write-Host "  - Day8结果: output/alpha_u_lut.json, output/alpha_u_curve.png"
Write-Host "  - Day9结果: output/metrics_emg_q0.json, output/emg_comparison_charts_q0.png"
Write-Host ""

