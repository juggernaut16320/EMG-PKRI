# 简化版：检查云端文件的PowerShell脚本
# 使用方法：
#   1. 修改下面的变量
#   2. 运行：.\check_cloud_files_simple.ps1

# ===== 请修改以下变量 =====
$CLOUD_USER = 'your_username'      # 修改为你的用户名
$CLOUD_HOST = 'your_server_ip'     # 修改为服务器IP或域名
$CLOUD_PORT = 22                   # SSH端口
$PROJECT_DIR = '/mnt/workspace/EMG-PKRI'  # 云端项目目录（根据实际情况修改）

Write-Host "检查云端文件: $CLOUD_USER@$CLOUD_HOST:$CLOUD_PORT"
Write-Host "项目目录: $PROJECT_DIR"
Write-Host ""

# 检查数据文件
Write-Host "--- 数据文件 ---"
$dataFiles = @('train.jsonl', 'dev.jsonl', 'test.jsonl', 'hard_eval_set.jsonl')
foreach ($file in $dataFiles) {
    $cmd = "test -f $PROJECT_DIR/data/$file && echo 'EXISTS' || echo 'NOT_EXISTS'"
    $result = ssh -p $CLOUD_PORT "${CLOUD_USER}@${CLOUD_HOST}" $cmd
    if ($result -match 'EXISTS') {
        $size = ssh -p $CLOUD_PORT "${CLOUD_USER}@${CLOUD_HOST}" "du -h $PROJECT_DIR/data/$file | cut -f1"
        $lines = ssh -p $CLOUD_PORT "${CLOUD_USER}@${CLOUD_HOST}" "wc -l < $PROJECT_DIR/data/$file"
        Write-Host "✅ data/$file - 大小: $size, 行数: $lines" -ForegroundColor Green
    } else {
        Write-Host "❌ data/$file - 不存在" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "--- 词表文件 ---"
$lexiconFiles = @('porn.txt', 'politics.txt', 'abuse.txt', 'regex_patterns.txt')
foreach ($file in $lexiconFiles) {
    $cmd = "test -f $PROJECT_DIR/configs/lexicons/$file && echo 'EXISTS' || echo 'NOT_EXISTS'"
    $result = ssh -p $CLOUD_PORT "${CLOUD_USER}@${CLOUD_HOST}" $cmd
    if ($result -match 'EXISTS') {
        $size = ssh -p $CLOUD_PORT "${CLOUD_USER}@${CLOUD_HOST}" "du -h $PROJECT_DIR/configs/lexicons/$file | cut -f1"
        $lines = ssh -p $CLOUD_PORT "${CLOUD_USER}@${CLOUD_HOST}" "wc -l < $PROJECT_DIR/configs/lexicons/$file"
        Write-Host "✅ configs/lexicons/$file - 大小: $size, 行数: $lines" -ForegroundColor Green
    } else {
        Write-Host "❌ configs/lexicons/$file - 不存在" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "检查完成"

