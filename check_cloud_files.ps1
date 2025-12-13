# PowerShell脚本：检查云端文件
# 使用方法：修改下面的变量，然后运行：.\check_cloud_files.ps1

# ===== 请修改以下变量 =====
$CLOUD_USER = 'your_username'      # 修改为你的用户名
$CLOUD_HOST = 'your_server_ip'     # 修改为服务器IP或域名
$CLOUD_PORT = 22                   # SSH端口
$PROJECT_DIR = '/path/to/EMG-PKRI' # 云端项目目录，例如：/mnt/workspace/EMG-PKRI 或 ~/EMG-PKRI

# ===== 需要检查的文件 =====
$FILES = @(
    'data/train.jsonl',
    'data/dev.jsonl',
    'data/test.jsonl',
    'data/hard_eval_set.jsonl',
    'configs/lexicons/porn.txt',
    'configs/lexicons/politics.txt',
    'configs/lexicons/abuse.txt',
    'configs/lexicons/regex_patterns.txt'
)

Write-Host '========================================='
Write-Host "检查云端文件 ($CLOUD_USER@$CLOUD_HOST:$CLOUD_PORT)"
Write-Host "项目目录: $PROJECT_DIR"
Write-Host '========================================='
Write-Host ''

foreach ($file in $FILES) {
    $remotePath = "$PROJECT_DIR/$file"
    
    # 构建SSH命令
    $sshCmd = "if [ -f '$remotePath' ]; then echo 'EXISTS'; du -h '$remotePath' | cut -f1; wc -l < '$remotePath' 2>/dev/null || echo 0; else echo 'NOT_EXISTS'; fi"
    
    try {
        # 执行SSH命令
        $result = ssh -p $CLOUD_PORT "${CLOUD_USER}@${CLOUD_HOST}" $sshCmd 2>&1
        
        if ($result -match 'EXISTS') {
            $lines = ($result | Select-String -Pattern '^\d+$').Matches[0].Value
            $size = ($result | Select-String -Pattern '^\d+[KMGT]?B?$').Matches[0].Value
            Write-Host "✅ 存在: $file (大小: $size, 行数: $lines)" -ForegroundColor Green
        } else {
            Write-Host "❌ 不存在: $file" -ForegroundColor Red
        }
    } catch {
        Write-Host "❌ 检查失败: $file - $_" -ForegroundColor Red
    }
}

Write-Host ''
Write-Host '========================================='
Write-Host '检查完成'
Write-Host '========================================='

