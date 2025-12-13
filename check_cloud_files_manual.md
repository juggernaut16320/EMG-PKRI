# 手动检查云端文件的方法

## 方法1：使用SSH直接检查（推荐）

如果你有SSH访问权限，可以直接在本地终端运行以下命令：

### Windows PowerShell

```powershell
# 设置变量（请修改为你的实际值）
$CLOUD_USER = 'your_username'
$CLOUD_HOST = 'your_server_ip'
$CLOUD_PORT = 22
$PROJECT_DIR = '/mnt/workspace/EMG-PKRI'  # 根据实际情况修改

# 检查单个文件
ssh -p $CLOUD_PORT "${CLOUD_USER}@${CLOUD_HOST}" "ls -lh $PROJECT_DIR/data/train.jsonl"

# 检查文件是否存在并显示行数
ssh -p $CLOUD_PORT "${CLOUD_USER}@${CLOUD_HOST}" "if [ -f $PROJECT_DIR/data/train.jsonl ]; then echo '存在'; wc -l $PROJECT_DIR/data/train.jsonl; else echo '不存在'; fi"

# 批量检查所有文件
$files = @(
    'data/train.jsonl',
    'data/dev.jsonl',
    'data/test.jsonl',
    'data/hard_eval_set.jsonl',
    'configs/lexicons/porn.txt',
    'configs/lexicons/politics.txt',
    'configs/lexicons/abuse.txt',
    'configs/lexicons/regex_patterns.txt'
)

foreach ($file in $files) {
    $result = ssh -p $CLOUD_PORT "${CLOUD_USER}@${CLOUD_HOST}" "if [ -f $PROJECT_DIR/$file ]; then echo '✅'; ls -lh $PROJECT_DIR/$file; else echo '❌ 不存在'; fi"
    Write-Host "$file : $result"
}
```

### Linux/Mac Bash

```bash
# 设置变量（请修改为你的实际值）
CLOUD_USER='your_username'
CLOUD_HOST='your_server_ip'
CLOUD_PORT=22
PROJECT_DIR='/mnt/workspace/EMG-PKRI'  # 根据实际情况修改

# 检查单个文件
ssh -p $CLOUD_PORT "${CLOUD_USER}@${CLOUD_HOST}" "ls -lh $PROJECT_DIR/data/train.jsonl"

# 检查文件是否存在并显示行数
ssh -p $CLOUD_PORT "${CLOUD_USER}@${CLOUD_HOST}" "if [ -f $PROJECT_DIR/data/train.jsonl ]; then echo '存在'; wc -l $PROJECT_DIR/data/train.jsonl; else echo '不存在'; fi"

# 批量检查
for file in \
    data/train.jsonl \
    data/dev.jsonl \
    data/test.jsonl \
    data/hard_eval_set.jsonl \
    configs/lexicons/porn.txt \
    configs/lexicons/politics.txt \
    configs/lexicons/abuse.txt \
    configs/lexicons/regex_patterns.txt; do
    echo "检查: $file"
    ssh -p $CLOUD_PORT "${CLOUD_USER}@${CLOUD_HOST}" "if [ -f $PROJECT_DIR/$file ]; then echo '✅ 存在'; ls -lh $PROJECT_DIR/$file; wc -l $PROJECT_DIR/$file; else echo '❌ 不存在'; fi"
    echo ""
done
```

## 方法2：登录到云端服务器检查

如果你可以直接登录到云端服务器：

```bash
# SSH登录
ssh -p 22 your_username@your_server_ip

# 进入项目目录
cd /mnt/workspace/EMG-PKRI  # 或你的实际项目路径

# 检查数据文件
ls -lh data/*.jsonl
wc -l data/*.jsonl

# 检查词表文件
ls -lh configs/lexicons/*.txt
wc -l configs/lexicons/*.txt
```

## 方法3：使用云平台控制台

如果你使用的是阿里云、腾讯云等平台：

1. 登录云平台控制台
2. 找到你的服务器实例
3. 使用"远程连接"功能
4. 在终端中执行上述检查命令

## 需要检查的文件列表

### 数据文件（必需）
- `data/train.jsonl` - 训练集
- `data/dev.jsonl` - 验证集
- `data/test.jsonl` - 测试集
- `data/hard_eval_set.jsonl` - 困难样本集（可选）

### 词表文件（必需）
- `configs/lexicons/porn.txt` - 色情词表
- `configs/lexicons/politics.txt` - 涉政词表
- `configs/lexicons/abuse.txt` - 辱骂词表
- `configs/lexicons/regex_patterns.txt` - 正则规则（可选）

## 快速检查命令（一行）

```bash
# 检查所有必需文件是否存在
ssh user@server "cd /path/to/EMG-PKRI && for f in data/train.jsonl data/dev.jsonl data/test.jsonl configs/lexicons/porn.txt configs/lexicons/politics.txt configs/lexicons/abuse.txt; do [ -f \$f ] && echo '✅' \$f || echo '❌' \$f; done"
```

