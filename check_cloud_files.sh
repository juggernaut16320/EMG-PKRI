#!/bin/bash
# 检查云端文件是否存在的脚本
# 使用方法：修改下面的变量，然后运行：bash check_cloud_files.sh

# ===== 请修改以下变量 =====
# 云端服务器SSH连接信息
CLOUD_USER="your_username"        # 修改为你的用户名
CLOUD_HOST="your_server_ip"       # 修改为服务器IP或域名
CLOUD_PORT="22"                   # SSH端口，默认22
PROJECT_DIR="/path/to/EMG-PKRI"   # 云端项目目录，默认可能是 /mnt/workspace/EMG-PKRI 或 ~/EMG-PKRI

# ===== 需要检查的文件列表 =====
# 数据文件
DATA_FILES=(
    "data/train.jsonl"
    "data/dev.jsonl"
    "data/test.jsonl"
    "data/hard_eval_set.jsonl"
)

# 词表文件
LEXICON_FILES=(
    "configs/lexicons/porn.txt"
    "configs/lexicons/politics.txt"
    "configs/lexicons/abuse.txt"
    "configs/lexicons/regex_patterns.txt"
)

# ===== 检查函数 =====
check_file() {
    local file_path="$1"
    ssh -p "$CLOUD_PORT" "${CLOUD_USER}@${CLOUD_HOST}" "
        if [ -f \"${PROJECT_DIR}/${file_path}\" ]; then
            size=\$(du -h \"${PROJECT_DIR}/${file_path}\" | cut -f1)
            lines=\$(wc -l < \"${PROJECT_DIR}/${file_path}\" 2>/dev/null || echo 0)
            echo \"✅ 存在: ${file_path}\"
            echo \"   大小: \$size\"
            echo \"   行数: \$lines\"
        else
            echo \"❌ 不存在: ${file_path}\"
        fi
    "
}

# ===== 执行检查 =====
echo "========================================="
echo "检查云端文件（${CLOUD_USER}@${CLOUD_HOST}:${CLOUD_PORT}）"
echo "项目目录: ${PROJECT_DIR}"
echo "========================================="
echo ""

echo "--- 数据文件 ---"
for file in "${DATA_FILES[@]}"; do
    check_file "$file"
    echo ""
done

echo "--- 词表文件 ---"
for file in "${LEXICON_FILES[@]}"; do
    check_file "$file"
    echo ""
done

echo "========================================="
echo "检查完成"
echo "========================================="

