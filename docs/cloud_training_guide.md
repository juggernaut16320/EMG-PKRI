# 云上训练指南

## 当前状态

- **模型**: Qwen3-1.7B + LoRA (r=8, alpha=16)
- **数据集**: 训练集 39,583 条，验证集 4,948 条
- **本地问题**: RTX 2070 (8GB) 显存不足，导致训练极慢（约 96 天）

## 重要提示：模型下载方法

**推荐使用 ModelScope 下载模型**，原因：
- ✅ 在阿里云 DSW 上最稳定，避免网络问题
- ✅ 自动处理镜像和 CDN 问题
- ✅ 下载速度稳定
- ❌ 避免使用 HuggingFace 直接下载（可能遇到 SSL/CDN 问题）
- ❌ 避免使用 git-lfs（可能遇到 DNS 解析问题）

## 云上训练配置建议

### 推荐云服务器配置

**最低配置（可运行）：**
- GPU: 16GB 显存（如 V100, T4）
- 批次大小: 8-12
- 预计时间: 2-4 小时

**推荐配置（高效）：**
- GPU: 24GB+ 显存（如 A10, A100-40GB）
- 批次大小: 16-32
- 预计时间: 1-2 小时

**最佳配置（最快）：**
- GPU: 40GB+ 显存（如 A100-80GB）
- 批次大小: 32-64
- 预计时间: 30-60 分钟

### 云上训练配置优化

在 `configs/config.yaml` 中，根据云服务器显存调整：

```yaml
# 16GB 显存（如 V100, T4）
training:
  per_device_train_batch_size: 12
  per_device_eval_batch_size: 24
  gradient_accumulation_steps: 1  # 可选，进一步增大有效批次

# 24GB 显存（如 A10）
training:
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 32
  gradient_accumulation_steps: 1

# 40GB+ 显存（如 A100）
training:
  per_device_train_batch_size: 32
  per_device_eval_batch_size: 64
  gradient_accumulation_steps: 1
```

### 云上训练步骤

1. **克隆代码**
   ```bash
   # 使用 Git 克隆项目
   git clone git@github.com:juggernaut16320/EMG-PKRI.git
   # 或使用 HTTPS
   git clone https://github.com/juggernaut16320/EMG-PKRI.git
   ```

2. **设置 Python 环境**
   ```bash
   # 进入项目目录
   cd EMG-PKRI
   
   # 创建虚拟环境（推荐使用 venv，不需要 conda）
   python3 -m venv venv
   
   # 激活虚拟环境
   source venv/bin/activate
   
   # 安装依赖
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **下载模型（重要：使用 ModelScope，避免网络问题）**
   
   **推荐方法：使用 ModelScope（阿里云服务，在阿里云 DSW 上最稳定）**
   
   ```bash
   # 激活虚拟环境
   source venv/bin/activate
   
   # 安装 ModelScope
   pip install modelscope
   
   # 设置模型下载目录（使用相对路径或环境变量）
   # 方式1：使用相对路径（推荐）
   MODELS_DIR="./models"  # 相对于项目根目录
   # 方式2：使用绝对路径（根据实际情况修改）
   # MODELS_DIR="/mnt/workspace/models"  # 云服务器路径示例
   
   # 使用 ModelScope 下载模型（会自动处理网络问题）
   python -c "from modelscope import snapshot_download; snapshot_download('qwen/Qwen3-1.7B', cache_dir='$MODELS_DIR')"
   
   # 等待下载完成（约 3-5 分钟，模型大小约 4GB）
   # 可以在另一个终端监控进度：
   # watch -n 2 du -sh $MODELS_DIR/qwen/
   
   # 下载完成后，查找模型实际路径
   # ModelScope 会将点号替换为下划线，路径类似：
   # ./models/qwen/Qwen3-1___7B 或 /mnt/workspace/models/qwen/Qwen3-1___7B
   find $MODELS_DIR -name "config.json" -path "*/Qwen3*" | head -1 | xargs dirname
   
   # 修改配置文件使用本地路径
   MODEL_PATH=$(find $MODELS_DIR -name "config.json" -path "*/Qwen3*" | head -1 | xargs dirname)
   # 转换为绝对路径（如果需要）
   MODEL_PATH=$(cd "$MODEL_PATH" && pwd)
   sed -i "s|name_or_path: \"Qwen/Qwen3-1.7B\"|name_or_path: \"$MODEL_PATH\"|g" configs/config.yaml
   
   # 验证修改
   grep "name_or_path" configs/config.yaml
   ```
   
   **注意事项：**
   - ModelScope 会将模型名称中的点号（`.`）替换为下划线（`___`）
   - 例如：`Qwen3-1.7B` → `Qwen3-1___7B`
   - 模型会下载到指定的 `cache_dir` 目录下
   - 下载完成后，使用 `find` 命令查找实际路径
   
   **如果 ModelScope 下载失败，备选方案：**
   
   ```bash
   # 设置模型目录（根据实际情况修改）
   MODELS_DIR="./models"  # 或使用绝对路径
   
   # 方案1：使用 git-lfs（如果网络允许）
   apt-get update && apt-get install -y git-lfs
   git lfs install
   git clone https://hf-mirror.com/Qwen/Qwen3-1.7B $MODELS_DIR/Qwen3-1.7B
   
   # 方案2：直接使用 HuggingFace（需要设置镜像）
   export HF_ENDPOINT=https://hf-mirror.com
   export HF_HUB_DISABLE_XET_CACHE=1
   # 然后直接运行训练脚本，会自动下载
   ```

3. **配置环境变量（如需要）**
   ```bash
   export GEMINI_API_KEY="your-api-key"  # 如果需要重新打标签
   ```

4. **调整配置文件**
   - 根据云服务器显存修改 `configs/config.yaml` 中的批次大小
   - 确认数据路径正确

5. **开始训练**
   ```bash
   python scripts/baseline_train.py
   ```

6. **监控训练**
   - 查看日志: `checkpoints/baseline-lora/logs/`
   - 使用 `nvidia-smi` 监控 GPU 使用情况
   - 使用 `watch -n 1 nvidia-smi` 实时监控

### 训练时间估算

基于不同配置的训练时间（3 epochs）：

| 显存 | 批次大小 | 总步数 | 每步时间 | 预计总时间 |
|------|---------|--------|---------|-----------|
| 8GB  | 8       | 14,844 | 17-43s  | ~96 天 ❌ |
| 16GB | 12      | 9,896  | 1-2s    | 2-4 小时 ✅ |
| 24GB | 16      | 7,422  | 0.8-1.5s| 1-2 小时 ✅ |
| 40GB | 32      | 3,711  | 0.5-1s  | 30-60 分钟 ✅ |

### 云上训练注意事项

1. **数据备份**: 训练前确保数据已上传
2. **断点续训**: 支持从 checkpoint 继续训练
   ```bash
   python scripts/baseline_train.py --resume-from-checkpoint checkpoints/baseline-lora/checkpoint-XXX
   ```
3. **模型下载**: 训练完成后下载 checkpoint
   ```bash
   scp -r user@server:/path/to/checkpoints/baseline-lora ./checkpoints/
   ```
4. **成本控制**: 
   - 使用 spot 实例降低成本
   - 训练完成后及时停止实例
   - 监控训练进度，避免长时间运行

### 推荐的云服务商

- **阿里云**: ECS GPU 实例（V100, A10, A100）
- **腾讯云**: GPU 云服务器（T4, V100, A100）
- **AWS**: EC2 GPU 实例（g4dn, p3, p4d）
- **Google Cloud**: Compute Engine GPU（T4, V100, A100）
- **AutoDL**: 按小时计费，性价比高

### 验证训练是否正常

训练开始后，检查：
1. GPU 利用率应该 > 80%
2. 显存使用率应该 > 70%
3. 每步时间应该在 1-2 秒内（16GB+ 显存）
4. 训练日志正常输出 loss 和 metrics

如果每步时间 > 5 秒，可能是：
- 显存不足（减小批次大小）
- 数据加载慢（增加 `dataloader_num_workers`）
- 网络问题（如果数据在远程）

## 模型下载验证步骤

### 验证模型文件完整性

```bash
# 设置模型目录（根据实际情况修改）
MODELS_DIR="./models"  # 或使用绝对路径

# 1. 查找模型路径
MODEL_PATH=$(find $MODELS_DIR -name "config.json" -path "*/Qwen3*" | head -1 | xargs dirname)
# 转换为绝对路径（如果需要）
MODEL_PATH=$(cd "$MODEL_PATH" && pwd)
echo "模型路径: $MODEL_PATH"

# 2. 检查必需文件
ls -lh $MODEL_PATH

# 应该看到以下文件：
# - config.json (约 726B)
# - tokenizer_config.json (约 9.73KB)
# - vocab.json (约 2.78MB)
# - merges.txt (约 1.67MB)
# - tokenizer.json (约 11.4MB)
# - model-00001-of-00002.safetensors (约 3.44GB)
# - model-00002-of-00002.safetensors (约 622MB)
# - model.safetensors.index.json (约 25.6KB)

# 3. 验证总大小（应该约 3.8-4GB）
du -sh $MODEL_PATH

# 4. 验证模型配置
cat $MODEL_PATH/config.json | grep -E "model_type|num_hidden_layers"
# 应该显示 "model_type": "qwen3"
```

**注意：** 上述命令中的 `MODELS_DIR` 变量需要根据实际情况设置：
- 本地开发：`MODELS_DIR="./models"`（相对路径）
- 云服务器：`MODELS_DIR="/mnt/workspace/models"`（根据实际路径修改）

## 常见问题排查

### 问题1：模型下载失败

**症状：** 下载超时、SSL 错误、DNS 解析失败

**原因：** 
- HuggingFace CDN (`cas-bridge.xethub.hf.co`) 无法访问
- git-lfs 下载大文件时 DNS 解析失败
- 网络不稳定

**解决方案：**
```bash
# ✅ 推荐：使用 ModelScope（阿里云服务，最稳定）
# 设置模型目录（根据实际情况修改）
MODELS_DIR="./models"  # 或使用绝对路径

pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('qwen/Qwen3-1.7B', cache_dir='$MODELS_DIR')"
```

### 问题2：找不到模型路径

**症状：** `No such file or directory` 错误

**原因：** ModelScope 会将模型名称中的点号（`.`）替换为三个下划线（`___`）

**解决方案：**
```bash
# 设置模型目录（根据实际情况修改）
MODELS_DIR="./models"  # 或使用绝对路径

# 查找实际路径
find $MODELS_DIR -name "config.json" -path "*/Qwen3*" | head -1 | xargs dirname

# 或手动查找
ls -la $MODELS_DIR/qwen/
# 目录名可能是 Qwen3-1___7B（注意是三个下划线，不是点号）
```

### 问题3：训练时找不到模型

**症状：** `OSError: Qwen/Qwen3-1.7B is not a local folder`

**原因：** 配置文件中的路径不正确

**解决方案：**
```bash
# 设置模型目录（根据实际情况修改）
MODELS_DIR="./models"  # 或使用绝对路径

# 1. 查找模型实际路径
MODEL_PATH=$(find $MODELS_DIR -name "config.json" -path "*/Qwen3*" | head -1 | xargs dirname)
# 转换为绝对路径（如果需要）
MODEL_PATH=$(cd "$MODEL_PATH" && pwd)

# 2. 修改配置文件（在项目根目录执行）
sed -i "s|name_or_path:.*|name_or_path: \"$MODEL_PATH\"|g" configs/config.yaml

# 3. 验证修改
grep "name_or_path" configs/config.yaml
```

### 问题4：ModelScope 下载很慢

**解决方案：**
- 等待下载完成（模型约 4GB，可能需要 3-5 分钟）
- 在另一个终端监控进度：`watch -n 2 du -sh $MODELS_DIR/qwen/`（需要先设置 `MODELS_DIR` 变量）
- 如果下载中断，重新运行下载命令（支持断点续传）

## 快速参考命令

```bash
# 完整流程（复制粘贴即可，根据实际情况修改路径）
# 进入项目目录（根据实际情况修改）
cd /path/to/EMG-PKRI  # 或 cd EMG-PKRI（如果已在项目目录）

# 激活虚拟环境
source venv/bin/activate

# 设置模型目录（根据实际情况修改）
MODELS_DIR="./models"  # 本地使用相对路径
# 或 MODELS_DIR="/mnt/workspace/models"  # 云服务器使用绝对路径

# 下载模型
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('qwen/Qwen3-1.7B', cache_dir='$MODELS_DIR')"

# 配置路径
MODEL_PATH=$(find $MODELS_DIR -name "config.json" -path "*/Qwen3*" | head -1 | xargs dirname)
# 转换为绝对路径（如果需要）
MODEL_PATH=$(cd "$MODEL_PATH" && pwd)
sed -i "s|name_or_path: \"Qwen/Qwen3-1.7B\"|name_or_path: \"$MODEL_PATH\"|g" configs/config.yaml

# 开始训练
python scripts/baseline_train.py
```

