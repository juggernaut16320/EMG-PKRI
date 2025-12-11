# 云上训练指南

## 当前状态

- **模型**: Qwen3-1.7B + LoRA (r=8, alpha=16)
- **数据集**: 训练集 39,583 条，验证集 4,948 条
- **本地问题**: RTX 2070 (8GB) 显存不足，导致训练极慢（约 96 天）

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

1. **上传代码和数据**
   ```bash
   # 上传整个项目目录
   scp -r /path/to/project user@server:/path/to/destination
   ```

2. **安装依赖**
   ```bash
   conda create -n emgpkri python=3.10
   conda activate emgpkri
   pip install -r requirements.txt
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

