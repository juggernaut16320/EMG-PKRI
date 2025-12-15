# pyahocorasick 库安装说明

## 问题

Day6 q0构建时，词表匹配效率很低，原因是未安装 `pyahocorasick` 库。

## 解决方案

代码已经支持使用 `pyahocorasick` 库（Aho-Corasick算法）来加速词表匹配：

- **使用 pyahocorasick**：O(m + k) 时间复杂度，高性能 ✅
- **未安装 pyahocorasick**：回退到 O(n × m) 字符串匹配，非常慢 ❌

## 安装方法

### 方法1：使用 requirements.txt（推荐）

```bash
pip install -r requirements.txt
```

### 方法2：单独安装

```bash
pip install pyahocorasick>=2.0.0
```

## 性能对比

对于包含约6000个词的词表，匹配一条文本：

- **使用 pyahocorasick（Aho-Corasick算法）**：
  - 时间复杂度：O(m + k)，其中m是文本长度，k是匹配数
  - 实际速度：**快10-100倍**

- **未安装（回退到字符串匹配）**：
  - 时间复杂度：O(n × m)，其中n是词表大小，m是文本长度
  - 实际速度：**非常慢**（对于6000个词，每条文本需要6000次字符串查找）

## 验证安装

运行 q0_builder.py 时，如果看到以下日志，说明已正确使用：

```
✓ 构建 Aho-Corasick 自动机缓存，共 3 个类别
✓ 自动机缓存构建完成，将使用高性能匹配算法
```

如果看到以下日志，说明未安装，需要安装：

```
⚠ 未安装 pyahocorasick，将使用较慢的字符串匹配方式。建议安装: pip install pyahocorasick
⚠ 将使用较慢的字符串匹配方式（建议安装 pyahocorasick: pip install pyahocorasick）
```

## 代码位置

相关代码在 `scripts/q0_builder.py`：
- 第29-35行：尝试导入 pyahocorasick
- 第152-179行：`build_automaton_cache` 函数构建自动机缓存
- 第129-138行：使用自动机进行高效匹配
- 第139-147行：回退到慢速匹配（当未安装时）

## 注意事项

- `pyahocorasick` 已在 `requirements.txt` 中添加（2025-12-15更新）
- 安装后无需修改代码，会自动使用高性能算法
- 建议在执行P0任务（q0参数调优）前先安装此库，否则调优过程会非常慢

