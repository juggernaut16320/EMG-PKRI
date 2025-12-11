# 中文推特数据清洗策略

## 一、推特内容特点分析

### 1.1 推特特有元素
- **URL链接**：`http://`, `https://`, `www.`, 短链接（如 `t.co/xxx`）
- **@提及**：`@username` 格式的用户名提及
- **#标签**：`#hashtag` 格式的话题标签
- **转发标记**：`RT @username:`, `Retweet`, `转发` 等
- **表情符号**：Unicode emoji、颜文字等
- **特殊格式**：`via @username`, `cc: @username` 等

### 1.2 中文推特特殊问题
- **繁简混用**：繁体字和简体字混合
- **编码问题**：乱码字符、无效Unicode
- **特殊符号**：全角/半角混用、特殊标点
- **重复内容**：重复字符、重复词汇（如"哈哈哈"）
- **极短文本**：只有表情、只有URL、只有@提及等

## 二、清洗策略设计

### 2.1 基础过滤（必须）

#### 2.1.1 空文本过滤
```python
- 完全空白的文本
- 只包含空白字符（空格、制表符、换行等）
```

#### 2.1.2 长度过滤
```python
- 最小长度：建议 5-10 个字符（去除只有表情或极短内容）
- 最大长度：无限制（保留所有长度的文本）
```

#### 2.1.3 编码过滤
```python
- 检测并移除无效的Unicode字符
- 移除无法解码的字节序列
- 统一编码为UTF-8
```

### 2.2 推特内容清理（可选，可配置）

#### 2.2.1 URL处理策略
**采用方案：替换为占位符**
```python
- 将URL替换为 [URL] 占位符
- 保留文本结构，便于后续分析
- 能正确处理URL后直接跟中文的情况
```

#### 2.2.2 @提及处理
**采用方案：替换为占位符**
```python
- 将 @username 替换为 [MENTION]
- 保留文本结构，便于后续分析
- 使用前瞻断言确保边界正确
```

#### 2.2.3 #标签处理
**说明：** #标签在数据抓取阶段已被删除，无需处理

#### 2.2.4 转发标记处理
**采用方案：提取转发内容**
```python
- 检测 RT @username: 或 转发 @username: 等转发标记
- 移除转发标记，只保留转发的原始文本内容
- 保留转发内容，因为可能包含敏感信息
```

### 2.3 中文文本特殊处理

#### 2.3.1 繁简统一
**说明：** 不进行繁简转换
```python
- 保留原始文本的繁简状态
- 现代中文NLP模型通常能处理繁简混合
- 避免误转换带来的信息损失
```

#### 2.3.2 乱码检测
```python
- 检测连续的特殊字符（如 ???, ???, ）
- 检测无法识别的字符序列
- 移除或标记乱码文本
```

#### 2.3.3 重复内容检测
```python
- 检测重复字符（如"哈哈哈哈"超过3次）
- 检测重复词汇（如"测试测试测试"）
- 可配置阈值
```

#### 2.3.4 特殊字符清理
```python
- 统一全角/半角标点符号
- 移除控制字符（除换行、制表符外）
- 规范化空白字符（多个空格合并为一个）
```

### 2.4 去重策略

#### 2.4.1 精确去重（采用方案）
```python
- 基于文本内容的精确匹配去重
- 使用MD5哈希值快速检测
- 保留第一次出现的记录
- 性能好，适合大规模数据处理
```

### 2.5 质量评分（可选）

可以为每条文本计算质量分数，用于后续筛选：

```python
质量分数 = f(
    文本长度,
    中文字符比例,
    是否包含URL,
    是否包含@提及,
    是否转发,
    乱码比例,
    重复度
)
```

## 三、实现方案设计

### 3.1 配置结构（config.yaml）

```yaml
data_cleaning:
  # 基础过滤
  min_length: 5              # 最小字符数
  max_length: null           # 最大字符数（无限制）
  remove_empty: true         # 移除空文本
  
  # URL处理
  url_handling: "replace"    # 替换为占位符
  url_placeholder: "[URL]"    # URL占位符
  
  # @提及处理
  mention_handling: "replace" # 替换为占位符
  mention_placeholder: "[MENTION]"  # @提及占位符
  
  # 转发处理
  retweet_handling: "extract" # 提取转发内容
  retweet_patterns:          # 转发标记模式
    - "RT @"
    - "转发 @"
    - "Retweet @"
  
  # 去重
  deduplication:
    exact: true              # 精确去重（基于MD5哈希）
```

### 3.2 函数设计

```python
def clean_text(text: str, config: dict) -> Optional[str]:
    """
    清洗单条文本
    
    Returns:
        清洗后的文本，如果被过滤则返回 None
    """
    pass

def remove_urls(text: str, mode: str = "keep") -> str:
    """移除或替换URL"""
    pass

def remove_mentions(text: str, mode: str = "keep") -> str:
    """移除或替换@提及"""
    pass

def handle_hashtags(text: str, mode: str = "keep") -> str:
    """处理#标签"""
    pass

def handle_retweets(text: str, mode: str = "keep") -> str:
    """处理转发内容"""
    pass

def detect_garbled(text: str) -> float:
    """检测乱码比例"""
    pass

def detect_repetition(text: str, threshold: int = 3) -> bool:
    """检测重复内容"""
    pass

def calculate_quality_score(text: str, config: dict) -> float:
    """计算质量分数"""
    pass

def deduplicate(items: list, config: dict) -> list:
    """去重"""
    pass
```

## 四、最终配置（针对敏感文本识别任务）

采用以下配置方案：

```yaml
data_cleaning:
  min_length: 5
  max_length: null           # 无限制
  url_handling: "replace"    # URL替换为[URL]
  mention_handling: "replace" # @提及替换为[MENTION]
  retweet_handling: "extract" # 提取转发内容
  deduplication:
    exact: true              # 精确去重
```

**特点：**
- 保留所有长度的文本（无最大长度限制）
- URL和@提及替换为占位符，保留文本结构
- 提取转发内容，保留可能的敏感信息
- 不进行繁简转换，保留原始文本
- 精确去重，性能好

## 五、实现建议

1. **分阶段实现**：
   - 第一阶段：基础过滤（空文本、长度、编码）
   - 第二阶段：推特内容清理（URL、@、#、转发）
   - 第三阶段：中文特殊处理（繁简、乱码、重复）
   - 第四阶段：去重和质量评分

2. **可配置性**：
   - 所有规则都应该是可配置的
   - 提供默认配置和推荐配置
   - 支持通过命令行参数覆盖配置

3. **日志和统计**：
   - 记录每种过滤规则过滤掉的样本数
   - 输出清洗统计报告
   - 便于调整清洗策略

4. **性能考虑**：
   - 使用正则表达式预编译
   - 去重使用哈希表优化
   - 大文件支持流式处理

