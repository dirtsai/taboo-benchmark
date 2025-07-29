# 中文Taboo实验完整框架

> 基于OpenHowNet构建中文数据集并进行全面评估  
> 参考base_test.ipynb的实验架构，专门针对中文语言模型优化

## 📋 实验概览

### 实验目标
- 构建高质量的中文Taboo游戏数据集
- 评估多个语言模型在中文语义理解任务上的表现
- 建立标准化的中文LLM评估基准

### 核心创新
1. **首次应用OpenHowNet**：利用中文知识图谱构建语义相关的禁用词
2. **中文特化设计**：针对中文语言特点优化格式检查和评估机制
3. **统一实验架构**：参考base_test成熟框架，确保实验科学性

## 🏗️ 实验架构

### 1. 环境配置模块 (Environment Setup)

```python
# 核心依赖
- OpenHowNet: 中文知识图谱和语义分析
- jieba: 中文分词
- pandas: 数据处理和分析
- requests: API调用

# 设置
- 随机种子: 42 (确保可复现)
- 编码: UTF-8
- 分词日志: 最小化
```

### 2. 数据集构建模块 (Dataset Construction)

#### 2.1 OpenHowNet初始化
```python
def initialize_hownet():
    """
    - 下载HowNet资源包
    - 初始化词典对象
    - 测试API可用性
    - 验证中文词汇访问
    """
```

#### 2.2 词汇筛选与处理
```python
def process_chinese_words():
    """词汇有效性检查"""
    # 筛选条件:
    - 包含中文字符 (\\u4e00-\\u9fff)
    - 长度在1-6字符之间
    - 排除特殊符号 (·、—、…等)
    - 词性明确可识别
```

#### 2.3 禁用词生成策略
```python
def generate_taboo_words(target_word, hownet_dict):
    """
    多层次禁用词提取:
    1. 义原定义提取 - 从HowNet定义中提取语义相关词
    2. 同义词扩展 - 利用知识图谱关系
    3. 词性通用词 - 按词性添加通用禁用词
    4. 质量保证 - 确保每个词汇有5个禁用词
    """
```

#### 2.4 数据集结构
```json
{
  "target": "目标词",
  "part_of_speech": "词性(noun/verb/adj/adv)",
  "taboo": ["禁用词1", "禁用词2", "禁用词3", "禁用词4", "禁用词5"],
  "category": "chinese_hownet",
  "senses": [
    {
      "zh_word": "中文词",
      "en_word": "英文词", 
      "zh_grammar": "中文词性",
      "Def": "义原定义",
      "sememes": "义原列表"
    }
  ],
  "metadata": {
    "sense_count": 1,
    "taboo_count": 5,
    "source": "openhownet"
  }
}
```

### 3. 游戏逻辑模块 (Game Logic)

#### 3.1 中文格式检查
```python
def extract_chinese_clue_text(response):
    """
    格式要求: [线索] 描述内容
    备用格式: 线索: 描述内容
    错误处理: 多次重试机制
    """

def extract_chinese_guess_word(response):
    """
    格式要求: [猜测] 词汇
    提取策略: 正则匹配中文词汇
    容错机制: 自动提取最可能的词
    """
```

#### 3.2 禁用词违规检测
```python
def check_chinese_taboo_violation(hint, taboo_words):
    """
    严格检查机制:
    - 完整词匹配
    - 部分匹配(3字符以上)
    - 核心字符匹配(2字符)
    - 即时终止游戏
    """
```

#### 3.3 游戏主函数
```python
def play_chinese_taboo_game(client, hinter_model, guesser_model, 
                           target_word, taboo_words, max_turns=5):
    """
    核心游戏流程:
    1. 构建中文系统提示
    2. 执行多轮对话
    3. 实时违规检测
    4. 详细结果记录
    5. 异常情况处理
    """
```

### 4. API客户端模块 (API Client)

#### 4.1 统一API接口
```python
class ChineseTabooClient:
    """
    - OpenRouter API集成
    - 请求重试机制
    - 响应清理处理
    - 中文编码保证
    """
```

#### 4.2 支持模型列表
```python
CHINESE_TEST_MODELS = [
    "openai/gpt-4o",              # GPT-4o 支持中文
    "google/gemini-2.5-flash",    # Gemini 支持中文  
    "deepseek/deepseek-chat-v3-0324",  # DeepSeek 中文模型
    "anthropic/claude-sonnet-4"   # Claude 支持中文
]
```

### 5. 实验执行模块 (Experiment Execution)

#### 5.1 统一实验运行器
```python
def run_chinese_taboo_experiment(client, models, dataset, config):
    """
    实验模式:
    - simple: 测试模式(少量词汇)
    - grouped: 全量模式(按hinter分组)
    
    配置参数:
    - experiment_type: test/full
    - max_turns: 最大轮数
    - batch_size: 批次大小
    - output_dir: 输出目录
    """
```

#### 5.2 测试实验流程
```python
def run_simple_chinese_experiment():
    """
    测试实验特点:
    1. 随机选择3-5个词汇
    2. 所有模型组合测试
    3. 快速验证系统功能
    4. 结果即时分析
    """
```

#### 5.3 全量实验流程
```python
def run_grouped_chinese_experiment():
    """
    全量实验特点:
    1. 遍历所有数据集词汇
    2. 按hinter模型分组执行
    3. 批次保存机制
    4. 详细统计分析
    """
```

### 6. 结果分析模块 (Result Analysis)

#### 6.1 基础统计
- 总体成功率
- 各模型表现(Hinter/Guesser角色)
- 词性表现差异
- 游戏轮数分布

#### 6.2 失败原因分析
- TABOO_VIOLATION: 违反禁用词规则
- FORMAT_FAILURE: 格式错误超限
- API_FAILURE: API调用失败
- MAX_TURNS_EXCEEDED: 轮数耗尽

#### 6.3 深度分析
- 禁用词违规模式
- 格式错误类型
- 模型对话质量
- 中文语义理解能力

## 🎯 实验流程

### Phase 1: 环境准备
1. 安装依赖包 (OpenHowNet, jieba, pandas, requests)
2. 初始化OpenHowNet词典
3. 配置API客户端
4. 设置随机种子

### Phase 2: 数据集构建  
1. 从HowNet获取中文词汇 (13万+)
2. 按词性筛选有效词汇
3. 随机选择平衡样本 (每词性25个)
4. 生成语义相关禁用词
5. 构建标准数据集格式

### Phase 3: 测试验证
1. 执行小规模测试实验 (3个词汇)
2. 验证游戏逻辑正确性
3. 检查格式要求执行
4. 确认违规检测机制

### Phase 4: 全量实验 (可选)
1. 遍历完整数据集 (100个词汇)
2. 所有模型组合测试 (1600场游戏)
3. 批次保存实验结果
4. 生成综合分析报告

### Phase 5: 结果分析
1. 基础统计分析
2. 模型性能对比
3. 失败原因分类
4. 语言特性分析

## 📊 数据集特色

### 规模与分布
- **总词汇数**: 100个 (可扩展)
- **词性分布**: 名词25个, 动词25个, 形容词25个, 副词25个
- **禁用词**: 每词5个, 总计500个
- **语义覆盖**: 基于OpenHowNet知识图谱

### 质量保证
- **语义相关性**: 禁用词与目标词语义相关
- **词性一致性**: 严格按标准词性分类
- **复杂度平衡**: 避免过于简单或复杂的词汇
- **文化适应性**: 选择中文母语者熟悉的词汇

### 技术优势
- **知识图谱支持**: 利用OpenHowNet语义关系
- **动态生成**: 基于义原定义自动生成禁用词
- **可扩展性**: 支持领域专业词汇扩展
- **标准化**: 统一的数据格式和评估标准

## 🔧 核心技术

### 中文语言处理
1. **分词技术**: jieba分词处理
2. **字符过滤**: Unicode中文字符范围
3. **格式检查**: 中英文双语格式支持
4. **编码处理**: UTF-8全程保证

### 语义分析
1. **义原提取**: HowNet义原系统
2. **关系推理**: 语义相似度计算
3. **同义词扩展**: 知识图谱遍历
4. **语境理解**: 上下文语义分析

### 违规检测
1. **完整匹配**: 精确字符串匹配
2. **部分匹配**: 词根和核心字符
3. **模糊匹配**: 容错拼写变体
4. **实时检测**: 游戏过程即时验证

## 📈 实验价值

### 学术贡献
1. **首创性**: 首个基于OpenHowNet的Taboo游戏数据集
2. **基准性**: 为中文LLM评估提供新基准
3. **系统性**: 完整的评估框架和工具链
4. **开放性**: 可复现的实验设计

### 应用价值
1. **模型评估**: 语义理解能力测试
2. **对话系统**: 多轮对话质量评估
3. **中文NLP**: 语言生成能力测试
4. **教育工具**: 语言学习游戏化

### 技术突破
1. **中文适配**: 针对中文特点的深度优化
2. **知识融合**: 传统NLP与知识图谱结合
3. **评估创新**: 游戏化的模型评估方式
4. **框架统一**: 测试与全量实验的一致架构

## 🚀 扩展方向

### 短期扩展
1. **数据规模**: 扩展到300-500个词汇
2. **模型覆盖**: 添加更多中文模型测试
3. **领域专业**: 增加医学、法律、科技词汇
4. **难度分级**: 按复杂度分类词汇

### 中期发展
1. **多语对比**: 中英文Taboo游戏对比
2. **文化适应**: 不同地区中文变体测试
3. **动态难度**: 自适应难度调整机制
4. **实时评估**: 在线评估系统开发

### 长期愿景
1. **标准化**: 建立行业标准评估基准
2. **生态化**: 构建完整的评估生态系统
3. **国际化**: 推广到国际中文教育
4. **智能化**: AI辅助的自动化评估

## 📋 文件结构

```
taboo_benchmark/
├── chinese_taboo_experiment_complete.ipynb  # 完整实验notebook
├── chinese_experiment_framework.md          # 实验框架文档  
├── data/
│   ├── chinese_dataset_complete.json        # 完整数据集
│   ├── chinese_dataset_simple.json          # 简化数据集
│   └── chinese_dataset_report.json          # 数据集报告
├── results/
│   └── chinese_experiment_*/                # 实验结果目录
│       ├── complete_results.csv             # 完整结果
│       └── analysis_report.json             # 分析报告
└── api_keys.json                            # API密钥配置
```

## 🎉 总结

中文Taboo实验系统成功地将OpenHowNet知识图谱技术与语言模型评估相结合，创建了首个针对中文的Taboo游戏评估基准。通过参考base_test的成熟架构，建立了科学、规范、可重复的实验框架，为中文语言模型的语义理解能力评估提供了创新的工具和方法。

该系统不仅在技术上具有突破性，在应用价值上也具有重要意义，为中文NLP领域的发展和中文语言模型的评估标准化做出了重要贡献。 