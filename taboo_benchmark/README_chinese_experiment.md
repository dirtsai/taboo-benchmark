# 中文Taboo实验系统

## 概述

`chinese_taboo_experiment.ipynb` 是一个完整的中文Taboo游戏实验系统，仿照 `base_test.ipynb` 的结构，专门针对中文语言模型设计。该系统使用 OpenHowNet 构建中文词汇数据集，并提供完整的实验评估框架。

## 主要特性

### 🏗️ 数据集构建
- **基于OpenHowNet**: 利用中文知识图谱的丰富语义信息
- **智能禁用词生成**: 通过同义词、相似度计算和定义分析生成相关禁用词
- **词性平衡**: 名词、动词、形容词、副词各25个，共100个词汇
- **中文优化**: 专门针对中文语言特点进行词汇筛选和处理

### 🎮 游戏系统
- **中文格式**: 使用 `[线索]` 和 `[猜测]` 等中文标记
- **违规检测**: 专门的中文禁用词检测机制
- **健壮API**: 支持重试和格式纠错的API调用
- **详细记录**: 完整的游戏过程和统计信息记录

### 🤖 模型支持
- GPT-4o（OpenAI）
- Gemini 2.5 Pro（Google）
- DeepSeek Chat v3（专业中文模型）
- 可扩展支持更多中文模型

## 安装依赖

```bash
pip install OpenHowNet jieba
```

或者安装完整的项目依赖：

```bash
pip install -r requirements.txt
```

## 使用步骤

### 1. 环境准备
确保已配置 `api_keys.json` 文件，包含 OpenRouter API 密钥。

### 2. 运行实验
打开 `chinese_taboo_experiment.ipynb` 并按顺序执行所有 cells：

1. **环境初始化**: 安装并配置 OpenHowNet
2. **数据集构建**: 生成100个中文词汇的Taboo数据集
3. **统计分析**: 分析数据集的词性分布和质量
4. **保存数据**: 生成多种格式的数据文件
5. **API设置**: 配置中文模型客户端
6. **游戏逻辑**: 定义中文Taboo游戏规则
7. **测试实验**: 执行小规模测试验证系统

### 3. 查看结果
实验完成后会生成以下文件：
- `data/chinese_dataset.json` - 完整数据集
- `data/chinese_dataset_simple.json` - 简化版数据集
- `data/chinese_dataset_report.json` - 数据集统计报告
- `results/chinese_test_results_*.csv` - 实验结果

## 数据集结构

每个词汇条目包含：
```json
{
  "target": "目标词",
  "part_of_speech": "词性",
  "taboo": ["禁用词1", "禁用词2", "禁用词3", "禁用词4", "禁用词5"],
  "category": "chinese_general",
  "senses": [...], // OpenHowNet义项信息
  "metadata": {
    "sense_count": 3,
    "taboo_count": 5,
    "source": "openhownet"
  }
}
```

## 实验配置

### 默认设置
- **词汇数量**: 100个（每个词性25个）
- **测试模型**: 3个主要中文支持模型
- **游戏轮数**: 最多5轮
- **测试规模**: 3个词汇（可调整）

### 自定义配置
可以通过修改相关参数来调整：
- `target_count_per_pos`: 每个词性的词汇数量
- `num_test_words`: 测试实验的词汇数量
- `max_turns`: 游戏最大轮数
- `CHINESE_TEST_MODELS`: 参与测试的模型列表

## 评估指标

### 成功率指标
- **总体成功率**: 所有游戏的成功比例
- **按模型分析**: 各模型作为hinter和guesser的表现
- **按词性分析**: 不同词性词汇的难度差异

### 失败分析
- **违反禁用词**: TABOO_VIOLATION
- **格式错误**: FORMAT_FAILURE  
- **API失败**: API_FAILURE
- **轮数耗尽**: MAX_TURNS_EXCEEDED

### 效率指标
- **平均轮数**: 成功游戏的平均轮数
- **API调用次数**: 总的API调用统计
- **游戏时长**: 每个游戏的执行时间

## 技术创新点

1. **首次应用OpenHowNet**: 将中文知识图谱用于Taboo游戏数据集构建
2. **中文语言适配**: 专门的中文格式检查和文本处理
3. **语义关系利用**: 通过同义词、相似度等多种方法生成高质量禁用词
4. **完整评估框架**: 提供从数据集构建到模型评估的完整pipeline

## 扩展方向

### 数据集扩展
- 增加词汇数量到300-500个
- 添加专业领域词汇（医学、法律、科技等）
- 引入难度分级机制

### 模型测试
- 集成更多中文模型（智谱GLM、百川、文心一言等）
- 对比不同模型在中文语义理解上的差异
- 研究中英文模型的跨语言表现

### 分析功能
- 中英文Taboo游戏对比研究
- 词汇难度与模型表现的关系分析
- 语义相似度对游戏成功率的影响

## 注意事项

1. **网络要求**: OpenHowNet首次使用需要下载数据，确保网络连接稳定
2. **API成本**: 实验会调用多个付费API，注意控制测试规模
3. **中文编码**: 所有文件使用UTF-8编码，结果文件使用UTF-8-BOM确保Excel兼容
4. **计算资源**: 数据集构建过程需要一定的计算时间，特别是相似度计算部分

## 故障排除

### OpenHowNet安装问题
```bash
# 如果安装失败，尝试升级pip
pip install --upgrade pip
pip install OpenHowNet

# 或者使用conda
conda install -c conda-forge openhownet
```

### API调用失败
- 检查api_keys.json文件是否正确配置
- 确认OpenRouter账户余额充足
- 验证模型名称是否正确

### 中文显示问题
- 确保Jupyter环境支持中文显示
- 检查系统中文字体配置
- 使用UTF-8编码保存和读取文件

## 贡献

欢迎提交Issue和Pull Request来改进这个中文Taboo实验系统！ 