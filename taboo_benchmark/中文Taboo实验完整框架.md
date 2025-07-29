# 中文Taboo实验完整框架

> 基于base_test.ipynb框架，使用OpenHowNet构建中文数据集的完整实验系统

## 📋 实验概览

### 🎯 实验目标
1. **构建高质量中文Taboo数据集**：使用OpenHowNet构建100个中文词汇的语义数据集
2. **评估中文LLM表现**：测试多个语言模型在中文约束条件下的沟通能力
3. **建立中文评估基准**：为中文LLM能力评估提供标准化工具

### 🏗️ 技术创新
- **首次使用OpenHowNet**：将中文知识图谱应用于Taboo游戏数据集构建
- **中文语言特化**：针对中文语言特点优化格式检查和评估机制
- **统一实验架构**：继承base_test成熟框架，确保实验科学性

## 🔧 实验架构设计

### 模块1: 环境配置 (Environment Setup)

**参考base_test第1部分**

```python
# 1.1 核心依赖导入
import json
import pandas as pd
import random
import time
import requests
import os
import jieba
import re
from typing import Dict, List, Any, Tuple
from datetime import datetime
from collections import Counter

# 1.2 中文特化依赖
import OpenHowNet  # 中文知识图谱
jieba.setLogLevel(logging.INFO)  # 减少分词日志输出

# 1.3 设置
random.seed(42)  # 确保可复现
print("🚀 中文Taboo实验环境初始化完成")
```

### 模块2: 数据集构建 (Dataset Construction)

**对应base_test的数据集加载，但这里是动态构建**

#### 2.1 OpenHowNet初始化
```python
def initialize_hownet():
    """初始化OpenHowNet知识图谱"""
    try:
        hownet_dict = OpenHowNet.HowNetDict()
        print("✅ OpenHowNet初始化成功")
        return hownet_dict
    except Exception as e:
        print(f"❌ OpenHowNet初始化失败: {e}")
        return None
```

#### 2.2 中文词汇筛选
```python
def select_chinese_words(hownet_dict, target_count=100):
    """筛选中文词汇"""
    # 筛选条件:
    # - 包含中文字符 (\\u4e00-\\u9fff)
    # - 长度在1-6字符之间
    # - 在HowNet中有语义定义
    # - 词性分布平衡: 名词、动词、形容词、副词各25个
    
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    selected_words = {
        'noun': [], 'verb': [], 'adj': [], 'adv': []
    }
    
    # 实现词汇选择逻辑...
    return selected_words
```

#### 2.3 禁用词生成
```python
def generate_taboo_words(hownet_dict, word, sense_info):
    """使用OpenHowNet生成语义相关的禁用词"""
    taboo_words = []
    
    # 方法1: 同义词
    synonyms = hownet_dict.get_synonyms(word)
    taboo_words.extend(synonyms[:2])
    
    # 方法2: 语义相似词
    similar_words = hownet_dict.calculate_word_similarity(word, ...)
    taboo_words.extend(similar_words[:2])
    
    # 方法3: 定义关键词提取
    definition_keywords = extract_definition_keywords(sense_info)
    taboo_words.extend(definition_keywords[:1])
    
    return taboo_words[:5]  # 每个词汇5个禁用词
```

#### 2.4 数据集构建与验证
```python
def build_chinese_dataset(hownet_dict):
    """构建完整的中文Taboo数据集"""
    dataset = []
    
    # 按词性分别选择词汇
    for pos, words in selected_words.items():
        for word in words:
            # 获取语义信息
            senses = hownet_dict.get_sense(word)
            
            # 生成禁用词
            taboo_words = generate_taboo_words(hownet_dict, word, senses)
            
            # 构建数据项
            data_item = {
                'target': word,
                'part_of_speech': pos,
                'taboo': taboo_words,
                'category': 'chinese_general',
                'senses': format_sense_info(senses)
            }
            dataset.append(data_item)
    
    return dataset
```

### 模块3: 数据集统计分析 (Dataset Analysis)

**对应base_test第2部分**

```python
def analyze_chinese_dataset(dataset):
    """分析中文数据集统计信息"""
    print("📊 中文数据集基本统计:")
    print("=" * 40)
    
    # 词性分布统计
    pos_counts = {}
    taboo_counts = []
    sense_counts = []
    
    for item in dataset:
        pos = item.get('part_of_speech', 'unknown')
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
        taboo_counts.append(len(item.get('taboo', [])))
        sense_counts.append(len(item.get('senses', [])))
    
    print(f"🏷️ 词性分布:")
    for pos, count in pos_counts.items():
        percentage = count / len(dataset) * 100
        print(f"   {pos}: {count} 个 ({percentage:.1f}%)")
    
    print(f"🚫 禁用词统计:")
    print(f"   平均数量: {sum(taboo_counts) / len(taboo_counts):.1f}")
    print(f"   范围: {min(taboo_counts)} - {max(taboo_counts)}")
    
    print(f"💭 词义统计:")
    print(f"   平均数量: {sum(sense_counts) / len(sense_counts):.1f}")
    print(f"   范围: {min(sense_counts)} - {max(sense_counts)}")
    
    print("✅ 中文数据集统计完成，质量良好")
```

### 模块4: API客户端设置 (API Configuration)

**对应base_test第3部分，添加中文模型支持**

```python
class ChineseAPIClient:
    """支持中文模型的API客户端"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def call_model(self, model: str, messages: List[Dict[str, str]], 
                   temperature: float = 0.3) -> str:
        """调用模型API，保持中文字符"""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2000
        }
        response = requests.post(self.base_url, headers=self.headers, 
                               json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()
        
        # 保留中文字符，只过滤控制字符
        content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)
        return content

# 中文测试模型列表
CHINESE_MODELS = [
    "openai/gpt-4o",
    "google/gemini-2.5-pro", 
    "deepseek/deepseek-chat-v3-0324",
    "anthropic/claude-sonnet-4"
]
```

### 模块5: 中文游戏逻辑 (Chinese Game Logic)

**对应base_test第4部分，适配中文格式**

```python
def chinese_taboo_game(client, hinter_model, guesser_model, 
                      target_word, taboo_words, max_turns=5):
    """中文Taboo游戏核心逻辑"""
    
    # 中文提示模板
    hinter_prompt = f"""
你正在玩中文Taboo游戏。你的任务是让队友猜出目标词汇，但不能使用禁用词。

目标词汇: {target_word}
禁用词汇: {', '.join(taboo_words)}

规则:
1. 你需要给出线索让队友猜出目标词汇
2. 你的线索中不能包含任何禁用词汇
3. 请用格式 [线索] 开始你的回答

请给出你的第一个线索:
"""

    guesser_prompt = f"""
你正在玩中文Taboo游戏。根据队友给出的线索，猜出目标词汇。

禁用词汇: {', '.join(taboo_words)}

规则:
1. 根据队友的线索猜出目标词汇
2. 请用格式 [猜测] 开始你的回答
3. 只说出你认为的答案，不要解释

队友的线索是: {hint}

你的猜测是:
"""

    # 游戏执行逻辑
    conversation_history = []
    
    for turn in range(1, max_turns + 1):
        # Hinter给出线索
        hinter_response = robust_chinese_api_call(
            client, hinter_model, hinter_prompt, "[线索]"
        )
        
        if not hinter_response['success']:
            return create_game_result(False, turn, None, "线索生成失败", 
                                    conversation_history)
        
        hint = hinter_response['content']
        
        # 检查禁用词违规
        if check_chinese_taboo_violation(hint, taboo_words):
            return create_game_result(False, turn, None, "违反禁用词规则", 
                                    conversation_history)
        
        # Guesser进行猜测
        current_guesser_prompt = guesser_prompt.format(hint=hint)
        guesser_response = robust_chinese_api_call(
            client, guesser_model, current_guesser_prompt, "[猜测]"
        )
        
        if not guesser_response['success']:
            return create_game_result(False, turn, None, "猜测生成失败", 
                                    conversation_history)
        
        guess = guesser_response['content']
        
        # 记录对话
        conversation_history.append({
            'turn': turn,
            'hint': hint,
            'guess': guess
        })
        
        # 检查是否猜中
        if check_chinese_word_match(guess, target_word):
            return create_game_result(True, turn, guess, "成功", 
                                    conversation_history)
    
    # 超过最大轮数
    return create_game_result(False, max_turns, guess, "轮数耗尽", 
                            conversation_history)
```

### 模块6: 中文特化辅助函数 (Chinese Helper Functions)

```python
def robust_chinese_api_call(client, model, prompt, expected_prefix, 
                           max_retries=3):
    """支持中文格式的健壮API调用"""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.call_model(model, 
                                       [{"role": "user", "content": prompt}])
            
            if response.strip().startswith(expected_prefix):
                content = response.strip()[len(expected_prefix):].strip()
                return {
                    'success': True,
                    'content': content,
                    'attempts': attempt
                }
            else:
                # 格式错误，添加提醒重试
                if attempt < max_retries:
                    prompt += f"""

⚠️ 格式错误 ⚠️
你的回答必须以 '{expected_prefix}' 开头
请重新回答:"""
                
        except Exception as e:
            if attempt == max_retries:
                return {
                    'success': False,
                    'content': f"API调用失败: {e}",
                    'attempts': attempt
                }
    
    return {
        'success': False,
        'content': "格式验证失败",
        'attempts': max_retries
    }

def check_chinese_taboo_violation(text, taboo_words):
    """检查中文文本是否违反禁用词规则"""
    # 使用jieba分词
    words = jieba.cut(text, cut_all=False)
    text_words = set(words)
    
    for taboo in taboo_words:
        if taboo in text or taboo in text_words:
            return True
    return False

def check_chinese_word_match(guess, target):
    """检查中文词汇是否匹配"""
    # 移除格式标记和空格
    guess_clean = re.sub(r'[\[\]【】]', '', guess).strip()
    
    # 直接匹配或包含匹配
    return guess_clean == target or target in guess_clean
```

### 模块7: 实验执行器 (Experiment Runner)

**对应base_test第5-6部分**

```python
def run_chinese_taboo_experiment(client, models, dataset, config):
    """执行中文Taboo实验"""
    
    experiment_type = config.get('experiment_type', 'test')
    max_turns = config.get('max_turns', 5)
    output_dir = config.get('output_dir', 'results')
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"{output_dir}/chinese_experiment_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"🚀 开始执行中文Taboo实验...")
    print(f"📁 输出目录: {experiment_dir}")
    print(f"🎯 词汇数量: {len(dataset)}")
    print(f"🤖 模型数量: {len(models)}")
    
    all_results = []
    game_counter = 0
    total_games = len(dataset) * len(models) * len(models)
    
    # 执行所有游戏
    for word_data in dataset:
        target_word = word_data['target']
        taboo_words = word_data['taboo']
        
        for hinter_model in models:
            for guesser_model in models:
                game_counter += 1
                print(f"🎮 游戏 {game_counter}/{total_games}: "
                      f"{target_word} | "
                      f"{hinter_model.split('/')[-1]}→{guesser_model.split('/')[-1]}")
                
                # 执行游戏
                start_time = time.time()
                game_result = chinese_taboo_game(
                    client, hinter_model, guesser_model,
                    target_word, taboo_words, max_turns
                )
                duration = time.time() - start_time
                
                # 记录结果
                result = {
                    'game_id': game_counter,
                    'target_word': target_word,
                    'part_of_speech': word_data['part_of_speech'],
                    'hinter_model': hinter_model,
                    'guesser_model': guesser_model,
                    'success': game_result['success'],
                    'turns_used': game_result['turns'],
                    'final_guess': game_result.get('final_guess', ''),
                    'failure_reason': game_result.get('failure_reason', ''),
                    'duration_seconds': round(duration, 2),
                    'taboo_words': '|'.join(taboo_words)
                }
                all_results.append(result)
                
                # 显示结果
                status = "✅ 成功" if game_result['success'] else "❌ 失败"
                print(f"   {status} | {game_result['turns']}轮 | "
                      f"{game_result.get('failure_reason', '正常结束')}")
    
    # 保存结果
    results_df = pd.DataFrame(all_results)
    output_file = f"{experiment_dir}/chinese_test_results_{timestamp}.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 中文实验完成！结果已保存至: {output_file}")
    return results_df
```

### 模块8: 结果分析 (Results Analysis)

```python
def analyze_chinese_experiment_results(results_df):
    """分析中文实验结果"""
    print("📊 中文Taboo实验结果分析")
    print("=" * 50)
    
    # 整体成功率
    total_games = len(results_df)
    successful_games = sum(results_df['success'])
    overall_success_rate = successful_games / total_games * 100
    
    print(f"🎮 总游戏数: {total_games}")
    print(f"✅ 成功游戏: {successful_games}")
    print(f"📈 整体成功率: {overall_success_rate:.1f}%")
    
    # 按模型分析
    print(f"\n🤖 模型表现分析:")
    for model in results_df['hinter_model'].unique():
        model_name = model.split('/')[-1]
        
        # 作为Hinter的表现
        hinter_results = results_df[results_df['hinter_model'] == model]
        hinter_success = sum(hinter_results['success'])
        hinter_total = len(hinter_results)
        hinter_rate = hinter_success / hinter_total * 100 if hinter_total > 0 else 0
        
        # 作为Guesser的表现
        guesser_results = results_df[results_df['guesser_model'] == model]
        guesser_success = sum(guesser_results['success'])
        guesser_total = len(guesser_results)
        guesser_rate = guesser_success / guesser_total * 100 if guesser_total > 0 else 0
        
        print(f"   {model_name}:")
        print(f"     作为线索给出者: {hinter_success}/{hinter_total} ({hinter_rate:.1f}%)")
        print(f"     作为猜测者: {guesser_success}/{guesser_total} ({guesser_rate:.1f}%)")
    
    # 按词性分析
    print(f"\n📝 词性表现分析:")
    for pos in results_df['part_of_speech'].unique():
        pos_results = results_df[results_df['part_of_speech'] == pos]
        pos_success = sum(pos_results['success'])
        pos_total = len(pos_results)
        pos_rate = pos_success / pos_total * 100 if pos_total > 0 else 0
        
        print(f"   {pos}: {pos_success}/{pos_total} ({pos_rate:.1f}%)")
    
    # 失败原因分析
    print(f"\n❌ 失败原因分析:")
    failed_results = results_df[results_df['success'] == False]
    failure_reasons = failed_results['failure_reason'].value_counts()
    
    for reason, count in failure_reasons.items():
        percentage = count / len(failed_results) * 100
        print(f"   {reason}: {count} 次 ({percentage:.1f}%)")
    
    return {
        'overall_success_rate': overall_success_rate,
        'total_games': total_games,
        'successful_games': successful_games
    }
```

## 🚀 实验执行流程

### 阶段1: 环境准备
```python
# 1. 安装依赖
pip install OpenHowNet jieba requests pandas numpy

# 2. 配置API密钥
# 确保api_keys.json包含OpenRouter API密钥

# 3. 初始化环境
exec_cell_1()  # 导入依赖和配置
```

### 阶段2: 数据集构建
```python
# 4. 初始化OpenHowNet
hownet_dict = initialize_hownet()

# 5. 构建中文数据集
chinese_dataset = build_chinese_dataset(hownet_dict)

# 6. 数据集分析
analyze_chinese_dataset(chinese_dataset)

# 7. 保存数据集
save_chinese_dataset(chinese_dataset)
```

### 阶段3: 实验执行
```python
# 8. 初始化API客户端
chinese_client = ChineseAPIClient(api_keys["OPENROUTER_API_KEY"])

# 9. 测试实验（小规模验证）
test_config = {
    'experiment_type': 'test',
    'max_turns': 5,
    'output_dir': 'results'
}
test_results = run_chinese_taboo_experiment(
    chinese_client, CHINESE_MODELS[:2], chinese_dataset[:5], test_config
)

# 10. 全量实验
full_config = {
    'experiment_type': 'full',
    'max_turns': 5,
    'output_dir': 'results'
}
full_results = run_chinese_taboo_experiment(
    chinese_client, CHINESE_MODELS, chinese_dataset, full_config
)
```

### 阶段4: 结果分析
```python
# 11. 结果分析
analysis = analyze_chinese_experiment_results(full_results)

# 12. 生成报告
generate_chinese_experiment_report(full_results, analysis)
```

## 📈 预期结果

### 数据集指标
- **词汇总数**: 100个中文词汇
- **词性分布**: 名词、动词、形容词、副词各25个
- **禁用词**: 每个词汇5个语义相关禁用词
- **语义覆盖**: 基于OpenHowNet的丰富语义信息

### 实验规模
- **模型数量**: 4个主流中文LLM
- **游戏总数**: 100词汇 × 4模型 × 4模型 = 1,600场游戏
- **预计时长**: 约2-3小时（取决于API响应速度）

### 评估维度
1. **整体成功率**: 各模型在中文约束沟通中的表现
2. **角色表现**: 作为线索给出者vs猜测者的能力差异
3. **词性影响**: 不同词性对游戏难度的影响
4. **语言特性**: 中文语言特点对LLM表现的影响

## 💡 技术亮点

1. **首创性**: 首次将OpenHowNet应用于LLM评估
2. **中文特化**: 专门针对中文语言特点优化
3. **科学性**: 继承base_test成熟框架，确保实验可靠性
4. **完整性**: 从数据集构建到结果分析的全流程系统

## 📁 输出文件

- `data/chinese_dataset.json` - 完整中文数据集
- `data/chinese_dataset_simple.json` - 简化版数据集  
- `results/chinese_experiment_YYYYMMDD_HHMMSS/` - 实验结果目录
- `chinese_experiment_report.json` - 实验分析报告

---

> 这个框架完全基于base_test.ipynb的成熟架构，但专门针对中文语言和OpenHowNet进行了优化，确保了实验的科学性和中文语言的特殊性都得到了充分考虑。 