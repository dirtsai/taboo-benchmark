#!/usr/bin/env python3
"""
Kimi Model Taboo Experiment
修改base_test copy实验，使kimi模型作为hinter和guesser和其他4个模型跑一轮quick80_dataset数据集
总共800场游戏
"""

import json
import pandas as pd
import random
import time
import requests
import os
from typing import Dict, List, Any
from datetime import datetime

# 设置随机种子确保可复现
random.seed(240)

def load_dataset(dataset_path: str = "data/quick80_dataset.json") -> List[Dict]:
    """加载Quick80数据集"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_api_keys(keys_file: str = "api_keys.json") -> Dict[str, str]:
    """加载API密钥"""
    with open(keys_file, 'r', encoding='utf-8') as f:
        return json.load(f)

class OpenRouterClient:
    """OpenRouter API客户端"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def chat_completion(self, model: str, messages: List[Dict], max_tokens: int = 150, temperature: float = 0.7):
        """调用聊天完成API"""
        url = f"{self.base_url}/chat/completions"
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API调用失败: {e}")
            return None

def safe_text_cleanup(text: str, max_length: int = 200) -> str:
    """安全清理文本，防止乱码和超长内容"""
    if not text or not isinstance(text, str):
        return "EMPTY_RESPONSE"
    
    # 移除多余空白字符
    text = ' '.join(text.split())
    
    # 截断过长内容
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text

def extract_clue_from_response(response: str) -> str:
    """从响应中提取线索"""
    if not response or not isinstance(response, str):
        return "FORMAT_ERROR"
    
    if '[CLUE]' in response.upper():
        import re
        match = re.search(r'\[CLUE\]\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        if match:
            return safe_text_cleanup(match.group(1).strip())
    
    return safe_text_cleanup(response)

def extract_guess_from_response(response: str) -> str:
    """从响应中提取猜测"""
    if not response or not isinstance(response, str):
        return "FORMAT_ERROR"
    
    if '[GUESS]' in response.upper():
        import re
        match = re.search(r'\[GUESS\]\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        if match:
            return safe_text_cleanup(match.group(1).strip())
    
    return safe_text_cleanup(response)

def check_taboo_violation(text: str, taboo_words: List[str]) -> bool:
    """检查是否违反禁用词规则"""
    if not text or not isinstance(text, str):
        return False
    
    text_lower = text.lower()
    for taboo in taboo_words:
        if taboo.lower() in text_lower:
            return True
    return False

def enhanced_play_taboo_game(client, hinter_model, guesser_model, target_word, taboo_words, max_turns=5):
    """增强版Taboo游戏，包含taboo violation检查"""
    conversation_history = []
    format_errors = 0
    hinter_failed_outputs = []
    guesser_failed_outputs = []
    
    # 初始化系统消息
    hinter_system = f"""你是Taboo游戏的提示者(Hinter)。你的任务是帮助猜测者猜出目标词"{target_word}"。

规则：
1. 你不能直接说出目标词"{target_word}"
2. 你不能使用以下禁用词：{', '.join(taboo_words)}
3. 每次只能给出一个简短的线索
4. 请用[CLUE]标签包围你的线索

示例格式：[CLUE] 这是一种动物"""

    guesser_system = f"""你是Taboo游戏的猜测者(Guesser)。根据提示者的线索猜出目标词。

规则：
1. 根据线索进行猜测
2. 每次只能猜一个词
3. 请用[GUESS]标签包围你的猜测

示例格式：[GUESS] 猫"""

    for turn in range(1, max_turns + 1):
        # Hinter给出线索
        hinter_messages = [
            {"role": "system", "content": hinter_system},
            {"role": "user", "content": f"请为目标词'{target_word}'给出第{turn}个线索。之前的对话：{' '.join(conversation_history[-4:]) if conversation_history else '无'}"}
        ]
        
        hinter_response = client.chat_completion(hinter_model, hinter_messages)
        if not hinter_response or 'choices' not in hinter_response:
            return {
                'success': False,
                'turns': turn,
                'final_guess': 'API_ERROR',
                'error': f'Hinter API调用失败 (第{turn}轮)',
                'failure_reason': 'API_ERROR'
            }
        
        hint_text = extract_clue_from_response(hinter_response['choices'][0]['message']['content'])
        
        if hint_text == "FORMAT_ERROR":
            format_errors += 1
            hinter_failed_outputs.append(hinter_response['choices'][0]['message']['content'])
        
        # 检查Hinter是否违反禁用词规则
        if check_taboo_violation(hint_text, taboo_words + [target_word]):
            return {
                'success': False,
                'turns': turn,
                'final_guess': 'TABOO_VIOLATION: Hinter违反规则',
                'error': f'TABOO_VIOLATION: Hinter在第{turn}轮违反规则，说了禁用词: {hint_text}',
                'failure_reason': 'TABOO_VIOLATION',
                'taboo_violation_turn': turn,
                'taboo_violation_hint': hint_text,
                'format_errors': format_errors,
                'hinter_failed_outputs': hinter_failed_outputs,
                'guesser_failed_outputs': guesser_failed_outputs,
                'conversation_history': conversation_history
            }
        
        conversation_history.append(f"Hinter: {hint_text}")
        
        # Guesser进行猜测
        guesser_messages = [
            {"role": "system", "content": guesser_system},
            {"role": "user", "content": f"根据以下线索猜测目标词：{' '.join(conversation_history)}"}
        ]
        
        guesser_response = client.chat_completion(guesser_model, guesser_messages)
        if not guesser_response or 'choices' not in guesser_response:
            return {
                'success': False,
                'turns': turn,
                'final_guess': 'API_ERROR',
                'error': f'Guesser API调用失败 (第{turn}轮)',
                'failure_reason': 'API_ERROR'
            }
        
        guess_text = extract_guess_from_response(guesser_response['choices'][0]['message']['content'])
        
        if guess_text == "FORMAT_ERROR":
            format_errors += 1
            guesser_failed_outputs.append(guesser_response['choices'][0]['message']['content'])
        
        conversation_history.append(f"Guesser: {guess_text}")
        
        # 检查是否猜对
        if guess_text.lower().strip() == target_word.lower().strip():
            return {
                'success': True,
                'turns': turn,
                'final_guess': guess_text,
                'conversation_history': conversation_history,
                'format_errors': format_errors,
                'hinter_failed_outputs': hinter_failed_outputs,
                'guesser_failed_outputs': guesser_failed_outputs,
                'all_hints': [msg for msg in conversation_history if msg.startswith('Hinter:')],
                'all_guesses': [msg for msg in conversation_history if msg.startswith('Guesser:')]
            }
        
        # 短暂延迟避免API限制
        time.sleep(0.5)
    
    # 游戏结束，未猜对
    return {
        'success': False,
        'turns': max_turns,
        'final_guess': guess_text,
        'conversation_history': conversation_history,
        'failure_reason': 'MAX_TURNS_REACHED',
        'format_errors': format_errors,
        'hinter_failed_outputs': hinter_failed_outputs,
        'guesser_failed_outputs': guesser_failed_outputs,
        'all_hints': [msg for msg in conversation_history if msg.startswith('Hinter:')],
        'all_guesses': [msg for msg in conversation_history if msg.startswith('Guesser:')]
    }

def run_kimi_experiment(client, dataset, config):
    """运行Kimi模型实验"""
    
    # Kimi模型配置
    kimi_model = "moonshot/moonshot-v1-8k"
    
    # 其他4个模型
    other_models = [
        "openai/gpt-4o",
        "google/gemini-2.5-pro", 
        "deepseek/deepseek-chat-v3-0324",
        "anthropic/claude-sonnet-4"
    ]
    
    experiment_type = config.get('experiment_type', 'kimi_experiment')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建实验目录
    exp_dir = f"experiments/kimi_experiment_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"🚀 开始Kimi模型实验")
    print(f"📊 实验类型: {experiment_type}")
    print(f"🤖 Kimi模型: {kimi_model}")
    print(f"🎯 其他模型: {len(other_models)} 个")
    for i, model in enumerate(other_models, 1):
        print(f"   {i}. {model}")
    print(f"📚 数据集大小: {len(dataset)} 条")
    print(f"🎮 总游戏数: {len(dataset) * len(other_models) * 2} (Kimi作为hinter和guesser)")
    
    all_results = []
    game_count = 0
    
    # Kimi作为Hinter，其他模型作为Guesser
    print(f"\n🎯 第1阶段: Kimi作为Hinter")
    for i, guesser_model in enumerate(other_models, 1):
        print(f"\n   对战模型 {i}/{len(other_models)}: {guesser_model}")
        
        for j, item in enumerate(dataset, 1):
            game_count += 1
            target_word = item['target']
            taboo_words = item['taboo']
            
            print(f"      游戏 {j}/{len(dataset)}: {target_word} (总进度: {game_count}/{len(dataset) * len(other_models) * 2})")
            
            try:
                game_result = enhanced_play_taboo_game(
                    client, kimi_model, guesser_model, target_word, taboo_words
                )
                
                result = {
                    'game_id': game_count,
                    'hinter_model': kimi_model,
                    'guesser_model': guesser_model,
                    'target_word': target_word,
                    'taboo_words': taboo_words,
                    'success': game_result['success'],
                    'turns': game_result['turns'],
                    'final_guess': game_result['final_guess'],
                    'experiment_type': experiment_type,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'phase': 'kimi_as_hinter'
                }
                
                if 'error' in game_result:
                    result['error'] = game_result['error']
                if 'failure_reason' in game_result:
                    result['failure_reason'] = game_result['failure_reason']
                
                all_results.append(result)
                
            except Exception as e:
                print(f"         ❌ 游戏执行失败: {e}")
                result = {
                    'game_id': game_count,
                    'hinter_model': kimi_model,
                    'guesser_model': guesser_model,
                    'target_word': target_word,
                    'taboo_words': taboo_words,
                    'success': False,
                    'turns': 0,
                    'final_guess': 'EXECUTION_ERROR',
                    'error': str(e),
                    'failure_reason': 'EXECUTION_ERROR',
                    'experiment_type': experiment_type,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'phase': 'kimi_as_hinter'
                }
                all_results.append(result)
            
            # 每50个游戏保存一次中间结果
            if game_count % 50 == 0:
                intermediate_df = pd.DataFrame(all_results)
                intermediate_path = f"{exp_dir}/intermediate_results_{game_count}.csv"
                intermediate_df.to_csv(intermediate_path, index=False, encoding='utf-8')
                print(f"         💾 中间结果已保存: {intermediate_path}")
    
    # Kimi作为Guesser，其他模型作为Hinter
    print(f"\n🎯 第2阶段: Kimi作为Guesser")
    for i, hinter_model in enumerate(other_models, 1):
        print(f"\n   对战模型 {i}/{len(other_models)}: {hinter_model}")
        
        for j, item in enumerate(dataset, 1):
            game_count += 1
            target_word = item['target']
            taboo_words = item['taboo']
            
            print(f"      游戏 {j}/{len(dataset)}: {target_word} (总进度: {game_count}/{len(dataset) * len(other_models) * 2})")
            
            try:
                game_result = enhanced_play_taboo_game(
                    client, hinter_model, kimi_model, target_word, taboo_words
                )
                
                result = {
                    'game_id': game_count,
                    'hinter_model': hinter_model,
                    'guesser_model': kimi_model,
                    'target_word': target_word,
                    'taboo_words': taboo_words,
                    'success': game_result['success'],
                    'turns': game_result['turns'],
                    'final_guess': game_result['final_guess'],
                    'experiment_type': experiment_type,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'phase': 'kimi_as_guesser'
                }
                
                if 'error' in game_result:
                    result['error'] = game_result['error']
                if 'failure_reason' in game_result:
                    result['failure_reason'] = game_result['failure_reason']
                
                all_results.append(result)
                
            except Exception as e:
                print(f"         ❌ 游戏执行失败: {e}")
                result = {
                    'game_id': game_count,
                    'hinter_model': hinter_model,
                    'guesser_model': kimi_model,
                    'target_word': target_word,
                    'taboo_words': taboo_words,
                    'success': False,
                    'turns': 0,
                    'final_guess': 'EXECUTION_ERROR',
                    'error': str(e),
                    'failure_reason': 'EXECUTION_ERROR',
                    'experiment_type': experiment_type,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'phase': 'kimi_as_guesser'
                }
                all_results.append(result)
            
            # 每50个游戏保存一次中间结果
            if game_count % 50 == 0:
                intermediate_df = pd.DataFrame(all_results)
                intermediate_path = f"{exp_dir}/intermediate_results_{game_count}.csv"
                intermediate_df.to_csv(intermediate_path, index=False, encoding='utf-8')
                print(f"         💾 中间结果已保存: {intermediate_path}")
    
    # 保存最终结果
    final_df = pd.DataFrame(all_results)
    final_path = f"{exp_dir}/kimi_experiment_final_results.csv"
    final_df.to_csv(final_path, index=False, encoding='utf-8')
    
    # 生成统计报告
    generate_kimi_report(final_df, exp_dir, kimi_model, other_models)
    
    print(f"\n✅ Kimi实验完成!")
    print(f"📊 总游戏数: {len(all_results)}")
    print(f"📁 结果保存至: {final_path}")
    
    return final_df

def generate_kimi_report(df, exp_dir, kimi_model, other_models):
    """生成Kimi实验报告"""
    
    report_path = f"{exp_dir}/kimi_experiment_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Kimi模型Taboo游戏实验报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Kimi模型: {kimi_model}\n")
        f.write(f"对战模型: {', '.join(other_models)}\n")
        f.write(f"总游戏数: {len(df)}\n\n")
        
        # 整体统计
        success_rate = df['success'].mean() * 100
        f.write(f"整体成功率: {success_rate:.1f}%\n")
        f.write(f"成功游戏数: {df['success'].sum()}\n")
        f.write(f"失败游戏数: {(~df['success']).sum()}\n\n")
        
        # 按阶段统计
        f.write("按阶段统计:\n")
        f.write("-" * 40 + "\n")
        
        for phase in ['kimi_as_hinter', 'kimi_as_guesser']:
            phase_df = df[df['phase'] == phase]
            if len(phase_df) > 0:
                phase_success_rate = phase_df['success'].mean() * 100
                phase_name = "Kimi作为Hinter" if phase == 'kimi_as_hinter' else "Kimi作为Guesser"
                f.write(f"{phase_name}:\n")
                f.write(f"  成功率: {phase_success_rate:.1f}%\n")
                f.write(f"  游戏数: {len(phase_df)}\n")
                f.write(f"  成功: {phase_df['success'].sum()}\n")
                f.write(f"  失败: {(~phase_df['success']).sum()}\n\n")
        
        # 按模型对战统计
        f.write("按模型对战统计:\n")
        f.write("-" * 40 + "\n")
        
        # Kimi作为Hinter的表现
        f.write("Kimi作为Hinter vs 其他模型:\n")
        kimi_hinter_df = df[df['phase'] == 'kimi_as_hinter']
        for model in other_models:
            model_df = kimi_hinter_df[kimi_hinter_df['guesser_model'] == model]
            if len(model_df) > 0:
                success_rate = model_df['success'].mean() * 100
                f.write(f"  vs {model}: {success_rate:.1f}% ({model_df['success'].sum()}/{len(model_df)})\n")
        
        f.write("\nKimi作为Guesser vs 其他模型:\n")
        kimi_guesser_df = df[df['phase'] == 'kimi_as_guesser']
        for model in other_models:
            model_df = kimi_guesser_df[kimi_guesser_df['hinter_model'] == model]
            if len(model_df) > 0:
                success_rate = model_df['success'].mean() * 100
                f.write(f"  vs {model}: {success_rate:.1f}% ({model_df['success'].sum()}/{len(model_df)})\n")
        
        # 失败原因分析
        f.write("\n失败原因分析:\n")
        f.write("-" * 40 + "\n")
        failed_df = df[~df['success']]
        if len(failed_df) > 0:
            failure_reasons = failed_df['failure_reason'].value_counts()
            for reason, count in failure_reasons.items():
                percentage = count / len(failed_df) * 100
                f.write(f"  {reason}: {count} ({percentage:.1f}%)\n")
        
        # 轮次统计
        f.write("\n轮次统计:\n")
        f.write("-" * 40 + "\n")
        successful_df = df[df['success']]
        if len(successful_df) > 0:
            avg_turns = successful_df['turns'].mean()
            f.write(f"成功游戏平均轮次: {avg_turns:.1f}\n")
            
            turn_distribution = successful_df['turns'].value_counts().sort_index()
            for turn, count in turn_distribution.items():
                percentage = count / len(successful_df) * 100
                f.write(f"  {turn}轮成功: {count} ({percentage:.1f}%)\n")
    
    print(f"📋 实验报告已生成: {report_path}")

def main():
    """主函数"""
    print("🚀 启动Kimi模型Taboo游戏实验")
    print("=" * 50)
    
    # 加载数据集
    print("📚 正在加载Quick80数据集...")
    try:
        dataset = load_dataset("data/quick80_dataset.json")
        print(f"✅ 数据集加载完成，共{len(dataset)}条记录")
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return
    
    # 初始化API客户端
    print("\n🔑 正在初始化API客户端...")
    try:
        api_keys = load_api_keys()
        client = OpenRouterClient(api_keys["OPENROUTER_API_KEY"])
        print("✅ API客户端初始化成功")
    except Exception as e:
        print(f"❌ API客户端初始化失败: {e}")
        return
    
    # 实验配置
    config = {
        'experiment_type': 'kimi_experiment',
        'max_turns': 5,
        'save_intermediate': True
    }
    
    # 运行实验
    print(f"\n🎮 开始运行实验...")
    try:
        results = run_kimi_experiment(client, dataset, config)
        print(f"\n🎉 实验成功完成!")
        print(f"📊 总游戏数: {len(results)}")
        print(f"✅ 成功率: {results['success'].mean()*100:.1f}%")
        
    except Exception as e:
        print(f"❌ 实验执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
