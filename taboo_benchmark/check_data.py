#!/usr/bin/env python3
import pandas as pd

# 读取中文数据
csv_path = "/Users/czl/Desktop/msc proj/code/taboo_benchmark/results/chinese_full_experiment_20250717_222959/chinese_full_results_20250717_222959.csv"
df = pd.read_csv(csv_path)

print("🔍 Checking Chinese Data...")
print(f"Total records: {len(df)}")
print(f"Overall success rate: {df['success'].mean()*100:.1f}%")

# 清理模型名称
df['hinter_clean'] = df['hinter_model'].str.replace('openai/', '').str.replace('google/', '').str.replace('deepseek/', '').str.replace('moonshotai/', '')
df['guesser_clean'] = df['guesser_model'].str.replace('openai/', '').str.replace('google/', '').str.replace('deepseek/', '').str.replace('moonshotai/', '')

models = ['gpt-4o', 'gemini-2.5-flash', 'deepseek-chat-v3-0324', 'kimi-k2']

print("\n📊 Hinter Success Rates (vs GPT-4o as Guesser):")
for model in models:
    if model == 'gpt-4o':
        continue
    data = df[(df['hinter_clean'] == model) & (df['guesser_clean'] == 'gpt-4o')]
    if len(data) > 0:
        success_rate = data['success'].mean() * 100
        success_count = data['success'].sum()
        print(f"  {model}: {success_count}/{len(data)} = {success_rate:.1f}%")
        
        # 显示一些具体例子
        successes = data[data['success'] == True].head(2)
        failures = data[data['success'] == False].head(2)
        
        print(f"    Success examples: {list(successes['target_word'])}")
        print(f"    Failure examples: {list(failures['target_word'])}")

print("\n📊 Raw Data Check:")
print("First few records:")
print(df[['target_word', 'hinter_clean', 'guesser_clean', 'success', 'final_guess']].head(10))

# 检查特定模型组合
print("\n🔍 Specific Model Combinations:")
for model in ['gemini-2.5-flash', 'deepseek-chat-v3-0324', 'kimi-k2']:
    subset = df[(df['hinter_clean'] == model) & (df['guesser_clean'] == 'gpt-4o')]
    print(f"\n{model} as Hinter vs GPT-4o:")
    print(f"  Total games: {len(subset)}")
    print(f"  Successes: {subset['success'].sum()}")
    print(f"  Success rate: {subset['success'].mean()*100:.2f}%")
    
    if len(subset) > 0:
        # 检查是否所有成功率都一样
        unique_words = subset['target_word'].unique()
        print(f"  Unique target words: {len(unique_words)}")
        
        # 按目标词分组检查
        word_stats = subset.groupby('target_word')['success'].agg(['count', 'sum', 'mean'])
        print(f"  Word-level success rates:")
        for word in unique_words[:5]:  # 显示前5个词
            stats = word_stats.loc[word]
            print(f"    {word}: {stats['sum']}/{stats['count']} = {stats['mean']*100:.1f}%")
