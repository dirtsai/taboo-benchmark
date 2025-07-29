#!/usr/bin/env python3
"""
中文实验TabooScore综合评估分析 - 生成图4.27
Chinese TabooScore Comprehensive Evaluation Analysis - Figure 4.27
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
sns.set_theme(style="white", font_scale=1.2)
colors = sns.color_palette("magma", 6)

def load_and_process_chinese_data(csv_path):
    """加载并处理中文实验数据"""
    print("加载中文实验数据...")
    df = pd.read_csv(csv_path)
    
    # 清理模型名称
    df['hinter_clean'] = df['hinter_model'].str.replace('openai/', '').str.replace('google/', '').str.replace('deepseek/', '').str.replace('moonshotai/', '')
    df['guesser_clean'] = df['guesser_model'].str.replace('openai/', '').str.replace('google/', '').str.replace('deepseek/', '').str.replace('moonshotai/', '')
    
    print(f"✅ 数据加载完成，共 {len(df)} 条记录")
    return df

def calculate_chinese_taboo_scores(df):
    """计算中文实验TabooScore综合得分"""
    print("计算中文TabooScore...")
    
    # 中文实验模型映射
    model_mapping = {
        'gpt-4o': 'GPT-4o',
        'gemini-2.5-flash': 'Gemini-2.5-Flash', 
        'deepseek-chat-v3-0324': 'DeepSeek-V3',
        'kimi-k2': 'Kimi-K2'
    }
    
    results = []
    
    # 针对每个模型计算整体表现
    for model in ['gpt-4o', 'gemini-2.5-flash', 'deepseek-chat-v3-0324', 'kimi-k2']:
        model_name = model_mapping[model]
        
        # 计算该模型作为Hinter的成功率
        hinter_data = df[df['hinter_clean'] == model]
        hint_success = hinter_data['success'].mean() * 100 if len(hinter_data) > 0 else 0
        hint_avg_turns = hinter_data[hinter_data['success']]['turns_used'].mean() if len(hinter_data[hinter_data['success']]) > 0 else 5.0
        
        # 计算该模型作为Guesser的成功率  
        guesser_data = df[df['guesser_clean'] == model]
        guess_success = guesser_data['success'].mean() * 100 if len(guesser_data) > 0 else 0
        
        # 处理平均轮次为NaN的情况
        if np.isnan(hint_avg_turns):
            hint_avg_turns = 5.0
        
        # 计算TabooScore (使用45-40-15权重)
        # TabooScore = 0.45 * Hint_Succ + 0.40 * Guess_Succ + 0.15 * Efficiency
        # Efficiency = 100 * (6 - avg_turns) / 5  # 标准化到0-100
        efficiency = 100 * (6 - hint_avg_turns) / 5
        taboo_score = 0.45 * hint_success + 0.40 * guess_success + 0.15 * efficiency
        
        results.append({
            'Model': model_name,
            'Hint_Succ': hint_success,
            'Guess_Succ': guess_success, 
            'Avg_Turns': hint_avg_turns,
            'Efficiency': efficiency,
            'TabooScore': taboo_score
        })
    
    return pd.DataFrame(results)

def main():
    print("开始中文TabooScore综合评估分析...")
    
    # 确保figures目录存在
    os.makedirs('figures', exist_ok=True)
    
    # 设置数据路径
    csv_path = "results/chinese_full_experiment_20250717_222959/chinese_full_results_20250717_222959.csv"
    
    try:
        # 1. 加载中文数据
        df = load_and_process_chinese_data(csv_path)
        
        # 2. 计算TabooScore
        scores_df = calculate_chinese_taboo_scores(df)
        
        # 3. 生成图4.27: 中文TabooScore构成堆叠图
        print("生成图4.27: 中文TabooScore构成堆叠图...")
        
        # 计算各组成部分的贡献
        scores_df = scores_df.copy()
        scores_df['Hint_Contribution'] = scores_df['Hint_Succ'] * 0.45
        scores_df['Guess_Contribution'] = scores_df['Guess_Succ'] * 0.40
        scores_df['Efficiency_Contribution'] = scores_df['Efficiency'] * 0.15
        
        # 按TabooScore排序（从低到高，便于水平条形图显示）
        scores_df = scores_df.sort_values('TabooScore', ascending=True)
        
        plt.figure(figsize=(12, 6))
        
        # 创建堆叠条形图
        models = scores_df['Model']
        hint_contrib = scores_df['Hint_Contribution']
        guess_contrib = scores_df['Guess_Contribution']
        efficiency_contrib = scores_df['Efficiency_Contribution']
        
        # 使用magma调色板的不同色调
        bars1 = plt.barh(models, hint_contrib, 
                         label='Hint Success Rate (45%)', 
                         color=colors[1], alpha=0.8)
        bars2 = plt.barh(models, guess_contrib, left=hint_contrib,
                         label='Guess Success Rate (40%)', 
                         color=colors[3], alpha=0.8)
        bars3 = plt.barh(models, efficiency_contrib,
                         left=hint_contrib + guess_contrib,
                         label='Efficiency (15%)', 
                         color=colors[5], alpha=0.8)
        
        # 添加总分数值标签
        for i, (model, total_score) in enumerate(zip(models, scores_df['TabooScore'])):
            plt.text(total_score + 0.5, i, f'{total_score:.1f}',
                    ha='left', va='center', fontsize=12, fontweight='bold')
        
        # 设置坐标轴和标签
        plt.xlabel('Score Contribution', fontsize=14)
        plt.ylabel('Model', fontsize=14)
        
        # 设置X轴范围
        plt.xlim(0, max(scores_df['TabooScore']) + 10)
        
        # 添加图例（放在框外）
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11, frameon=True, fancybox=True, shadow=True)
        
        # 美化图表
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # 保存图4.27
        plt.savefig('figures/figure_4_27_chinese_taboo_score_breakdown.pdf', 
                    dpi=300, bbox_inches='tight')
        plt.savefig('figures/figure_4_27_chinese_taboo_score_breakdown.png', 
                    dpi=300, bbox_inches='tight')
        print("✓ 图4.27已保存")
        plt.show()
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
