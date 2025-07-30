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
    """计算中文实验TabooScore综合得分 - 使用指定数值"""
    print("使用指定的TabooScore数值...")
    
    # 使用用户提供的具体数值
    results = [
        {
            'Model': 'DeepSeek-V3',
            'Hint_Succ': 30.0,
            'Guess_Succ': 30.0,
            'TabooScore': 30.0
        },
        {
            'Model': 'Kimi-K2',
            'Hint_Succ': 17.5,
            'Guess_Succ': 27.5,
            'TabooScore': 22.5
        },
        {
            'Model': 'Gemini-2.5-Flash',
            'Hint_Succ': 17.5,
            'Guess_Succ': 22.5,
            'TabooScore': 20.0
        },
        {
            'Model': 'GPT-4o',
            'Hint_Succ': 17.5,
            'Guess_Succ': 17.5,
            'TabooScore': 17.5
        }
    ]
    
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
        
        # 输出详细表格
        print("\n" + "="*80)
        print("中文TabooScore综合评估结果表格")
        print("="*80)
        print(f"{'Model':<20} {'Hint_Succ':<12} {'Guess_Succ':<12} {'TabooScore':<12}")
        print("-"*80)
        
        # 按TabooScore从高到低排序显示
        display_df = scores_df.sort_values('TabooScore', ascending=False)
        for _, row in display_df.iterrows():
            print(f"{row['Model']:<20} {row['Hint_Succ']:<12.1f} {row['Guess_Succ']:<12.1f} {row['TabooScore']:<12.1f}")
        
        print("-"*80)
        print("公式: TabooScore = (Hint_Succ + Guess_Succ) / 2")
        print("="*80)
        
        # 生成LaTeX表格格式
        print("\nLaTeX表格格式:")
        print("\\begin{tabular}{|l|c|c|c|}")
        print("\\hline")
        print("Model & Hint-Succ \\% & Guess-Succ \\% & TabooScore \\\\")
        print("\\hline")
        for _, row in display_df.iterrows():
            hint_bold = "\\textbf{" + f"{row['Hint_Succ']:.1f}" + "}" if row['Hint_Succ'] == display_df['Hint_Succ'].max() else f"{row['Hint_Succ']:.1f}"
            guess_bold = "\\textbf{" + f"{row['Guess_Succ']:.1f}" + "}" if row['Guess_Succ'] == display_df['Guess_Succ'].max() else f"{row['Guess_Succ']:.1f}"
            score_bold = "\\textbf{" + f"{row['TabooScore']:.1f}" + "}" if row['TabooScore'] == display_df['TabooScore'].max() else f"{row['TabooScore']:.1f}"
            anchor_text = " (anchor)" if row['Model'] == 'GPT-4o' else ""
            print(f"{row['Model']}{anchor_text} & {hint_bold} & {guess_bold} & {score_bold} \\\\")
        print("\\hline")
        print("\\end{tabular}")
        print("\n")
        
        # 3. 生成图4.27: 中文TabooScore构成堆叠图
        print("生成图4.27: 中文TabooScore构成堆叠图...")
        
        # 计算各组成部分的贡献（50-50权重）
        scores_df = scores_df.copy()
        scores_df['Hint_Contribution'] = scores_df['Hint_Succ'] * 0.50
        scores_df['Guess_Contribution'] = scores_df['Guess_Succ'] * 0.50
        
        # 按TabooScore排序（从低到高，便于水平条形图显示）
        scores_df = scores_df.sort_values('TabooScore', ascending=True)
        
        plt.figure(figsize=(12, 6))
        
        # 创建堆叠条形图
        models = scores_df['Model']
        hint_contrib = scores_df['Hint_Contribution']
        guess_contrib = scores_df['Guess_Contribution']
        
        # 使用magma调色板的不同色调
        bars1 = plt.barh(models, hint_contrib, 
                         label='Hint Success Rate (50%)', 
                         color=colors[1], alpha=0.8)
        bars2 = plt.barh(models, guess_contrib, left=hint_contrib,
                         label='Guess Success Rate (50%)', 
                         color=colors[3], alpha=0.8)
        
        # 添加总分数值标签
        for i, (model, total_score) in enumerate(zip(models, scores_df['TabooScore'])):
            plt.text(total_score + 0.5, i, f'{total_score:.1f}',
                    ha='left', va='center', fontsize=12, fontweight='bold')
        
        # 设置坐标轴和标签
        plt.xlabel('Score Contribution', fontsize=14)
        plt.ylabel('Model', fontsize=14)
        
        # 设置X轴范围
        plt.xlim(0, max(scores_df['TabooScore']) + 5)
        
        # 添加图例（放在右侧框外）
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
