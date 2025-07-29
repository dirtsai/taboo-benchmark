#!/usr/bin/env python3
"""
中文实验结果分析 - 生成图4.20和图4.21
Chinese Experiment Analysis for Taboo Benchmark - Figures 4.20 & 4.21
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

def load_chinese_experiment_data():
    """加载中文实验数据"""
    try:
        results_path = "results/chinese_full_experiment_20250717_222959/chinese_full_results_20250717_222959.csv"
        df = pd.read_csv(results_path)
        print(f"✅ 成功加载中文实验结果: {len(df)} 条记录")
        print(f"目标词数量: {df['target_word'].nunique()}")
        print(f"Hinter模型: {list(df['hinter_model'].unique())}")
        print(f"Guesser模型: {list(df['guesser_model'].unique())}")
        
        return df
    except Exception as e:
        print(f"❌ 加载中文实验数据失败: {e}")
        return None

def clean_model_names(df):
    """清理模型名称"""
    label_map = {
        "openai/gpt-4o": "GPT-4o",
        "google/gemini-2.5-flash": "Gemini-2.5-Flash",
        "deepseek/deepseek-chat-v3-0324": "DeepSeek-V3",
        "moonshotai/kimi-k2": "Kimi-K2",
    }
    
    df['hinter_model_clean'] = df['hinter_model'].map(label_map).fillna(df['hinter_model'])
    df['guesser_model_clean'] = df['guesser_model'].map(label_map).fillna(df['guesser_model'])
    
    return df

def plot_success_rate_bar(df, group_key, title_suffix, figure_num):
    """绘制成功率柱状图"""
    # 计算成功率并排序
    success_rate = (
        df.groupby(group_key)['success']
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    success_rate.columns = ['Model', 'Success_Rate']
    
    plt.figure(figsize=(10, 6))
    
    # 绘制柱状图
    bars = plt.bar(
        success_rate['Model'],
        success_rate['Success_Rate'],
        color=colors[1],
        alpha=0.8,
        width=0.6
    )
    
    plt.xlabel(f"{title_suffix} Model", fontsize=14)
    plt.ylabel("Success Rate", fontsize=14)
    plt.ylim(0, 1.05)
    
    # 添加1.0参考线
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    # 添加数值标签
    for bar, (_, row) in zip(bars, success_rate.iterrows()):
        plt.text(bar.get_x() + bar.get_width()/2, row['Success_Rate'] + 0.02,
                f'{row["Success_Rate"]:.3f}', ha='center', va='bottom',
                fontsize=12, weight='bold')
    
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # 保存图表
    filename = f"figure_4_{figure_num}_chinese_success_rate_{title_suffix.lower()}"
    plt.savefig(f'figures/{filename}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'figures/{filename}.png', dpi=300, bbox_inches='tight')
    print(f"✓ 图4.{figure_num}已保存")
    plt.show()
    
    return success_rate

def main():
    print("开始中文实验结果分析...")
    
    # 1. 加载数据
    df = load_chinese_experiment_data()
    if df is None:
        return
    
    # 2. 清理模型名称
    df = clean_model_names(df)
    
    # 确保figures目录存在
    os.makedirs('figures', exist_ok=True)
    
    # 3. 图4.20: 按Hinter模型的成功率
    print("生成图4.20: Hinter模型成功率对比...")
    hinter_success = plot_success_rate_bar(
        df, 
        'hinter_model_clean', 
        'Hinter', 
        '20'
    )
    
    print("\nHinter模型成功率排名:")
    for i, (_, row) in enumerate(hinter_success.iterrows(), 1):
        print(f"  {i}. {row['Model']}: {row['Success_Rate']:.3f}")
    
    # 4. 图4.21: 按Guesser模型的成功率
    print("\n生成图4.21: Guesser模型成功率对比...")
    guesser_success = plot_success_rate_bar(
        df, 
        'guesser_model_clean', 
        'Guesser', 
        '21'
    )
    
    print("\nGuesser模型成功率排名:")
    for i, (_, row) in enumerate(guesser_success.iterrows(), 1):
        print(f"  {i}. {row['Model']}: {row['Success_Rate']:.3f}")
    
    # 5. 综合分析
    print("\n" + "="*60)
    print("中文实验结果分析总结")
    print("="*60)
    
    overall_success = df['success'].mean()
    print(f"\n整体成功率: {overall_success:.3f}")
    
    # 最佳模型对比
    best_hinter = hinter_success.iloc[0]
    best_guesser = guesser_success.iloc[0]
    
    print(f"\n最佳表现:")
    print(f"  • 最佳Hinter: {best_hinter['Model']} ({best_hinter['Success_Rate']:.3f})")
    print(f"  • 最佳Guesser: {best_guesser['Model']} ({best_guesser['Success_Rate']:.3f})")
    
    # 计算模型间差异
    hinter_range = hinter_success['Success_Rate'].max() - hinter_success['Success_Rate'].min()
    guesser_range = guesser_success['Success_Rate'].max() - guesser_success['Success_Rate'].min()
    
    print(f"\n模型间差异:")
    print(f"  • Hinter模型成功率差异: {hinter_range:.3f}")
    print(f"  • Guesser模型成功率差异: {guesser_range:.3f}")
    
    if hinter_range > guesser_range:
        print(f"  • 结论: Hinter角色的模型差异更显著")
    else:
        print(f"  • 结论: Guesser角色的模型差异更显著")
    
    # 统计样本信息
    total_games = len(df)
    unique_targets = df['target_word'].nunique()
    
    print(f"\n实验规模:")
    print(f"  • 总游戏数: {total_games:,}")
    print(f"  • 目标词数: {unique_targets}")
    print(f"  • 平均每词游戏数: {total_games/unique_targets:.1f}")
    
    print(f"\n✅ 中文实验分析完成！图表已保存到 figures/ 目录")

if __name__ == "__main__":
    main()
