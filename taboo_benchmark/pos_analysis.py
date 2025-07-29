#!/usr/bin/env python3
"""
词性(POS)分析 - 生成图4.19
Part of Speech Analysis for Taboo Benchmark - Figure 4.19
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import wordfreq
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
sns.set_theme(style="white", font_scale=1.2)
colors = sns.color_palette("magma", 6)

def load_experiment_data():
    """加载实验数据"""
    try:
        # 文件路径
        merged_results_path = "results/quick80_merged_results_complete.csv"
        quick80_dataset_path = "quick80_dataset.json"
        
        # 加载实验结果
        df = pd.read_csv(merged_results_path)
        print(f"✅ 成功加载实验结果: {len(df)} 条记录")
        
        # 加载quick80数据集获取POS信息
        with open(quick80_dataset_path, 'r') as f:
            quick80_data = json.load(f)
        
        # 创建target_word到POS的映射
        target_to_pos = {}
        for entry in quick80_data:
            if 'target' in entry and 'part_of_speech' in entry:
                target_to_pos[entry['target']] = entry['part_of_speech']
        
        # 添加POS信息到数据框
        df['pos'] = df['target_word'].map(target_to_pos)
        
        # 检查缺失POS标签的单词
        missing_pos = df[df['pos'].isna()]['target_word'].unique()
        if len(missing_pos) > 0:
            print(f"⚠️  {len(missing_pos)} 个目标词缺少POS标签")
        
        # 移除缺失POS的记录
        df = df.dropna(subset=['pos'])
        print(f"清理后数据: {len(df)} 条记录")
        
        return df
        
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return None

def get_word_frequency(word):
    """使用wordfreq库获取词频(Zipf量级)"""
    return wordfreq.zipf_frequency(word, 'en')

def main():
    print("开始词性(POS)分析...")
    
    # 1. 加载数据
    df = load_experiment_data()
    if df is None:
        return
    
    # 确保figures目录存在
    os.makedirs('figures', exist_ok=True)
    
    # 2. 分析POS效应
    print("分析POS效应...")
    
    # 获取唯一POS值
    pos_values = df['pos'].dropna().unique()
    print(f"数据集中的POS类型 ({len(pos_values)} 种): {sorted(pos_values)}")
    
    # 3. 计算各POS的整体成功率
    overall_pos_success = (
        df.groupby('pos')['success']
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    overall_pos_success.columns = ['Part_of_Speech', 'Success_Rate']
    
    # 4. 计算各POS的平均词频
    pos_word_freqs = []
    for pos in pos_values:
        # 获取该POS的所有目标词
        target_words = df[df['pos'] == pos]['target_word'].unique()
        
        # 计算平均频率
        if len(target_words) > 0:
            avg_freq = sum(get_word_frequency(word) for word in target_words) / len(target_words)
        else:
            avg_freq = 0
        
        pos_word_freqs.append({
            'Part_of_Speech': pos,
            'Average_Word_Frequency': avg_freq,
            'Word_Count': len(target_words)
        })
    
    # 转换为DataFrame
    pos_word_freqs_df = pd.DataFrame(pos_word_freqs)
    
    # 合并成功率和频率数据
    pos_stats = pd.merge(overall_pos_success, pos_word_freqs_df, on='Part_of_Speech')
    pos_stats = pos_stats.sort_values('Success_Rate', ascending=False)
    
    print("\nPOS成功率和词频统计:")
    print(pos_stats)
    
    # 5. 生成图4.19: POS成功率和词频分析
    print("生成图4.19: POS成功率和词频分析...")
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # 绘制成功率柱状图
    bars = ax1.bar(
        pos_stats['Part_of_Speech'], 
        pos_stats['Success_Rate'],
        color=colors[1], 
        alpha=0.8,
        width=0.6
    )
    
    ax1.set_xlabel("Part of Speech", fontsize=14)
    ax1.set_ylabel("Success Rate", fontsize=14)
    ax1.set_ylim(0, 1.05)
    
    # 添加1.0参考线
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    # 添加成功率数值标签
    for bar, (_, row) in zip(bars, pos_stats.iterrows()):
        ax1.text(bar.get_x() + bar.get_width()/2, row['Success_Rate'] + 0.02, 
                f'{row["Success_Rate"]:.3f}', ha='center', va='bottom', 
                fontsize=11, weight='bold')
        # 添加词汇数量标签
        ax1.text(bar.get_x() + bar.get_width()/2, 0.05, 
                f"n={int(row['Word_Count'])}", ha='center', va='bottom', 
                color='white', fontsize=10, fontweight='bold')
    
    # 创建第二个Y轴用于词频
    ax2 = ax1.twinx()
    ax2.set_ylim(0, max(pos_stats['Average_Word_Frequency']) * 1.2)
    
    # 绘制词频折线图
    line = ax2.plot(
        pos_stats['Part_of_Speech'], 
        pos_stats['Average_Word_Frequency'],
        marker='o', 
        color=colors[4], 
        linewidth=3, 
        markersize=8, 
        label='Average Word Frequency'
    )
    
    ax2.set_ylabel("Average Word Frequency (Zipf)", fontsize=14, color=colors[4])
    ax2.tick_params(axis='y', labelcolor=colors[4])
    
    # 设置坐标轴
    ax1.set_xticks(range(len(pos_stats)))
    ax1.set_xticklabels(pos_stats['Part_of_Speech'], fontsize=12, rotation=0)
    ax1.tick_params(axis='y', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    
    # 添加网格
    ax1.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图4.19
    plt.savefig('figures/figure_4_19_pos_success_rate_frequency.pdf', 
                dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure_4_19_pos_success_rate_frequency.png', 
                dpi=300, bbox_inches='tight')
    print("✓ 图4.19已保存")
    plt.show()
    
    # 6. 打印详细分析结果
    print("\n" + "="*60)
    print("词性(POS)分析结果总结")
    print("="*60)
    
    print(f"\nPOS类型排名 (按成功率降序):")
    for i, (_, row) in enumerate(pos_stats.iterrows(), 1):
        print(f"  {i}. {row['Part_of_Speech']}: 成功率 {row['Success_Rate']:.3f}, "
              f"平均词频 {row['Average_Word_Frequency']:.3f}, "
              f"词汇数 {int(row['Word_Count'])}")
    
    # 计算相关性
    from scipy.stats import pearsonr
    correlation, p_value = pearsonr(pos_stats['Success_Rate'], pos_stats['Average_Word_Frequency'])
    
    print(f"\n相关性分析:")
    print(f"  • 成功率与词频相关系数: {correlation:.3f}")
    print(f"  • 显著性水平: p = {p_value:.3f}")
    
    if p_value < 0.05:
        if correlation > 0:
            print(f"  • 结论: 词频越高的POS类型成功率显著更高")
        else:
            print(f"  • 结论: 词频越高的POS类型成功率显著更低")
    else:
        print(f"  • 结论: 成功率与词频无显著相关性")
    
    # 找出极端值
    highest_success = pos_stats.iloc[0]
    lowest_success = pos_stats.iloc[-1]
    
    print(f"\n极端值分析:")
    print(f"  • 最高成功率: {highest_success['Part_of_Speech']} ({highest_success['Success_Rate']:.3f})")
    print(f"  • 最低成功率: {lowest_success['Part_of_Speech']} ({lowest_success['Success_Rate']:.3f})")
    print(f"  • 成功率差异: {highest_success['Success_Rate'] - lowest_success['Success_Rate']:.3f}")
    
    print(f"\n✅ POS分析完成！图表已保存到 figures/ 目录")

if __name__ == "__main__":
    main()
