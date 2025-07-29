#!/usr/bin/env python3
"""
词频对成功率的影响分析 - 生成图4.9和图4.10
Word Frequency Analysis for Taboo Benchmark - Figures 4.9 & 4.10
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordfreq import zipf_frequency
from scipy import stats as scipy_stats
import os

# 设置绘图风格
sns.set_theme(style="white", font_scale=1.2)
colors = sns.color_palette("magma", 6)

def format_scientific(num):
    """将数字格式化为科学计数法，使用×而不是e"""
    if num == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(num))))
    mantissa = num / (10 ** exp)
    if exp == 0:
        return f"{mantissa:.1f}"
    return f"{mantissa:.1f}×10$^{{{exp}}}$"

def main():
    print("开始词频对成功率的影响分析...")
    
    # 1. 读取实验结果
    csv_path = "results/taboo_experiment_20250712_004918/complete_experiment_results.csv"
    df = pd.read_csv(csv_path)
    print(f"已加载 {len(df):,} 行数据")
    
    # 2. 计算词频
    print("计算词频...")
    df["word_frequency"] = df["target_word"].str.lower().apply(
        lambda w: 10 ** (zipf_frequency(w, "en") - 9)
    )
    # 处理极罕见词汇
    df["word_frequency"].replace(0, 1e-9, inplace=True)
    df["frequency_log"] = np.log10(df["word_frequency"])
    
    # 3. 词频分类
    bins = [-np.inf, -7, -6, np.inf]
    labels = ["Low (<10$^{-7}$)", "Medium (10$^{-7}$–10$^{-6}$)", "High (>10$^{-6}$)"]
    df["frequency_category"] = pd.cut(df["frequency_log"], bins=bins, labels=labels)
    
    # 4. 模型名称清理
    label_map = {
        "openai/gpt-4o": "GPT-4o",
        "google/gemini-2.5-pro": "Gemini-2.5-Pro",
        "deepseek/deepseek-chat-v3-0324": "DeepSeek-V3",
        "anthropic/claude-sonnet-4": "Claude-Sonnet-4",
    }
    df["hinter_model_clean"] = df["hinter_model"].map(label_map).fillna(df["hinter_model"])
    
    # 5. 统计分析
    print("进行统计分析...")
    
    # 整体词频类别影响
    overall_frequency = df.groupby('frequency_category').agg(
        TotalGames=('success', 'count'),
        SuccessRate=('success', 'mean'),
        AverageTurns=('turns_used', 'mean')
    ).round(3)
    
    # 按词汇计算统计（用于散点图）
    word_stats = df.groupby('target_word').agg(
        word_frequency=('word_frequency', 'first'),
        success=('success', 'mean'),
        frequency_category=('frequency_category', 'first')
    ).reset_index()
    
    # 相关性分析
    slope, intercept, r_value, p_value, _ = scipy_stats.linregress(
        np.log10(word_stats['word_frequency']), word_stats['success']
    )
    
    print(f"词频与成功率相关系数: r = {r_value:.3f}, p = {p_value:.4g}")
    
    # 6. 创建图表
    print("生成图表...")
    
    # 确保figures目录存在
    os.makedirs('figures', exist_ok=True)
    
    # 图4.9: 词频类别成功率柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(overall_frequency)), 
                   overall_frequency['SuccessRate'], 
                   color=colors[1], alpha=0.8, edgecolor='black', linewidth=1)
    
    plt.xlabel('Word Frequency Category', fontsize=14)
    plt.ylabel('Success Rate', fontsize=14)
    plt.ylim(0, 1.2)
    
    # 添加1.0参考线
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    # 设置x轴标签
    plt.xticks(range(len(overall_frequency)), overall_frequency.index, fontsize=12)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, overall_frequency['SuccessRate'])):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', 
                fontsize=12, weight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # 保存图4.9
    plt.savefig('figures/figure_4_9_word_frequency_success_rate.pdf', 
                dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure_4_9_word_frequency_success_rate.png', 
                dpi=300, bbox_inches='tight')
    print("✓ 图4.9已保存")
    plt.show()
    
    # 图4.10: 词频与成功率散点图+趋势线
    plt.figure(figsize=(12, 8))
    
    # 按类别绘制散点
    color_map = {cat: colors[i] for i, cat in enumerate(word_stats['frequency_category'].unique())}
    for cat, grp in word_stats.groupby('frequency_category'):
        if pd.notna(cat):  # 确保类别不是NaN
            plt.scatter(grp['word_frequency'], grp['success'],
                       color=color_map[cat], alpha=0.7, s=50, 
                       label=cat, edgecolors='black', linewidth=0.5)
    
    # 添加趋势线
    xmin, xmax = word_stats['word_frequency'].agg(['min', 'max'])
    line_x = np.logspace(np.log10(xmin), np.log10(xmax), 100)
    line_y = slope * np.log10(line_x) + intercept
    plt.plot(line_x, line_y, color=colors[5], linewidth=2, alpha=0.8,
             label=f'Trend Line (r={r_value:.3f})')
    
    plt.xscale('log')
    plt.xlabel('Word Frequency (log scale)', fontsize=14)
    plt.ylabel('Success Rate', fontsize=14)
    plt.ylim(0, 1.2)
    
    # 添加1.0参考线
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    # 自定义x轴标签为科学计数法
    ax = plt.gca()
    x_ticks = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    x_labels = [format_scientific(x) for x in x_ticks]
    plt.xticks(x_ticks, x_labels, fontsize=11)
    
    plt.legend(title='Frequency Category', bbox_to_anchor=(1.05, 1), 
               loc='upper left', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # 保存图4.10
    plt.savefig('figures/figure_4_10_word_frequency_scatter_trend.pdf', 
                dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure_4_10_word_frequency_scatter_trend.png', 
                dpi=300, bbox_inches='tight')
    print("✓ 图4.10已保存")
    plt.show()
    
    # 7. 打印分析结果
    print("\n" + "="*60)
    print("词频分析结果总结")
    print("="*60)
    
    print("\n按词频类别的成功率:")
    for cat, row in overall_frequency.iterrows():
        print(f"  • {cat}: {row['SuccessRate']:.1%} 成功率 "
              f"({int(row['TotalGames'])} 局, 平均 {row['AverageTurns']:.1f} 轮)")
    
    print(f"\n相关性分析:")
    print(f"  • 词频对数与成功率相关系数: r = {r_value:.3f}")
    print(f"  • 显著性检验: p = {p_value:.4g}")
    
    if p_value < 0.05:
        if r_value > 0:
            trend = "高频词（常见词）显著提高成功率"
        else:
            trend = "低频词（罕见词）显著提高成功率"
    else:
        trend = "词频与成功率无显著关系"
    
    print(f"  • 结论: {trend}")
    
    # 极端词频示例
    print(f"\n高频词示例 (前5个):")
    top_freq = word_stats.nlargest(5, 'word_frequency')
    for _, row in top_freq.iterrows():
        freq_str = format_scientific(row['word_frequency'])
        print(f"  • {row['target_word']}: 频率 {freq_str}, 成功率 {row['success']:.1%}")
    
    print(f"\n低频词示例 (前5个):")
    bottom_freq = word_stats.nsmallest(5, 'word_frequency')
    for _, row in bottom_freq.iterrows():
        freq_str = format_scientific(row['word_frequency'])
        print(f"  • {row['target_word']}: 频率 {freq_str}, 成功率 {row['success']:.1%}")
    
    print(f"\n✅ 分析完成！图表已保存到 figures/ 目录")

if __name__ == "__main__":
    main()
