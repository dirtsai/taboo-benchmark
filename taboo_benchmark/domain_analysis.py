#!/usr/bin/env python3
"""
Domain效应分析 - 按category和model的成功率分析
Domain Effect Analysis for Taboo Benchmark
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import wordfreq
import os
import pathlib

# 设置绘图风格
sns.set_theme(style="white", font_scale=1.2)
colors = sns.color_palette("magma", 6)

def get_word_frequency(word):
    """使用 wordfreq 库获取词频（以每百万词中出现的次数表示）"""
    return wordfreq.zipf_frequency(word, 'en')

def main():
    print("开始Domain效应分析...")
    
    # 1. 读取实验结果
    csv_path = "results/taboo_experiment_20250712_004918/complete_experiment_results.csv"
    complete_experiment_results = pd.read_csv(csv_path)
    print(f"已加载 {len(complete_experiment_results):,} 行数据")
    
    # 2. 提取唯一的 category 值
    categories = complete_experiment_results['category'].unique()
    print(f"数据集中的唯一 category 值（共 {len(categories)} 个）：")
    print(categories)
    
    # 3. 计算总体成功率（按 category）
    overall_category_success = (
        complete_experiment_results
        .groupby('category')['success']
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    overall_category_success.columns = ['Category', 'Success_Rate']
    
    print("\n各 Category 的总体平均成功率（降序）：")
    for _, row in overall_category_success.iterrows():
        print(f"  • {row['Category']}: {row['Success_Rate']:.1%}")
    
    # 4. 计算每个 category 的平均词频
    print("计算各category的平均词频...")
    category_word_freqs = []
    for category in categories:
        # 获取该 category 的所有目标词
        target_words = complete_experiment_results[
            complete_experiment_results['category'] == category
        ]['target_word'].unique()
        
        # 计算这些词的平均词频
        if len(target_words) > 0:
            avg_freq = sum(get_word_frequency(word) for word in target_words) / len(target_words)
        else:
            avg_freq = 0
        
        category_word_freqs.append({
            'Category': category,
            'Average_Word_Frequency': avg_freq
        })
    
    # 转换为 DataFrame
    category_word_freqs_df = pd.DataFrame(category_word_freqs)
    
    # 合并成功率和词频数据
    category_stats = pd.merge(overall_category_success, category_word_freqs_df, on='Category')
    
    # 确保figures目录存在
    out_dir = pathlib.Path("figures")
    out_dir.mkdir(exist_ok=True)
    
    # 5. 图1: 总体成功率（按 category）并叠加词频
    print("生成图1: Category成功率和词频分析...")
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # 绘制条形图
    bars = ax1.bar(
        category_stats['Category'], 
        category_stats['Success_Rate'],
        color=colors[1],
        alpha=0.8,
        width=0.6
    )
    
    ax1.set_ylim(0.8, 1.0)  # 80% to 100%
    ax1.set_xlabel("Category", fontsize=14)
    ax1.set_ylabel("Success Rate", fontsize=14)  # 去掉括号
    
    # 添加百分比标签
    for i, (bar, row) in enumerate(zip(bars, category_stats.itertuples())):
        ax1.text(
            bar.get_x() + bar.get_width()/2, 
            bar.get_height() + 0.01, 
            f"{row.Success_Rate:.1%}", 
            ha='center', 
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )
    
    # 创建第二个 y 轴
    ax2 = ax1.twinx()
    ax2.set_ylim(0, 4.5) 
    
    # 绘制词频折线图
    line = ax2.plot(
        category_stats['Category'], 
        category_stats['Average_Word_Frequency'],
        marker='o',
        color=colors[4],
        linewidth=3,
        markersize=8,
        label='Average Word Frequency'
    )
    
    ax2.set_ylabel("Average Word Frequency (Zipf)", fontsize=14, color=colors[4])
    
    # 设置x轴标签为水平（摆正）
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, ha='center', fontsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12, colors=colors[4])
    
    # 添加网格
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 添加1.0参考线
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    plt.tight_layout()
    
    # 保存图1
    fname = "figure_4_15_category_success_rate_word_frequency"
    plt.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(out_dir / f"{fname}.png", bbox_inches="tight", dpi=300)
    print("✓ 图4.15已保存")
    plt.show()
    
    # 6. 按 category 和 hinter_model 分组计算成功率
    print("生成图2: 各模型在不同category的表现...")
    
    category_hinter_success = (
        complete_experiment_results
        .groupby(['category', 'hinter_model'])['success']
        .mean()
        .reset_index()
    )
    
    # 使用标签映射
    label_map = {
        "openai/gpt-4o": "GPT-4o",
        "google/gemini-2.5-pro": "Gemini-2.5-Pro",
        "deepseek/deepseek-chat-v3-0324": "DeepSeek-V3",
        "anthropic/claude-sonnet-4": "Claude-Sonnet-4",
    }
    
    # 应用标签映射
    category_hinter_success['hinter_model_clean'] = category_hinter_success['hinter_model'].map(label_map)
    
    # 7. 创建分组条形图，按模型比较各个 category
    plt.figure(figsize=(16, 8))
    
    # 绘制分组条形图
    ax = sns.barplot(
        x='category', 
        y='success',
        hue='hinter_model_clean',
        data=category_hinter_success,
        palette="magma"
    )
    
    ax.set_xlabel("Category", fontsize=14)
    ax.set_ylabel("Success Rate", fontsize=14)  # 去掉括号
    ax.set_ylim(0, 1.05)
    
    # 添加1.0参考线
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    # 设置x轴标签为水平（摆正）
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # 将图例放在框外面
    ax.legend(title="Hinter Model", fontsize=11, title_fontsize=12, 
              bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # 保存图2
    fname = "figure_4_16_hinter_model_by_category"
    plt.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(out_dir / f"{fname}.png", bbox_inches="tight", dpi=300)
    print("✓ 图4.16已保存")
    plt.show()
    
    # 8. 打印分析结果
    print("\n" + "="*60)
    print("Domain效应分析结果总结")
    print("="*60)
    
    print("\n各Category的成功率排名:")
    for i, (_, row) in enumerate(overall_category_success.iterrows(), 1):
        freq_info = category_word_freqs_df[category_word_freqs_df['Category'] == row['Category']]
        avg_freq = freq_info['Average_Word_Frequency'].iloc[0] if len(freq_info) > 0 else 0
        print(f"  {i}. {row['Category']}: {row['Success_Rate']:.1%} 成功率 "
              f"(平均词频: {avg_freq:.2f})")
    
    # 计算category间的成功率差异
    max_success = overall_category_success['Success_Rate'].max()
    min_success = overall_category_success['Success_Rate'].min()
    success_range = max_success - min_success
    
    print(f"\nCategory效应分析:")
    print(f"  • 最高成功率: {max_success:.1%}")
    print(f"  • 最低成功率: {min_success:.1%}")
    print(f"  • 成功率差异范围: {success_range:.1%}")
    
    if success_range > 0.05:  # 5%以上差异
        print(f"  • 结论: Category对成功率有显著影响")
    else:
        print(f"  • 结论: Category对成功率影响较小")
    
    # 分析词频与成功率的关系
    freq_success_corr = category_stats[['Average_Word_Frequency', 'Success_Rate']].corr().iloc[0, 1]
    print(f"\n词频与成功率的相关性:")
    print(f"  • 相关系数: r = {freq_success_corr:.3f}")
    
    if abs(freq_success_corr) > 0.3:
        if freq_success_corr > 0:
            print(f"  • 结论: 词频越高的category，成功率越高")
        else:
            print(f"  • 结论: 词频越高的category，成功率越低")
    else:
        print(f"  • 结论: 词频与category成功率无明显关系")
    
    print(f"\n✅ 分析完成！图表已保存到 figures/ 目录")

if __name__ == "__main__":
    main()
