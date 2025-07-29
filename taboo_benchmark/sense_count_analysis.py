#!/usr/bin/env python3
"""
词义数量(Sense Count)对成功率的影响分析 - 生成图4.13和图4.14
Sense Count Analysis for Taboo Benchmark - Figures 4.13 & 4.14
"""

import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordfreq import zipf_frequency
from scipy import stats
import os

# 设置绘图风格
sns.set_theme(style="white", font_scale=1.2)
colors = sns.color_palette("magma", 6)

def main():
    print("开始词义数量对成功率的影响分析...")
    
    # 1. 读取实验结果
    csv_path = "results/taboo_experiment_20250712_004918/complete_experiment_results.csv"
    df = pd.read_csv(csv_path)
    print(f"已加载 {len(df):,} 行数据")
    
    # 2. 加载数据集和提取sense count
    print("加载词义数量数据...")
    with open('data/dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    dataset_df = pd.DataFrame(dataset)
    dataset_df['metadata'] = dataset_df['metadata'].apply(lambda x: x if isinstance(x, dict) else {})
    dataset_df['sense_count'] = dataset_df['metadata'].apply(lambda x: x.get('sense_count', 1))
    
    print(f"Dataset加载完成，共{len(dataset_df)}个词条")
    
    # 3. 数据清理和过滤
    print("进行数据清理...")
    
    # 确定目标词的列名
    target_column = 'target_word' if 'target_word' in df.columns else 'target'
    
    # 过滤掉有格式错误或taboo违反的目标词
    if 'failure_reason' in df.columns:
        problematic_targets = df[
            (df['success'] == False) & 
            (df['failure_reason'].isin(['format_violation', 'taboo_violation']))
        ][target_column].unique()
        
        df = df[~df[target_column].isin(problematic_targets)]
        print(f"移除了 {len(problematic_targets)} 个有问题的目标词")
        
        # 只保留成功样本和max_turns失败样本
        df = df[
            (df['success'] == True) | 
            ((df['success'] == False) & (df['failure_reason'] == 'MAX_TURNS_EXCEEDED') &
             (~df[target_column].isin(['humor', 'backward', 'organization'])))
        ]
    
    # 4. 模型名称清理
    label_map = {
        "openai/gpt-4o": "GPT-4o",
        "google/gemini-2.5-pro": "Gemini-2.5-Pro",
        "deepseek/deepseek-chat-v3-0324": "DeepSeek-V3",
        "anthropic/claude-sonnet-4": "Claude-Sonnet-4",
    }
    df["hinter_model_clean"] = df["hinter_model"].map(label_map).fillna(df["hinter_model"])
    
    # 5. 合并sense count数据
    df['sense_count'] = df[target_column].map(
        dataset_df.set_index('target')['sense_count']
    )
    
    print(f"成功匹配sense count信息的比例: {df['sense_count'].notna().mean():.3f}")
    
    # 6. 过滤样本量不足的sense count区间
    MIN_SAMPLE_SIZE = 80
    sense_count_samples = df['sense_count'].value_counts()
    valid_sense_counts = sense_count_samples[sense_count_samples >= MIN_SAMPLE_SIZE].index.tolist()
    
    print(f"过滤前sense count区间数量: {df['sense_count'].nunique()}")
    print(f"过滤后sense count区间数量: {len(valid_sense_counts)}")
    
    # 创建过滤后的DataFrame
    filtered_df = df[df['sense_count'].isin(valid_sense_counts)]
    
    # 7. 统计分析
    print("进行统计分析...")
    
    # 整体sense count对成功率的影响
    overall_sense = filtered_df.groupby('sense_count').agg({
        'success': ['count', 'mean'],
        'turns_used': 'mean'
    }).round(3)
    overall_sense.columns = ['Total_Games', 'Success_Rate', 'Average_Turns']
    overall_sense = overall_sense.reset_index()
    
    # 按模型和sense count的分析
    sense_success = filtered_df.groupby(['sense_count', 'hinter_model_clean']).agg({
        'success': ['count', 'mean'],
        'turns_used': 'mean'
    }).round(3)
    sense_success.columns = ['Games', 'Success_Rate', 'Average_Turns']
    sense_success = sense_success.reset_index()
    
    # 相关性分析
    correlation = filtered_df[['sense_count', 'success']].corr().iloc[0, 1]
    print(f"Sense Count与成功率的相关性: {correlation:.4f}")
    
    # 确保figures目录存在
    os.makedirs('figures', exist_ok=True)
    
    # 8. 图4.13: Sense count成功率柱状图（对应原图2）
    print("生成图4.13: Sense count柱状图...")
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        overall_sense['sense_count'], 
        overall_sense['Success_Rate'],
        color=colors[1], 
        alpha=0.8,
        width=0.7
    )
    
    plt.xlabel("Number of Senses", fontsize=14)
    plt.ylabel("Success Rate", fontsize=14)
    plt.ylim(0, 1.05)
    
    # 添加1.0参考线
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    # 添加数值标签
    for i, (bar, row) in enumerate(zip(bars, overall_sense.itertuples())):
        plt.text(bar.get_x() + bar.get_width()/2, row.Success_Rate + 0.02, 
                f'{row.Success_Rate:.3f}', ha='center', va='bottom', 
                fontsize=12, weight='bold')
        # 添加样本数量标签
        plt.text(bar.get_x() + bar.get_width()/2, 0.05, f"n={row.Total_Games}", 
                ha='center', va='bottom', color='white', fontsize=12, fontweight='bold')
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # 保存图4.13
    plt.savefig('figures/figure_4_13_sense_count_bar_chart.pdf', 
                dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure_4_13_sense_count_bar_chart.png', 
                dpi=300, bbox_inches='tight')
    print("✓ 图4.13已保存")
    plt.show()
    
    # 9. 图4.14: Sense count散点图（对应原图4）
    print("生成图4.14: Sense count散点图...")
    
    plt.figure(figsize=(12, 8))
    
    # 按模型绘制散点
    for i, model in enumerate(filtered_df['hinter_model_clean'].unique()):
        model_data = filtered_df[filtered_df['hinter_model_clean'] == model]
        success_by_sense = model_data.groupby('sense_count')['success'].mean()
        # 点的大小与样本量成正比
        sizes = model_data.groupby('sense_count').size() * 8
        plt.scatter(success_by_sense.index, success_by_sense.values, 
                   label=model, color=colors[i], alpha=0.7, s=sizes,
                   edgecolors='black', linewidth=0.5)
    
    plt.xlabel("Number of Senses", fontsize=14)
    plt.ylabel("Success Rate", fontsize=14)
    plt.ylim(0, 1.05)
    
    # 添加1.0参考线
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(title='Model', fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=0.5)
    plt.tight_layout()
    
    # 保存图4.14
    plt.savefig('figures/figure_4_14_sense_count_scatter.pdf', 
                dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure_4_14_sense_count_scatter.png', 
                dpi=300, bbox_inches='tight')
    print("✓ 图4.14已保存")
    plt.show()
    
    # 10. 打印分析结果
    print("\n" + "="*60)
    print("词义数量分析结果总结")
    print("="*60)
    
    print("\n按词义数量的成功率:")
    for _, row in overall_sense.iterrows():
        print(f"  • {int(row['sense_count'])} 个词义: {row['Success_Rate']:.1%} 成功率 "
              f"({int(row['Total_Games'])} 局游戏, 平均 {row['Average_Turns']:.1f} 轮)")
    
    print(f"\n相关性分析:")
    print(f"  • 词义数量与成功率相关系数: r = {correlation:.3f}")
    
    if abs(correlation) > 0.1:
        if correlation > 0:
            trend = "词义数量越多，成功率越高"
        else:
            trend = "词义数量越多，成功率越低"
    else:
        trend = "词义数量与成功率无明显关系"
    
    print(f"  • 结论: {trend}")
    
    # 极端词义数量示例
    print(f"\n词义数量最少的词汇:")
    min_sense = filtered_df.groupby(target_column).agg(
        sense_count=('sense_count', 'first'),
        success_rate=('success', 'mean')
    ).nsmallest(5, 'sense_count')
    for word, row in min_sense.iterrows():
        print(f"  • {word}: {int(row['sense_count'])} 个词义, 成功率 {row['success_rate']:.1%}")
    
    print(f"\n词义数量最多的词汇:")
    max_sense = filtered_df.groupby(target_column).agg(
        sense_count=('sense_count', 'first'),
        success_rate=('success', 'mean')
    ).nlargest(5, 'sense_count')
    for word, row in max_sense.iterrows():
        print(f"  • {word}: {int(row['sense_count'])} 个词义, 成功率 {row['success_rate']:.1%}")
    
    print(f"\n✅ 分析完成！图表已保存到 figures/ 目录")

if __name__ == "__main__":
    main()
