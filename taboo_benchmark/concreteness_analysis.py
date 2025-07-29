#!/usr/bin/env python3
"""
抽象性(Concreteness)对成功率的影响分析 - 生成图4.11和图4.12
Concreteness Analysis for Taboo Benchmark - Figures 4.11 & 4.12
"""

import json
import pathlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import linregress
import os

# 设置绘图风格
sns.set_theme(style="white", font_scale=1.2)
colors = sns.color_palette("magma", 6)

def main():
    print("开始抽象性对成功率的影响分析...")
    
    # 1. 读取实验结果
    csv_path = "results/taboo_experiment_20250712_004918/complete_experiment_results.csv"
    df = pd.read_csv(csv_path)
    print(f"已加载 {len(df):,} 行数据")
    
    # 2. 模型名称清理
    label_map = {
        "openai/gpt-4o": "GPT-4o",
        "google/gemini-2.5-pro": "Gemini-2.5-Pro",
        "deepseek/deepseek-chat-v3-0324": "DeepSeek-V3",
        "anthropic/claude-sonnet-4": "Claude-Sonnet-4",
    }
    df["hinter_model_clean"] = df["hinter_model"].map(label_map).fillna(df["hinter_model"])
    
    # 3. 合并抽象性评分数据
    print("加载抽象性评分数据...")
    if "concreteness_score" not in df.columns:
        json_path = pathlib.Path("taboo_benchmark/data/dataset.json")
        with open(json_path, "r", encoding="utf-8") as f:
            js = json.load(f)
        json_records = [
            {"target_word": item["target"].lower(),
             "concreteness_score": item["metadata"].get("concreteness_score")}
            for item in js
        ]
        concrete_map = pd.DataFrame(json_records)
        df = df.merge(concrete_map, on="target_word", how="left")
        print(f"已合并 concreteness_score，当前缺失值 "
              f"{df['concreteness_score'].isna().sum()} / {len(df)}")
    
    # 4. 按单词维度聚合
    concrete_df = df[df['concreteness_score'].notna()].copy()
    word_level_df = concrete_df.groupby('target_word').agg(
        concreteness_score=('concreteness_score', 'first'),  # 每个单词的具体性评分
        success_rate=('success', 'mean'),                    # 每个单词的平均成功率
        sample_count=('success', 'count')                    # 每个单词的实验次数
    ).reset_index()
    
    print(f"有具体性评分的单词总数: {len(word_level_df)}")
    print(f"具体性评分范围: {word_level_df['concreteness_score'].min():.2f} - {word_level_df['concreteness_score'].max():.2f}")
    
    # 5. 相关性分析
    slope, intercept, r_val, p_val, _ = linregress(
        word_level_df['concreteness_score'], word_level_df['success_rate'])
    print(f"抽象性与成功率相关系数: r = {r_val:.3f}, p = {p_val:.4g}")
    
    # 确保figures目录存在
    os.makedirs('figures', exist_ok=True)
    
    # 6. 图4.11: 抽象性分箱柱状图
    print("生成图4.11: 抽象性柱状图...")
    
    # 将抽象性分成几个区间
    word_level_df['concreteness_bin'] = pd.cut(
        word_level_df['concreteness_score'],
        bins=[1, 2, 3, 4, 5],
        labels=['1-2', '2-3', '3-4', '4-5']
    )
    
    # 按区间统计
    bin_stats = word_level_df.groupby('concreteness_bin').agg(
        avg_success=('success_rate', 'mean'),
        word_count=('target_word', 'count'),
        std_dev=('success_rate', 'std')
    ).reset_index()
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        bin_stats['concreteness_bin'], 
        bin_stats['avg_success'],
        yerr=bin_stats['std_dev'],
        capsize=5,
        color=colors[1], 
        alpha=0.8,
        edgecolor='black',
        linewidth=1,
        width=0.7
    )
    
    plt.xlabel("Concreteness Score Range", fontsize=14)
    plt.ylabel("Average Success Rate", fontsize=14)
    plt.ylim(0, 1.2)
    
    # 添加1.0参考线
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    # 添加单词数量标签
    for i, (bar, row) in enumerate(zip(bars, bin_stats.itertuples())):
        plt.text(bar.get_x() + bar.get_width()/2, 0.05, f"n={row.word_count}", 
                ha='center', va='bottom', color='white', fontsize=12, fontweight='bold')
        # 添加成功率数值标签
        plt.text(bar.get_x() + bar.get_width()/2, row.avg_success + 0.02, 
                f'{row.avg_success:.3f}', ha='center', va='bottom', 
                fontsize=12, weight='bold')
    
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # 保存图4.11
    plt.savefig('figures/figure_4_11_concreteness_bar_chart.pdf', 
                dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure_4_11_concreteness_bar_chart.png', 
                dpi=300, bbox_inches='tight')
    print("✓ 图4.11已保存")
    plt.show()
    
    # 7. 图4.12: 抽象性散点图 + 趋势线
    print("生成图4.12: 抽象性散点图...")
    
    plt.figure(figsize=(12, 8))
    
    # 散点大小与样本量成正比
    scatter = plt.scatter(word_level_df['concreteness_score'], 
                         word_level_df['success_rate'],
                         s=word_level_df['sample_count'] * 5,
                         alpha=0.7,
                         color=colors[2],
                         edgecolors='black',
                         linewidth=0.5)
    
    # 添加LOWESS曲线
    try:
        lowess = sm.nonparametric.lowess
        z = lowess(word_level_df['success_rate'], 
                   word_level_df['concreteness_score'], 
                   frac=0.6)
        plt.plot(z[:, 0], z[:, 1], color=colors[5], linewidth=3, 
                label='LOWESS Curve')
    except:
        print("LOWESS曲线计算失败，跳过")
    
    # 添加线性趋势线
    x_line = np.linspace(word_level_df['concreteness_score'].min(),
                         word_level_df['concreteness_score'].max(), 100)
    plt.plot(x_line, slope * x_line + intercept, 'k--', linewidth=2,
             label=f'Linear Trend (r={r_val:.3f})')
    
    plt.xlabel("Concreteness Score (1 = Abstract, 5 = Concrete)", fontsize=14)
    plt.ylabel("Success Rate", fontsize=14)
    plt.ylim(0, 1.2)
    
    # 添加1.0参考线
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # 保存图4.12
    plt.savefig('figures/figure_4_12_concreteness_scatter_trend.pdf', 
                dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure_4_12_concreteness_scatter_trend.png', 
                dpi=300, bbox_inches='tight')
    print("✓ 图4.12已保存")
    plt.show()
    
    # 8. 创建摘要表格
    print("生成摘要统计...")
    
    # 计算每个区间的详细统计数据
    summary_table = word_level_df.groupby('concreteness_bin').agg(
        word_count=('target_word', 'count'),
        mean_success=('success_rate', 'mean'),
        median_success=('success_rate', 'median'),
        std_dev=('success_rate', 'std'),
        min_success=('success_rate', 'min'),
        max_success=('success_rate', 'max')
    ).reset_index()
    
    # 添加总体统计
    overall_stats = pd.DataFrame({
        'concreteness_bin': ['Overall'],
        'word_count': [len(word_level_df)],
        'mean_success': [word_level_df['success_rate'].mean()],
        'median_success': [word_level_df['success_rate'].median()],
        'std_dev': [word_level_df['success_rate'].std()],
        'min_success': [word_level_df['success_rate'].min()],
        'max_success': [word_level_df['success_rate'].max()]
    })
    
    summary_table = pd.concat([summary_table, overall_stats], ignore_index=True)
    
    # 格式化表格
    summary_table = summary_table.round(3)
    summary_table.columns = ['Concreteness Level', 'Word Count', 'Mean Success', 
                             'Median Success', 'Std Dev', 'Min Success', 'Max Success']
    
    # 保存表格
    summary_table.to_csv('figures/concreteness_summary_table.csv', index=False)
    
    # 9. 打印分析结果
    print("\n" + "="*60)
    print("抽象性分析结果总结")
    print("="*60)
    
    print("\n按抽象性区间的成功率:")
    for _, row in bin_stats.iterrows():
        print(f"  • {row['concreteness_bin']}: {row['avg_success']:.1%} 成功率 "
              f"({int(row['word_count'])} 个词汇, 标准差 {row['std_dev']:.3f})")
    
    print(f"\n相关性分析:")
    print(f"  • 抽象性与成功率相关系数: r = {r_val:.3f}")
    print(f"  • 显著性检验: p = {p_val:.4g}")
    
    if p_val < 0.05:
        if r_val > 0:
            trend = "具体词汇显著提高成功率"
        else:
            trend = "抽象词汇显著提高成功率"
    else:
        trend = "抽象性与成功率无显著关系"
    
    print(f"  • 结论: {trend}")
    
    # 极端抽象性示例
    print(f"\n最抽象词汇示例 (前5个):")
    most_abstract = word_level_df.nsmallest(5, 'concreteness_score')
    for _, row in most_abstract.iterrows():
        print(f"  • {row['target_word']}: 抽象性 {row['concreteness_score']:.2f}, 成功率 {row['success_rate']:.1%}")
    
    print(f"\n最具体词汇示例 (前5个):")
    most_concrete = word_level_df.nlargest(5, 'concreteness_score')
    for _, row in most_concrete.iterrows():
        print(f"  • {row['target_word']}: 抽象性 {row['concreteness_score']:.2f}, 成功率 {row['success_rate']:.1%}")
    
    print(f"\n✅ 分析完成！图表已保存到 figures/ 目录")
    print("\n摘要统计表:")
    print(summary_table)

if __name__ == "__main__":
    main()
