#!/usr/bin/env python3
"""
中文实验字符数提示对比分析 - 生成图4.25a和图4.25b
Chinese Experiment Character Count Comparison Analysis - Figures 4.25a & 4.25b
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

def load_experiment_data():
    """加载实验数据"""
    try:
        # 无字符数提示的实验结果
        results_path_no_hint = "results/chinese_full_experiment_20250717_222959/chinese_full_results_20250717_222959.csv"
        no_hint_results = pd.read_csv(results_path_no_hint)
        print(f"✅ 成功加载无字符数提示结果: {len(no_hint_results)} 条记录")
        
        # 有字符数提示的实验结果
        results_path_with_hint = "taboo_benchmark/results/chinese_merged_results_fixed_20250723_013406.csv"
        with_hint_results = pd.read_csv(results_path_with_hint)
        print(f"✅ 成功加载有字符数提示结果: {len(with_hint_results)} 条记录")
        
        return no_hint_results, with_hint_results
    except Exception as e:
        print(f"❌ 加载实验数据失败: {e}")
        return None, None

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

def plot_success_rate_comparison(df_no_hint, df_with_hint, group_key, title_suffix, figure_num):
    """绘制字符数提示对比柱状图"""
    
    # 计算成功率
    rate_no_hint = (
        df_no_hint.groupby(group_key)['success']
        .mean()
        .sort_values(ascending=False)
    )
    
    rate_with_hint = (
        df_with_hint.groupby(group_key)['success']
        .mean()
        .reindex(rate_no_hint.index)
    )
    
    # 准备并排柱状图数据
    models = rate_no_hint.index
    x = np.arange(len(models))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    
    # 创建并排柱状图
    bars1 = plt.bar(x - width/2, rate_no_hint.values, width,
                   label='Without Character Count',
                   color=colors[1], alpha=0.8)
    bars2 = plt.bar(x + width/2, rate_with_hint.values, width,
                   label='With Character Count',
                   color=colors[4], alpha=0.8)
    
    plt.xlabel(f"{title_suffix} Model", fontsize=14)
    plt.ylabel("Success Rate", fontsize=14)
    plt.ylim(0, 1.05)
    
    # 添加1.0参考线
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    # 设置x轴标签
    plt.xticks(x, models, fontsize=12, rotation=0)
    plt.yticks(fontsize=12)
    
    # 添加数值标签
    for i, (val1, val2) in enumerate(zip(rate_no_hint.values, rate_with_hint.values)):
        plt.text(i - width/2, val1 + 0.02, f"{val1:.3f}", ha="center", va="bottom",
                fontsize=11, weight="bold")
        plt.text(i + width/2, val2 + 0.02, f"{val2:.3f}", ha="center", va="bottom",
                fontsize=11, weight="bold")
    
    # 添加图例
    plt.legend(fontsize=11, loc='upper right', frameon=True, fancybox=True, shadow=True)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # 保存图表
    filename = f"figure_4_25{figure_num}_chinese_character_count_{title_suffix.lower()}"
    plt.savefig(f'figures/{filename}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'figures/{filename}.png', dpi=300, bbox_inches='tight')
    print(f"✓ 图4.25{figure_num}已保存")
    plt.show()
    
    return rate_no_hint, rate_with_hint

def main():
    print("开始中文实验字符数提示对比分析...")
    
    # 1. 加载数据
    no_hint_results, with_hint_results = load_experiment_data()
    if no_hint_results is None or with_hint_results is None:
        return
    
    # 2. 清理模型名称
    no_hint_results = clean_model_names(no_hint_results)
    with_hint_results = clean_model_names(with_hint_results)
    
    # 确保figures目录存在
    os.makedirs('figures', exist_ok=True)
    
    # 3. 图4.25a: Hinter模型对比
    print("生成图4.25a: Hinter模型字符数提示对比...")
    rate_hinter_no, rate_hinter_with = plot_success_rate_comparison(
        no_hint_results, with_hint_results,
        'hinter_model_clean',
        'Hinter',
        'a'
    )
    
    print("\nHinter模型成功率对比:")
    print("无字符数提示:")
    for model, rate in rate_hinter_no.items():
        print(f"  {model}: {rate:.3f}")
    print("有字符数提示:")
    for model, rate in rate_hinter_with.items():
        print(f"  {model}: {rate:.3f}")
    
    # 4. 图4.25b: Guesser模型对比
    print("\n生成图4.25b: Guesser模型字符数提示对比...")
    rate_guesser_no, rate_guesser_with = plot_success_rate_comparison(
        no_hint_results, with_hint_results,
        'guesser_model_clean',
        'Guesser',
        'b'
    )
    
    print("\nGuesser模型成功率对比:")
    print("无字符数提示:")
    for model, rate in rate_guesser_no.items():
        print(f"  {model}: {rate:.3f}")
    print("有字符数提示:")
    for model, rate in rate_guesser_with.items():
        print(f"  {model}: {rate:.3f}")
    
    # 5. 综合分析
    print("\n" + "="*60)
    print("字符数提示效果分析总结")
    print("="*60)
    
    # 计算平均提升效果
    hinter_improvements = []
    guesser_improvements = []
    
    print(f"\nHinter角色字符数提示效果:")
    for model in rate_hinter_no.index:
        if model in rate_hinter_with.index:
            improvement = rate_hinter_with[model] - rate_hinter_no[model]
            hinter_improvements.append(improvement)
            print(f"  • {model}: {improvement:+.3f} ({improvement*100:+.1f}%)")
    
    print(f"\nGuesser角色字符数提示效果:")
    for model in rate_guesser_no.index:
        if model in rate_guesser_with.index:
            improvement = rate_guesser_with[model] - rate_guesser_no[model]
            guesser_improvements.append(improvement)
            print(f"  • {model}: {improvement:+.3f} ({improvement*100:+.1f}%)")
    
    # 计算平均效果
    if hinter_improvements:
        avg_hinter_improvement = np.mean(hinter_improvements)
        print(f"\nHinter角色平均提升: {avg_hinter_improvement:+.3f} ({avg_hinter_improvement*100:+.1f}%)")
    
    if guesser_improvements:
        avg_guesser_improvement = np.mean(guesser_improvements)
        print(f"Guesser角色平均提升: {avg_guesser_improvement:+.3f} ({avg_guesser_improvement*100:+.1f}%)")
    
    # 分析哪个角色受益更多
    if hinter_improvements and guesser_improvements:
        if avg_hinter_improvement > avg_guesser_improvement:
            print(f"\n结论: Hinter角色从字符数提示中获得更大收益")
        elif avg_guesser_improvement > avg_hinter_improvement:
            print(f"\n结论: Guesser角色从字符数提示中获得更大收益")
        else:
            print(f"\n结论: 两个角色从字符数提示中获得相似收益")
    
    # 找出受益最大和最小的模型
    all_improvements = list(zip(['Hinter']*len(hinter_improvements), rate_hinter_no.index, hinter_improvements)) + \
                      list(zip(['Guesser']*len(guesser_improvements), rate_guesser_no.index, guesser_improvements))
    
    if all_improvements:
        best_improvement = max(all_improvements, key=lambda x: x[2])
        worst_improvement = min(all_improvements, key=lambda x: x[2])
        
        print(f"\n极端值分析:")
        print(f"  • 最大受益: {best_improvement[0]} {best_improvement[1]} ({best_improvement[2]:+.3f})")
        print(f"  • 最小受益: {worst_improvement[0]} {worst_improvement[1]} ({worst_improvement[2]:+.3f})")
    
    print(f"\n✅ 字符数提示对比分析完成！图表已保存到 figures/ 目录")

if __name__ == "__main__":
    main()
