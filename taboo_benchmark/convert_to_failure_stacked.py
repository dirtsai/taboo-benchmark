#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将传统的Violation Rate柱状图改为失败原因分解的堆叠图
只显示失败部分，不包含成功率
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置图表样式
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

def create_failure_reasons_stacked_chart():
    """创建失败原因堆叠图，替代单一violation rate图"""
    
    # 加载数据
    results_df = pd.read_csv('results/taboo_experiment_20250712_004918/complete_experiment_results.csv')
    
    # 模型名称映射
    model_name_mapping = {
        'anthropic/claude-sonnet-4': 'Claude Sonnet 4',
        'openai/gpt-4o': 'GPT-4o', 
        'google/gemini-2.5-pro': 'Gemini 2.5 Pro',
        'deepseek/deepseek-chat-v3-0324': 'DeepSeek Chat V3'
    }
    
    results_df['hinter_model_clean'] = results_df['hinter_model'].map(model_name_mapping)
    
    # 只分析失败的游戏
    failed_games = results_df[results_df['success'] == False]
    
    # 计算失败原因分解
    failure_breakdown = failed_games.groupby('hinter_model_clean').apply(
        lambda x: pd.Series({
            'Taboo_Violation': (x['has_taboo_violation'] == True).sum(),
            'Max_Turns_Exceeded': ((x['has_taboo_violation'] == False) & (x['turns_used'] >= 10)).sum(),
            'Other_Failure': ((x['has_taboo_violation'] == False) & (x['turns_used'] < 10)).sum()
        })
    ).astype(int)
    
    # 计算基于总游戏数的失败率
    total_games_per_model = results_df.groupby('hinter_model_clean').size()
    failure_rates = failure_breakdown.div(total_games_per_model, axis=0)
    
    # 创建对比图：传统 vs 堆叠失败原因
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # === 左图：传统方法 ===
    # 只显示violation rate (重现您的图片)
    violation_rates = failure_breakdown['Taboo_Violation'] / total_games_per_model
    
    colors_traditional = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 对应图片中的颜色
    bars1 = ax1.bar(violation_rates.index, violation_rates.values, 
                    color=colors_traditional, edgecolor='white', linewidth=0.8)
    
    ax1.set_title('Traditional: Taboo Violation Rate Only', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Violation Rate', fontsize=12)
    ax1.set_ylim(0, 0.06)
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签（重现原图风格）
    for bar, rate in zip(bars1, violation_rates.values):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002, 
                f'{rate:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax1.tick_params(axis='x', rotation=45)
    
    # === 右图：失败原因堆叠图 ===
    # 显示所有失败原因的分解
    colors_stacked = ['#d62728', '#ff7f0e', '#9467bd']  # 红色违规，橙色超时，紫色其他
    
    failure_rates.plot(kind='bar', stacked=True, ax=ax2, 
                      color=colors_stacked, edgecolor='white', linewidth=0.8)
    
    ax2.set_title('Stacked: Complete Failure Reason Breakdown', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Failure Rate', fontsize=12)
    ax2.legend(['Taboo Violation', 'Max Turns Exceeded', 'Other Failures'], 
              title='Failure Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_ylim(0, 0.25)  # 调整以适应总失败率
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 添加总失败率标签
    total_failure_rates = failure_breakdown.sum(axis=1) / total_games_per_model
    for i, (model, total_rate) in enumerate(total_failure_rates.items()):
        violation_rate = violation_rates[model]
        ax2.text(i, total_rate + 0.01, 
                f'Total: {total_rate:.3f}\nViolation: {violation_rate:.3f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('comprehensive_figures/failure_reasons_stacked_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印对比分析
    print("=" * 60)
    print("📊 Failure Analysis: Traditional vs Stacked Breakdown")
    print("=" * 60)
    
    print("\n🔴 Traditional Approach (Your original chart):")
    print("❌ Only shows taboo violation rates")
    print("❌ Missing other failure types (timeouts, other issues)")
    print("❌ Can't see relative importance of different failure modes")
    
    print("\n🟢 Stacked Failure Breakdown:")
    print("✅ Shows all failure types in context")
    print("✅ Violation rate visible as red portion")
    print("✅ Reveals dominant failure modes per model")
    print("✅ Same total information but with breakdown")
    
    print("\n📈 Detailed Failure Analysis:")
    for model in failure_breakdown.index:
        total_games = total_games_per_model[model]
        violation_rate = failure_breakdown.loc[model, 'Taboo_Violation'] / total_games
        timeout_rate = failure_breakdown.loc[model, 'Max_Turns_Exceeded'] / total_games
        other_rate = failure_breakdown.loc[model, 'Other_Failure'] / total_games
        total_failure_rate = (violation_rate + timeout_rate + other_rate)
        
        print(f"\n• {model}:")
        print(f"  - Total Failure Rate: {total_failure_rate:.3f}")
        print(f"  - Taboo Violations: {violation_rate:.3f} ({violation_rate/total_failure_rate*100:.1f}% of failures)")
        print(f"  - Max Turns Exceeded: {timeout_rate:.3f} ({timeout_rate/total_failure_rate*100:.1f}% of failures)")
        print(f"  - Other Failures: {other_rate:.3f} ({other_rate/total_failure_rate*100:.1f}% of failures)")
    
    print(f"\n✨ Charts saved to: comprehensive_figures/failure_reasons_stacked_comparison.png")
    
    return failure_breakdown, failure_rates

if __name__ == "__main__":
    failure_breakdown, failure_rates = create_failure_reasons_stacked_chart() 