#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将传统的Violation Rate柱状图改为信息丰富的堆叠图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置图表样式
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12

def create_comprehensive_stacked_chart():
    """创建完整的堆叠图替代传统violation rate图"""
    
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
    
    # 计算完整的性能分解
    performance_breakdown = results_df.groupby('hinter_model_clean').apply(
        lambda x: pd.Series({
            'Success': (x['success'] == True).sum(),
            'Taboo_Violation': ((x['success'] == False) & (x['has_taboo_violation'] == True)).sum(),
            'Other_Failure': ((x['success'] == False) & (x['has_taboo_violation'] == False)).sum()
        })
    ).astype(int)
    
    # 计算百分比
    performance_percentages = performance_breakdown.div(performance_breakdown.sum(axis=1), axis=0) * 100
    
    # 创建对比图：传统 vs 堆叠
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # === 左图：传统方法 ===
    # 只显示violation rate (重现您的图片)
    violation_rates = performance_breakdown['Taboo_Violation'] / performance_breakdown.sum(axis=1)
    
    colors_traditional = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 对应图片中的颜色
    bars1 = ax1.bar(violation_rates.index, violation_rates.values, 
                    color=colors_traditional, edgecolor='white', linewidth=0.8)
    
    ax1.set_title('Traditional Approach: Taboo Violation Rate Only', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Violation Rate', fontsize=12)
    ax1.set_ylim(0, 0.06)
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签（重现原图风格）
    for bar, rate in zip(bars1, violation_rates.values):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002, 
                f'{rate:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax1.tick_params(axis='x', rotation=45)
    
    # === 右图：堆叠图方法 ===
    # 显示完整性能分解
    colors_stacked = ['#2ca02c', '#d62728', '#ff7f0e']  # 绿色成功，红色违规，橙色其他失败
    
    performance_percentages.plot(kind='bar', stacked=True, ax=ax2, 
                                color=colors_stacked, edgecolor='white', linewidth=0.8)
    
    ax2.set_title('Stacked Approach: Complete Performance Breakdown', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.legend(['Success', 'Taboo Violation', 'Other Failures'], 
              title='Game Outcome', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 添加成功率标签
    for i, (model, row) in enumerate(performance_breakdown.iterrows()):
        total_games = row.sum()
        success_rate = row['Success'] / total_games * 100
        violation_rate = row['Taboo_Violation'] / total_games * 100
        ax2.text(i, 102, f'{success_rate:.1f}% success\n{violation_rate:.1f}% violation', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('comprehensive_figures/traditional_vs_stacked_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印对比分析
    print("=" * 60)
    print("📊 Traditional vs Stacked Chart Comparison")
    print("=" * 60)
    
    print("\n🔴 Traditional Approach (Your original chart):")
    print("❌ Only shows violation rates")
    print("❌ Missing success rate information")
    print("❌ No context about overall performance")
    print("❌ Requires multiple charts for complete picture")
    
    print("\n🟢 Stacked Chart Approach:")
    print("✅ Shows complete performance breakdown")
    print("✅ Success rate immediately visible (green portion)")
    print("✅ Violation rate in context (red portion)")
    print("✅ All information in single chart")
    
    print("\n📈 Detailed Results:")
    for model in performance_breakdown.index:
        row = performance_breakdown.loc[model]
        total = row.sum()
        success_pct = row['Success'] / total * 100
        violation_pct = row['Taboo_Violation'] / total * 100
        other_pct = row['Other_Failure'] / total * 100
        
        print(f"\n• {model}:")
        print(f"  - Success: {success_pct:.1f}% ({row['Success']} games)")
        print(f"  - Violations: {violation_pct:.1f}% ({row['Taboo_Violation']} games)")
        print(f"  - Other Failures: {other_pct:.1f}% ({row['Other_Failure']} games)")
    
    print(f"\n✨ Charts saved to: comprehensive_figures/traditional_vs_stacked_comparison.png")
    
    return performance_breakdown, performance_percentages

if __name__ == "__main__":
    performance_breakdown, performance_percentages = create_comprehensive_stacked_chart() 