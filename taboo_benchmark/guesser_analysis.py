#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guesser 维度分析脚本
分析不同模型作为guesser时的表现
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 尝试导入可选库
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# 设置颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

def main():
    # 加载数据
    print("加载实验数据...")
    results_df = pd.read_csv('results/taboo_experiment_20250712_004918/complete_experiment_results.csv')

    # 清理模型名称
    model_name_mapping = {
        'anthropic/claude-sonnet-4': 'Claude Sonnet 4',
        'openai/gpt-4o': 'GPT-4o',
        'google/gemini-2.5-pro': 'Gemini 2.5 Pro',
        'deepseek/deepseek-chat-v3-0324': 'DeepSeek Chat V3'
    }

    results_df['hinter_model_clean'] = results_df['hinter_model'].map(model_name_mapping)
    results_df['guesser_model_clean'] = results_df['guesser_model'].map(model_name_mapping)

    # Guesser 维度性能分析
    print('🎯 Guesser 模型性能分析')
    print('='*60)

    # 1. 创建 4x4 成功率表格 (Hinter x Guesser)
    print('\n📊 4x4 成功率表格 (Hinter vs Guesser)')
    print('行：Hinter模型 | 列：Guesser模型\n')

    # 计算每个 hinter-guesser 组合的成功率
    hinter_guesser_success = results_df.groupby(['hinter_model_clean', 'guesser_model_clean'])['success'].mean().unstack()

    # 创建一个干净的 4x4 表格
    models_order = ['Claude Sonnet 4', 'Gemini 2.5 Pro', 'DeepSeek Chat V3', 'GPT-4o']

    # 重新排序行和列
    hinter_guesser_table = hinter_guesser_success.reindex(models_order, columns=models_order)

    print('成功率矩阵 (行：Hinter | 列：Guesser):')
    print(hinter_guesser_table.round(3).to_string())

    # 2. Guesser 模型整体表现
    print('\n\n🎯 Guesser 模型整体表现:')
    guesser_overall = results_df.groupby('guesser_model_clean').agg({
        'success': ['count', 'mean'],
        'turns_used': lambda x: x[results_df.loc[x.index, 'success']].mean(),
        'has_taboo_violation': 'mean'
    }).round(3)

    guesser_overall.columns = ['总游戏数', '成功率', '平均轮数', '违规率']
    guesser_overall = guesser_overall.reindex(models_order)

    print(guesser_overall.to_string())

    # 3. 与 Hinter 表现对比
    print('\n\n🔄 Hinter vs Guesser 表现对比:')
    hinter_overall = results_df.groupby('hinter_model_clean').agg({
        'success': 'mean',
        'turns_used': lambda x: x[results_df.loc[x.index, 'success']].mean(),
    }).round(3)

    comparison_table = pd.DataFrame({
        'Hinter成功率': hinter_overall['success'],
        'Guesser成功率': guesser_overall['成功率'],
        'Hinter平均轮数': hinter_overall['turns_used'],
        'Guesser平均轮数': guesser_overall['平均轮数']
    })

    comparison_table['成功率差异'] = comparison_table['Guesser成功率'] - comparison_table['Hinter成功率']
    comparison_table['轮数差异'] = comparison_table['Guesser平均轮数'] - comparison_table['Hinter平均轮数']

    print(comparison_table.round(3).to_string())

    # 5. 关键发现总结
    print('\n\n🔍 关键发现:')

    # 找出最佳和最差的 guesser
    best_guesser = guesser_overall['成功率'].idxmax()
    worst_guesser = guesser_overall['成功率'].idxmin()
    best_rate = guesser_overall.loc[best_guesser, '成功率']
    worst_rate = guesser_overall.loc[worst_guesser, '成功率']

    print(f'  • 最佳Guesser: {best_guesser} ({best_rate:.1%})')
    print(f'  • 最差Guesser: {worst_guesser} ({worst_rate:.1%})')
    print(f'  • Guesser性能差距: {best_rate - worst_rate:.1%}')

    # 分析角色适应性
    print('\n🎭 角色适应性分析:')
    for model in models_order:
        hinter_rate = comparison_table.loc[model, 'Hinter成功率']
        guesser_rate = comparison_table.loc[model, 'Guesser成功率']
        diff = comparison_table.loc[model, '成功率差异']
        
        if abs(diff) < 0.02:
            role_pref = '平衡型'
        elif diff > 0:
            role_pref = '更适合Guesser'
        else:
            role_pref = '更适合Hinter'
        
        print(f'  • {model}: {role_pref} (Hinter: {hinter_rate:.1%}, Guesser: {guesser_rate:.1%}, 差异: {diff:+.1%})')

    # 最佳组合分析
    print('\n🎯 最佳Hinter-Guesser组合:')
    best_combinations = []
    for hinter in models_order:
        for guesser in models_order:
            if not pd.isna(hinter_guesser_table.loc[hinter, guesser]):
                best_combinations.append((hinter, guesser, hinter_guesser_table.loc[hinter, guesser]))

    best_combinations.sort(key=lambda x: x[2], reverse=True)

    print('Top 5 最佳组合:')
    for i, (hinter, guesser, rate) in enumerate(best_combinations[:5], 1):
        print(f'  {i}. {hinter} (Hinter) + {guesser} (Guesser): {rate:.1%}')

    print('\nBottom 5 最差组合:')
    for i, (hinter, guesser, rate) in enumerate(best_combinations[-5:], 1):
        print(f'  {i}. {hinter} (Hinter) + {guesser} (Guesser): {rate:.1%}')

    print('\n✅ Guesser 维度分析完成！')
    
    # 保存结果到CSV文件
    print('\n💾 保存分析结果...')
    
    # 保存4x4矩阵
    hinter_guesser_table.to_csv('guesser_analysis_4x4_matrix.csv', encoding='utf-8')
    
    # 保存guesser整体表现
    guesser_overall.to_csv('guesser_overall_performance.csv', encoding='utf-8')
    
    # 保存角色对比
    comparison_table.to_csv('hinter_guesser_comparison.csv', encoding='utf-8')
    
    print('  • 4x4成功率矩阵已保存至: guesser_analysis_4x4_matrix.csv')
    print('  • Guesser整体表现已保存至: guesser_overall_performance.csv') 
    print('  • 角色对比表已保存至: hinter_guesser_comparison.csv')

if __name__ == '__main__':
    main() 