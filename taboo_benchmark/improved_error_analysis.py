#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的错误分析脚本 - 使用堆叠图展示失败原因
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 颜色配置
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

def load_and_prepare_data():
    """加载和准备数据"""
    results_df = pd.read_csv('results/taboo_experiment_20250712_004918/complete_experiment_results.csv')
    
    # 清理模型名称
    model_name_mapping = {
        'anthropic/claude-sonnet-4': 'Claude Sonnet 4',
        'openai/gpt-4o': 'GPT-4o', 
        'google/gemini-2.5-pro': 'Gemini 2.5 Pro',
        'deepseek/deepseek-chat-v3-0324': 'DeepSeek Chat V3'
    }
    
    results_df['hinter_model_clean'] = results_df['hinter_model'].map(model_name_mapping)
    return results_df

def create_failure_stack_charts(merged_df):
    """创建改进的失败原因堆叠图"""
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Enhanced Failure Analysis with Stacked Charts', fontsize=16, fontweight='bold', y=0.98)
    
    # 1. 失败原因按模型的堆叠柱状图 (绝对数量)
    failed_games = merged_df[merged_df['success'] == False]
    
    # 准备失败原因数据
    failure_stack_data = failed_games.groupby(['hinter_model_clean', 'failure_reason']).size().unstack(fill_value=0)
    
    # 定义失败原因颜色映射
    failure_colors = {
        'MAX_TURNS_EXCEEDED': '#ff9999',
        'TABOO_VIOLATION': '#ffcc99', 
        'FORMAT_FAILURE': '#99ccff',
        'API_FAILURE': '#cc99ff',
        'OTHER': '#99ff99'
    }
    
    # 确保所有失败原因都有颜色
    failure_reasons = failure_stack_data.columns.tolist()
    chart_colors = [failure_colors.get(reason, colors[i % len(colors)]) for i, reason in enumerate(failure_reasons)]
    
    # 绘制堆叠柱状图 (绝对数量)
    failure_stack_data.plot(kind='bar', stacked=True, ax=ax1, color=chart_colors, width=0.7)
    ax1.set_title('Failure Reasons by Model (Absolute Counts)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Number of Failed Games')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(title='Failure Reason', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 添加总失败数标签
    for i, (model, row) in enumerate(failure_stack_data.iterrows()):
        total_failures = row.sum()
        ax1.text(i, total_failures + 5, f'{total_failures}', ha='center', va='bottom', fontweight='bold')
    
    # 2. 失败原因按模型的堆叠柱状图 (百分比)
    failure_pct_data = failure_stack_data.div(failure_stack_data.sum(axis=1), axis=0) * 100
    failure_pct_data.plot(kind='bar', stacked=True, ax=ax2, color=chart_colors, width=0.7)
    ax2.set_title('Failure Reasons by Model (Percentage)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Percentage of Failed Games')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title='Failure Reason', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_ylim(0, 100)
    
    # 添加百分比标签
    for i, (model, row) in enumerate(failure_pct_data.iterrows()):
        cumulative = 0
        for j, (reason, pct) in enumerate(row.items()):
            if pct > 5:  # 只显示超过5%的标签
                ax2.text(i, cumulative + pct/2, f'{pct:.1f}%', ha='center', va='center', 
                        fontweight='bold', fontsize=9)
            cumulative += pct
    
    # 3. 成功率 vs 失败率堆叠图
    model_performance = merged_df.groupby('hinter_model_clean').agg({
        'success': ['count', 'sum']
    }).round(1)
    model_performance.columns = ['total_games', 'successful_games']
    model_performance['failed_games'] = model_performance['total_games'] - model_performance['successful_games']
    model_performance['success_rate'] = model_performance['successful_games'] / model_performance['total_games'] * 100
    model_performance['failure_rate'] = model_performance['failed_games'] / model_performance['total_games'] * 100
    
    # 堆叠柱状图 - 成功vs失败
    x_pos = np.arange(len(model_performance))
    ax3.bar(x_pos, model_performance['success_rate'], label='Success Rate', 
           color='#2ecc71', alpha=0.8, width=0.6)
    ax3.bar(x_pos, model_performance['failure_rate'], 
           bottom=model_performance['success_rate'], label='Failure Rate',
           color='#e74c3c', alpha=0.8, width=0.6)
    
    ax3.set_title('Success vs Failure Rate by Model', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Percentage')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(model_performance.index, rotation=45)
    ax3.legend()
    ax3.set_ylim(0, 100)
    
    # 添加成功率标签
    for i, (model, row) in enumerate(model_performance.iterrows()):
        success_rate = row['success_rate']
        ax3.text(i, success_rate/2, f'{success_rate:.1f}%', ha='center', va='center', 
                fontweight='bold', color='white', fontsize=10)
        if row['failure_rate'] > 5:
            ax3.text(i, success_rate + row['failure_rate']/2, f'{row["failure_rate"]:.1f}%', 
                    ha='center', va='center', fontweight='bold', color='white', fontsize=10)
    
    # 4. 违规类型的详细堆叠分析
    violation_games = merged_df[merged_df['has_taboo_violation'] == True]
    if len(violation_games) > 0:
        # 按模型分析违规游戏的turn分布
        violation_turn_data = violation_games.groupby(['hinter_model_clean', 'taboo_violation_turn']).size().unstack(fill_value=0)
        
        # 使用渐变色显示不同轮次的违规
        turn_colors = ['#ffe6e6', '#ffb3b3', '#ff8080', '#ff4d4d', '#ff1a1a']
        
        if not violation_turn_data.empty:
            violation_turn_data.plot(kind='bar', stacked=True, ax=ax4, 
                                   color=turn_colors[:len(violation_turn_data.columns)], width=0.7)
            ax4.set_title('Taboo Violations by Turn and Model', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Model')
            ax4.set_ylabel('Number of Violations')
            ax4.tick_params(axis='x', rotation=45)
            ax4.legend(title='Violation Turn', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # 添加总违规数标签
            for i, (model, row) in enumerate(violation_turn_data.iterrows()):
                total_violations = row.sum()
                if total_violations > 0:
                    ax4.text(i, total_violations + 1, f'{total_violations}', 
                            ha='center', va='bottom', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No Taboo Violations Found', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Taboo Violations by Turn and Model', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # 保存图片
    plt.savefig('comprehensive_figures/enhanced_failure_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Enhanced failure analysis saved as 'enhanced_failure_analysis.png'")
    
    return fig

def create_comparative_stack_chart(merged_df):
    """创建模型对比的综合堆叠图"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. 总体游戏结果堆叠图
    model_results = merged_df.groupby('hinter_model_clean').agg({
        'success': ['count', 'sum'],
        'has_taboo_violation': 'sum',
        'has_format_errors': 'sum'
    })
    
    model_results.columns = ['total_games', 'successful_games', 'violation_games', 'format_error_games']
    model_results['failed_games'] = model_results['total_games'] - model_results['successful_games']
    model_results['clean_failed_games'] = (model_results['failed_games'] - 
                                          model_results['violation_games'] - 
                                          model_results['format_error_games'])
    
    # 创建堆叠数据
    stack_data = pd.DataFrame({
        'Success': model_results['successful_games'],
        'Clean Failure': model_results['clean_failed_games'], 
        'Taboo Violation': model_results['violation_games'],
        'Format Error': model_results['format_error_games']
    })
    
    # 绘制堆叠图
    colors_stack = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    stack_data.plot(kind='bar', stacked=True, ax=ax1, color=colors_stack, width=0.7)
    ax1.set_title('Game Outcomes by Model (Absolute)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Number of Games')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 添加总数标签
    for i, (model, row) in enumerate(stack_data.iterrows()):
        total = row.sum()
        ax1.text(i, total + 10, f'{total}', ha='center', va='bottom', fontweight='bold')
    
    # 2. 百分比堆叠图
    stack_pct = stack_data.div(stack_data.sum(axis=1), axis=0) * 100
    stack_pct.plot(kind='bar', stacked=True, ax=ax2, color=colors_stack, width=0.7)
    ax2.set_title('Game Outcomes by Model (Percentage)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Percentage')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_ylim(0, 100)
    
    # 添加主要类别的百分比标签
    for i, (model, row) in enumerate(stack_pct.iterrows()):
        cumulative = 0
        for j, (category, pct) in enumerate(row.items()):
            if pct > 8:  # 只显示超过8%的标签
                ax2.text(i, cumulative + pct/2, f'{pct:.1f}%', ha='center', va='center',
                        fontweight='bold', color='white' if j == 0 else 'black', fontsize=9)
            cumulative += pct
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('comprehensive_figures/comparative_stack_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Comparative stack analysis saved as 'comparative_stack_analysis.png'")
    
    return fig

def print_enhanced_failure_summary(merged_df):
    """打印增强的失败分析总结"""
    
    print("\n" + "="*80)
    print("🔍 ENHANCED FAILURE ANALYSIS SUMMARY")
    print("="*80)
    
    # 整体失败统计
    total_games = len(merged_df)
    failed_games = len(merged_df[merged_df['success'] == False])
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Total Games: {total_games:,}")
    print(f"   Failed Games: {failed_games:,} ({failed_games/total_games*100:.1f}%)")
    print(f"   Success Rate: {(total_games-failed_games)/total_games*100:.1f}%")
    
    # 按失败原因分析
    print(f"\n❌ Failure Breakdown:")
    failure_counts = merged_df[merged_df['success'] == False]['failure_reason'].value_counts()
    for reason, count in failure_counts.items():
        pct = count / failed_games * 100
        pct_total = count / total_games * 100
        print(f"   {reason}: {count} games ({pct:.1f}% of failures, {pct_total:.1f}% of total)")
    
    # 按模型分析失败模式
    print(f"\n🤖 Model-Specific Failure Patterns:")
    for model in merged_df['hinter_model_clean'].unique():
        model_df = merged_df[merged_df['hinter_model_clean'] == model]
        model_failed = model_df[model_df['success'] == False]
        
        print(f"\n   📱 {model}:")
        print(f"      Total Games: {len(model_df)}")
        print(f"      Failed Games: {len(model_failed)} ({len(model_failed)/len(model_df)*100:.1f}%)")
        
        if len(model_failed) > 0:
            model_failures = model_failed['failure_reason'].value_counts()
            for reason, count in model_failures.items():
                pct = count / len(model_failed) * 100
                print(f"         {reason}: {count} ({pct:.1f}%)")
    
    # 违规详细分析
    violation_games = merged_df[merged_df['has_taboo_violation'] == True]
    if len(violation_games) > 0:
        print(f"\n🚫 Taboo Violation Details:")
        print(f"   Total Violations: {len(violation_games)}")
        
        # 按轮次分析违规
        violation_by_turn = violation_games['taboo_violation_turn'].value_counts().sort_index()
        print(f"   Violations by Turn:")
        for turn, count in violation_by_turn.items():
            pct = count / len(violation_games) * 100
            print(f"      Turn {turn}: {count} ({pct:.1f}%)")
    
    print("\n" + "="*80)

def main():
    """主函数"""
    print("🔍 Enhanced Failure Analysis with Stacked Charts")
    print("="*60)
    
    # 加载数据
    print("Loading data...")
    merged_df = load_and_prepare_data()
    
    # 创建增强的失败分析图表
    print("\nCreating enhanced failure analysis charts...")
    fig1 = create_failure_stack_charts(merged_df)
    plt.show()
    
    # 创建综合对比堆叠图
    print("\nCreating comparative stack charts...")
    fig2 = create_comparative_stack_chart(merged_df)
    plt.show()
    
    # 打印详细分析
    print_enhanced_failure_summary(merged_df)
    
    print("\n✅ Enhanced failure analysis completed!")
    print("📁 Charts saved in 'comprehensive_figures/' directory")

if __name__ == "__main__":
    main() 