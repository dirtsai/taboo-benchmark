#!/usr/bin/env python3
"""
中文实验错误分析 - 生成图4.22
Chinese Experiment Error Analysis for Taboo Benchmark - Figure 4.22
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

def load_chinese_experiment_data():
    """加载中文实验数据"""
    try:
        results_path = "results/chinese_full_experiment_20250717_222959/chinese_full_results_20250717_222959.csv"
        df = pd.read_csv(results_path)
        print(f"✅ 成功加载中文实验结果: {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"❌ 加载中文实验数据失败: {e}")
        return None

def analyze_model_errors(df):
    """分析各模型的错误情况"""
    # 模型名称映射
    label_map = {
        "openai/gpt-4o": "GPT-4o",
        "google/gemini-2.5-flash": "Gemini-2.5-Flash",
        "deepseek/deepseek-chat-v3-0324": "DeepSeek-V3",
        "moonshotai/kimi-k2": "Kimi-K2",
    }
    
    # 初始化错误计数器
    model_errors = {model_short: 0 for model_short in label_map.values()}
    
    print("分析各类错误...")
    
    # 1. 统计Hinter模型格式错误
    hinter_format_errors = df[
        df['failure_reason'].str.contains('线索生成失败: 格式验证失败', na=False)
    ]
    print(f"Hinter格式错误: {len(hinter_format_errors)} 条")
    
    for model in hinter_format_errors['hinter_model'].unique():
        if model in label_map:
            count = len(hinter_format_errors[hinter_format_errors['hinter_model'] == model])
            model_errors[label_map[model]] += count
            print(f"  {label_map[model]}: {count} 条Hinter格式错误")
    
    # 2. 统计Guesser模型格式错误
    guesser_format_errors = df[
        df['failure_reason'].str.contains('猜测生成失败: 格式验证失败', na=False)
    ]
    print(f"Guesser格式错误: {len(guesser_format_errors)} 条")
    
    for model in guesser_format_errors['guesser_model'].unique():
        if model in label_map:
            count = len(guesser_format_errors[guesser_format_errors['guesser_model'] == model])
            model_errors[label_map[model]] += count
            print(f"  {label_map[model]}: {count} 条Guesser格式错误")
    
    # 3. 统计Taboo违反错误
    taboo_violations = df[
        df['failure_reason'].str.contains('违反禁用词规则', na=False)
    ]
    print(f"Taboo违反错误: {len(taboo_violations)} 条")
    
    for model in taboo_violations['hinter_model'].unique():
        if model in label_map:
            count = len(taboo_violations[taboo_violations['hinter_model'] == model])
            model_errors[label_map[model]] += count
            print(f"  {label_map[model]}: {count} 条Taboo违反错误")
    
    # 转换为DataFrame
    error_df = pd.DataFrame({
        'Model': list(model_errors.keys()),
        'Error_Count': list(model_errors.values())
    }).sort_values('Error_Count', ascending=False)
    
    return error_df, model_errors

def main():
    print("开始中文实验错误分析...")
    
    # 1. 加载数据
    df = load_chinese_experiment_data()
    if df is None:
        return
    
    # 确保figures目录存在
    os.makedirs('figures', exist_ok=True)
    
    # 2. 分析模型错误
    error_df, model_errors = analyze_model_errors(df)
    
    # 3. 生成图4.22: 模型错误统计
    print("生成图4.22: 模型错误统计...")
    
    plt.figure(figsize=(10, 6))
    
    # 绘制柱状图
    bars = plt.bar(
        error_df['Model'],
        error_df['Error_Count'],
        color=colors[1],
        alpha=0.8,
        width=0.6
    )
    
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Error Count", fontsize=14)
    
    # 设置Y轴范围
    max_errors = error_df['Error_Count'].max()
    plt.ylim(0, max_errors * 1.15)
    
    # 添加数值标签
    for bar, (_, row) in zip(bars, error_df.iterrows()):
        plt.text(bar.get_x() + bar.get_width()/2, row['Error_Count'] + max_errors * 0.02,
                f'{int(row["Error_Count"])}', ha='center', va='bottom',
                fontsize=12, weight='bold')
    
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # 保存图4.22
    plt.savefig('figures/figure_4_22_chinese_model_errors.pdf', 
                dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure_4_22_chinese_model_errors.png', 
                dpi=300, bbox_inches='tight')
    print("✓ 图4.22已保存")
    plt.show()
    
    # 4. 打印详细分析结果
    print("\n" + "="*60)
    print("中文实验错误分析结果总结")
    print("="*60)
    
    print(f"\n模型错误排名 (按错误数量降序):")
    for i, (_, row) in enumerate(error_df.iterrows(), 1):
        print(f"  {i}. {row['Model']}: {int(row['Error_Count'])} 个错误")
    
    # 计算错误率
    total_records = len(df)
    total_errors = sum(model_errors.values())
    error_rate = total_errors / total_records
    
    print(f"\n整体错误统计:")
    print(f"  • 总记录数: {total_records:,}")
    print(f"  • 总错误数: {total_errors}")
    print(f"  • 整体错误率: {error_rate:.3f} ({error_rate*100:.1f}%)")
    
    # 分析错误类型分布
    hinter_format_count = len(df[df['failure_reason'].str.contains('线索生成失败: 格式验证失败', na=False)])
    guesser_format_count = len(df[df['failure_reason'].str.contains('猜测生成失败: 格式验证失败', na=False)])
    taboo_violation_count = len(df[df['failure_reason'].str.contains('违反禁用词规则', na=False)])
    
    print(f"\n错误类型分布:")
    print(f"  • Hinter格式错误: {hinter_format_count} ({hinter_format_count/total_errors*100:.1f}%)")
    print(f"  • Guesser格式错误: {guesser_format_count} ({guesser_format_count/total_errors*100:.1f}%)")
    print(f"  • Taboo违反错误: {taboo_violation_count} ({taboo_violation_count/total_errors*100:.1f}%)")
    
    # 找出最可靠和最不可靠的模型
    most_reliable = error_df.iloc[-1]  # 错误最少
    least_reliable = error_df.iloc[0]  # 错误最多
    
    print(f"\n模型可靠性对比:")
    print(f"  • 最可靠模型: {most_reliable['Model']} ({int(most_reliable['Error_Count'])} 个错误)")
    print(f"  • 最不可靠模型: {least_reliable['Model']} ({int(least_reliable['Error_Count'])} 个错误)")
    print(f"  • 可靠性差异: {int(least_reliable['Error_Count'] - most_reliable['Error_Count'])} 个错误")
    
    print(f"\n✅ 中文实验错误分析完成！图表已保存到 figures/ 目录")

if __name__ == "__main__":
    main()
