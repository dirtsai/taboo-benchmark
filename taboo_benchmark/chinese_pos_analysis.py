#!/usr/bin/env python3
"""
中文实验词性(POS)分析 - 生成图4.23和图4.24
Chinese Experiment POS Analysis for Taboo Benchmark - Figures 4.23 & 4.24
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
        print(f"词性类型: {sorted(df['part_of_speech'].unique())}")
        return df
    except Exception as e:
        print(f"❌ 加载中文实验数据失败: {e}")
        return None

def clean_model_names(df):
    """清理模型名称"""
    label_map = {
        "openai/gpt-4o": "GPT-4o",
        "google/gemini-2.5-flash": "Gemini-2.5-Flash",
        "deepseek/deepseek-chat-v3-0324": "DeepSeek-V3",
        "moonshotai/kimi-k2": "Kimi-K2",
    }
    
    df['hinter_model_clean'] = df['hinter_model'].map(label_map).fillna(df['hinter_model'])
    return df

def main():
    print("开始中文实验词性(POS)分析...")
    
    # 1. 加载数据
    df = load_chinese_experiment_data()
    if df is None:
        return
    
    # 2. 清理模型名称
    df = clean_model_names(df)
    
    # 确保figures目录存在
    os.makedirs('figures', exist_ok=True)
    
    # 3. 图4.23: 整体词性成功率
    print("生成图4.23: 整体词性成功率...")
    
    # 计算整体词性成功率并排序
    overall_pos_success = (
        df.groupby('part_of_speech')['success']
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    overall_pos_success.columns = ['Part_of_Speech', 'Success_Rate']
    
    plt.figure(figsize=(10, 6))
    
    # 绘制柱状图
    bars = plt.bar(
        overall_pos_success['Part_of_Speech'],
        overall_pos_success['Success_Rate'],
        color=colors[1],
        alpha=0.8,
        width=0.6
    )
    
    plt.xlabel("Part of Speech", fontsize=14)
    plt.ylabel("Success Rate", fontsize=14)
    plt.ylim(0, 1.05)
    
    # 添加1.0参考线
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    # 添加数值标签
    for bar, (_, row) in zip(bars, overall_pos_success.iterrows()):
        plt.text(bar.get_x() + bar.get_width()/2, row['Success_Rate'] + 0.02,
                f'{row["Success_Rate"]:.3f}', ha='center', va='bottom',
                fontsize=12, weight='bold')
    
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # 保存图4.23
    plt.savefig('figures/figure_4_23_chinese_pos_overall_success.pdf', 
                dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure_4_23_chinese_pos_overall_success.png', 
                dpi=300, bbox_inches='tight')
    print("✓ 图4.23已保存")
    plt.show()
    
    # 4. 图4.24: 按模型和词性的成功率
    print("生成图4.24: 按模型和词性的成功率...")
    
    # 计算按模型和词性的成功率
    pos_model_success = (
        df.groupby(['part_of_speech', 'hinter_model_clean'])['success']
        .mean()
        .reset_index()
    )
    pos_model_success.columns = ['Part_of_Speech', 'Model', 'Success_Rate']
    
    # 获取唯一的词性和模型
    pos_categories = sorted(df['part_of_speech'].unique())
    models = sorted(df['hinter_model_clean'].unique())
    
    plt.figure(figsize=(14, 8))
    
    # 设置分组柱状图
    bar_width = 0.2
    positions = np.arange(len(pos_categories))
    
    # 为每个模型绘制一组柱子
    for i, model in enumerate(models):
        model_data = pos_model_success[pos_model_success['Model'] == model]
        
        # 确保数据按照pos_categories的顺序排列
        ordered_data = []
        for pos in pos_categories:
            success_rate = model_data[model_data['Part_of_Speech'] == pos]['Success_Rate'].values
            if len(success_rate) > 0:
                ordered_data.append(success_rate[0])
            else:
                ordered_data.append(0)
        
        # 计算当前模型柱子的位置
        model_positions = positions + (i - 1.5) * bar_width
        
        # 绘制柱子
        plt.bar(
            model_positions, 
            ordered_data, 
            width=bar_width, 
            label=model,
            color=colors[i % len(colors)],
            alpha=0.8
        )
        
        # 在柱子上方添加数值标签
        for j, value in enumerate(ordered_data):
            if value > 0:  # 只在有数据时添加标签
                plt.text(
                    model_positions[j], 
                    value + 0.02, 
                    f"{value:.2f}", 
                    ha='center', 
                    va='bottom',
                    fontsize=9,
                    weight='bold'
                )
    
    plt.xlabel("Part of Speech", fontsize=14)
    plt.ylabel("Success Rate", fontsize=14)
    plt.ylim(0, 1.05)
    
    # 添加1.0参考线
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    # 设置x轴刻度位置和标签
    plt.xticks(positions, pos_categories, fontsize=12, rotation=0)
    plt.yticks(fontsize=12)
    
    # 添加图例
    plt.legend(title='Model', fontsize=11, title_fontsize=12, 
               bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 添加网格线
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图4.24
    plt.savefig('figures/figure_4_24_chinese_pos_model_success.pdf', 
                dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure_4_24_chinese_pos_model_success.png', 
                dpi=300, bbox_inches='tight')
    print("✓ 图4.24已保存")
    plt.show()
    
    # 5. 打印详细分析结果
    print("\n" + "="*60)
    print("中文实验词性(POS)分析结果总结")
    print("="*60)
    
    print(f"\n整体词性成功率排名:")
    for i, (_, row) in enumerate(overall_pos_success.iterrows(), 1):
        print(f"  {i}. {row['Part_of_Speech']}: {row['Success_Rate']:.3f}")
    
    # 分析词性差异
    pos_range = overall_pos_success['Success_Rate'].max() - overall_pos_success['Success_Rate'].min()
    print(f"\n词性间成功率差异: {pos_range:.3f}")
    
    # 找出最佳和最差词性
    best_pos = overall_pos_success.iloc[0]
    worst_pos = overall_pos_success.iloc[-1]
    
    print(f"\n极端值分析:")
    print(f"  • 最佳词性: {best_pos['Part_of_Speech']} ({best_pos['Success_Rate']:.3f})")
    print(f"  • 最差词性: {worst_pos['Part_of_Speech']} ({worst_pos['Success_Rate']:.3f})")
    
    # 分析各模型在不同词性上的表现
    print(f"\n各模型词性表现分析:")
    for model in models:
        model_data = pos_model_success[pos_model_success['Model'] == model]
        avg_success = model_data['Success_Rate'].mean()
        print(f"  • {model}: 平均成功率 {avg_success:.3f}")
        
        # 找出该模型最佳和最差词性
        if len(model_data) > 0:
            best_pos_for_model = model_data.loc[model_data['Success_Rate'].idxmax()]
            worst_pos_for_model = model_data.loc[model_data['Success_Rate'].idxmin()]
            print(f"    - 最佳词性: {best_pos_for_model['Part_of_Speech']} ({best_pos_for_model['Success_Rate']:.3f})")
            print(f"    - 最差词性: {worst_pos_for_model['Part_of_Speech']} ({worst_pos_for_model['Success_Rate']:.3f})")
    
    print(f"\n✅ 中文实验词性分析完成！图表已保存到 figures/ 目录")

if __name__ == "__main__":
    main()
