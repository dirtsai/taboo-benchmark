#!/usr/bin/env python3
"""
定量分析成功率热力图 - 生成图4.28-4.33
Quantitative Analysis Success Rate Heatmaps - Figures 4.28-4.33
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
sns.set_theme(style="white", font_scale=1.2)

def load_and_process_data():
    """加载并处理定量分析数据"""
    print("加载定量分析数据...")
    
    results_path = "results/quantitative_analysis_20250717_213821/quantitative_analysis_merged_20250717_213821.csv"
    df = pd.read_csv(results_path)
    
    print(f"原始数据行数: {len(df)}")
    
    # 过滤特定模型
    target_models = ['deepseek/deepseek-chat-v3-0324', 'google/gemini-2.5-flash']
    df_filtered = df[df['hinter_model'].isin(target_models)].copy()
    
    # 过滤特定温度
    target_temperatures = [0.1, 0.3, 0.7]
    df_filtered = df_filtered[df_filtered['temperature'].isin(target_temperatures)]
    
    # 过滤特定禁忌词数量
    target_taboo_counts = [1, 3, 5]
    df_filtered = df_filtered[df_filtered['taboo_count'].isin(target_taboo_counts)]
    
    # 过滤特定提示词长度
    target_hint_lengths = [1, 5, 10]
    df_filtered = df_filtered[df_filtered['hint_word_count'].isin(target_hint_lengths)]
    
    print(f"过滤后数据行数: {len(df_filtered)}")
    print(f"过滤后的模型: {df_filtered['hinter_model'].unique()}")
    print(f"过滤后的温度: {sorted(df_filtered['temperature'].unique())}")
    print(f"过滤后的taboo_count: {sorted(df_filtered['taboo_count'].unique())}")
    print(f"过滤后的hint_word_count: {sorted(df_filtered['hint_word_count'].unique())}")
    
    # 模型名称映射
    label_map = {
        "deepseek/deepseek-chat-v3-0324": "DeepSeek-V3",
        "google/gemini-2.5-flash": "Gemini-2.5-Flash",
    }
    
    df_filtered['model_short'] = df_filtered['hinter_model'].map(label_map)
    
    return df_filtered, target_temperatures, target_taboo_counts, target_hint_lengths

def create_success_rate_matrix(data, model, target_hint_lengths, target_temperatures):
    """为指定模型创建成功率矩阵"""
    model_data = data[data['model_short'] == model]
    
    if len(model_data) == 0:
        # 创建空矩阵
        pivot_table = pd.DataFrame(index=target_hint_lengths, columns=target_temperatures)
        pivot_table = pivot_table.fillna(0)
        return pivot_table
    
    # 计算成功率
    success_rates = model_data.groupby(['hint_word_count', 'temperature']).agg({
        'success': ['count', 'sum']
    })
    success_rates.columns = ['total_count', 'success_count']
    success_rates['success_rate'] = success_rates['success_count'] / success_rates['total_count']
    success_rates = success_rates.reset_index()
    
    # 创建透视表
    pivot_table = success_rates.pivot(index='hint_word_count', columns='temperature', values='success_rate')
    
    # 确保所有组合都存在
    pivot_table = pivot_table.reindex(index=target_hint_lengths, columns=target_temperatures)
    
    return pivot_table

def create_individual_heatmaps(df_filtered, target_temperatures, target_taboo_counts, target_hint_lengths):
    """创建单独的热力图（图4.28-4.33）"""
    print("生成单独热力图...")
    
    # 确保figures目录存在
    os.makedirs('figures', exist_ok=True)
    
    figure_numbers = {
        ('DeepSeek-V3', 1): '4.28',
        ('Gemini-2.5-Flash', 1): '4.29',
        ('DeepSeek-V3', 3): '4.30',
        ('Gemini-2.5-Flash', 3): '4.31',
        ('DeepSeek-V3', 5): '4.32',
        ('Gemini-2.5-Flash', 5): '4.33'
    }
    
    for taboo_count in target_taboo_counts:
        print(f"\n处理 Taboo Count = {taboo_count}")
        
        # 过滤当前taboo_count的数据
        taboo_data = df_filtered[df_filtered['taboo_count'] == taboo_count]
        
        if len(taboo_data) == 0:
            print(f"警告: taboo_count={taboo_count} 没有数据")
            continue
        
        for model in ['DeepSeek-V3', 'Gemini-2.5-Flash']:
            # 创建成功率矩阵
            pivot_table = create_success_rate_matrix(taboo_data, model, target_hint_lengths, target_temperatures)
            
            # 打印成功率表
            print(f"\n{model} 成功率矩阵（taboo_count={taboo_count}）:")
            print("hint_word_count ↓ \\ temperature →\t", end="")
            for temp in target_temperatures:
                print(f"{temp}\t", end="")
            print()
            
            for hint_len in target_hint_lengths:
                print(f"{hint_len}\t\t\t\t", end="")
                for temp in target_temperatures:
                    if hint_len in pivot_table.index and temp in pivot_table.columns:
                        rate = pivot_table.loc[hint_len, temp]
                        if pd.isna(rate):
                            print("N/A\t", end="")
                        else:
                            print(f"{rate:.3f}\t", end="")
                    else:
                        print("N/A\t", end="")
                print()
            
            # 创建单独热力图
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # 使用magma色板保持一致性
            cmap = 'magma'
            
            # 绘制热力图
            sns.heatmap(pivot_table, 
                       annot=True, fmt='.3f', cmap=cmap,
                       cbar_kws={'label': 'Success Rate'},
                       ax=ax, vmin=0, vmax=1)
            
            # 设置标题和标签
            figure_num = figure_numbers.get((model, taboo_count), 'X.X')
            ax.set_xlabel('Temperature', fontsize=14)
            ax.set_ylabel('Hint Word Count', fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            plt.tight_layout()
            
            # 保存图表
            filename = f"figure_{figure_num.replace('.', '_')}_{model.lower().replace('-', '_')}_taboo_{taboo_count}_heatmap"
            plt.savefig(f'figures/{filename}.pdf', dpi=300, bbox_inches='tight')
            plt.savefig(f'figures/{filename}.png', dpi=300, bbox_inches='tight')
            print(f"✓ 图{figure_num}已保存")
            
            plt.show()

def create_combined_heatmaps(df_filtered, target_temperatures, target_taboo_counts, target_hint_lengths):
    """创建并排热力图（用于对比）"""
    print("生成并排对比热力图...")
    
    for taboo_count in target_taboo_counts:
        print(f"\n处理 Taboo Count = {taboo_count} 并排图")
        
        # 过滤当前taboo_count的数据
        taboo_data = df_filtered[df_filtered['taboo_count'] == taboo_count]
        
        if len(taboo_data) == 0:
            continue
        
        # 为两个模型创建成功率矩阵
        deepseek_matrix = create_success_rate_matrix(taboo_data, 'DeepSeek-V3', target_hint_lengths, target_temperatures)
        gemini_matrix = create_success_rate_matrix(taboo_data, 'Gemini-2.5-Flash', target_hint_lengths, target_temperatures)
        
        # 创建并排热力图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # DeepSeek热力图
        sns.heatmap(deepseek_matrix, 
                   annot=True, fmt='.3f', cmap='plasma', 
                   cbar_kws={'label': 'Success Rate'},
                   ax=ax1, vmin=0, vmax=1)
        ax1.set_title(f'DeepSeek-V3 (Taboo Count = {taboo_count})', fontsize=16)
        ax1.set_xlabel('Temperature', fontsize=14)
        ax1.set_ylabel('Hint Word Count', fontsize=14)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        
        # Gemini热力图
        sns.heatmap(gemini_matrix, 
                   annot=True, fmt='.3f', cmap='viridis', 
                   cbar_kws={'label': 'Success Rate'},
                   ax=ax2, vmin=0, vmax=1)
        ax2.set_title(f'Gemini-2.5-Flash (Taboo Count = {taboo_count})', fontsize=16)
        ax2.set_xlabel('Temperature', fontsize=14)
        ax2.set_ylabel('Hint Word Count', fontsize=14)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()
        
        # 保存并排图
        filename = f"combined_heatmaps_taboo_count_{taboo_count}"
        plt.savefig(f'figures/{filename}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'figures/{filename}.png', dpi=300, bbox_inches='tight')
        print(f"✓ 并排热力图（taboo_count={taboo_count}）已保存")
        
        plt.show()

def generate_statistics_summary(df_filtered):
    """生成统计汇总"""
    print("\n" + "="*60)
    print("整体统计汇总")
    print("="*60)
    
    overall_stats = df_filtered.groupby(['model_short', 'taboo_count', 'hint_word_count', 'temperature']).agg({
        'success': ['count', 'sum']
    })
    overall_stats.columns = ['total_count', 'success_count']
    overall_stats['success_rate'] = overall_stats['success_count'] / overall_stats['total_count']
    overall_stats = overall_stats.reset_index()
    
    print("详细统计表:")
    print(overall_stats.to_string(index=False))
    
    # 按模型和禁忌词数量汇总
    print(f"\n按模型和禁忌词数量的平均成功率:")
    model_taboo_summary = overall_stats.groupby(['model_short', 'taboo_count'])['success_rate'].mean().reset_index()
    for _, row in model_taboo_summary.iterrows():
        print(f"  {row['model_short']} (taboo_count={row['taboo_count']}): {row['success_rate']:.3f}")

def main():
    print("开始定量分析成功率热力图生成...")
    
    try:
        # 1. 加载和处理数据
        df_filtered, target_temperatures, target_taboo_counts, target_hint_lengths = load_and_process_data()
        
        # 2. 创建单独热力图（图4.28-4.33）
        create_individual_heatmaps(df_filtered, target_temperatures, target_taboo_counts, target_hint_lengths)
        
        # 3. 创建并排对比热力图
        create_combined_heatmaps(df_filtered, target_temperatures, target_taboo_counts, target_hint_lengths)
        
        # 4. 生成统计汇总
        generate_statistics_summary(df_filtered)
        
        print(f"\n✅ 所有热力图已保存到 figures/ 目录")
        print("生成的图表:")
        print("  • 图4.28: DeepSeek-V3在1个禁忌词下的成功率热力图")
        print("  • 图4.29: Gemini-2.5-Flash在1个禁忌词下的成功率热力图")
        print("  • 图4.30: DeepSeek-V3在3个禁忌词下的成功率热力图")
        print("  • 图4.31: Gemini-2.5-Flash在3个禁忌词下的成功率热力图")
        print("  • 图4.32: DeepSeek-V3在5个禁忌词下的成功率热力图")
        print("  • 图4.33: Gemini-2.5-Flash在5个禁忌词下的成功率热力图")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
