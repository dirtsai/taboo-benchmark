#!/usr/bin/env python3
"""
BERT语义相似度分析 - 生成图4.17和图4.18
BERT Semantic Similarity Analysis for Taboo Benchmark - Figures 4.17 & 4.18
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

def load_bert_similarity_results(filepath):
    """加载BERT相似度分析结果"""
    try:
        similarity_df = pd.read_csv(filepath, encoding='utf-8')
        print(f"✅ 成功加载BERT相似度结果: {len(similarity_df)} 条记录")
        print(f"数据列: {list(similarity_df.columns)}")
        
        # 显示基本统计
        print(f"\n📊 基本统计:")
        print(f"  • 总猜测数: {len(similarity_df):,}")
        print(f"  • 平均相似度: {similarity_df['similarity'].mean():.4f}")
        print(f"  • 相似度范围: {similarity_df['similarity'].min():.4f} - {similarity_df['similarity'].max():.4f}")
        
        return similarity_df
    except Exception as e:
        print(f"❌ 加载结果文件失败: {e}")
        return None

def clean_model_names(df):
    """清理模型名称"""
    label_map = {
        "openai/gpt-4o": "GPT-4o",
        "google/gemini-2.5-pro": "Gemini-2.5-Pro", 
        "deepseek/deepseek-chat-v3-0324": "DeepSeek-V3",
        "anthropic/claude-sonnet-4": "Claude-Sonnet-4",
    }
    
    if 'hinter_model' in df.columns:
        df['hinter_model_clean'] = df['hinter_model'].map(label_map).fillna(df['hinter_model'])
    if 'guesser_model' in df.columns:
        df['guesser_model_clean'] = df['guesser_model'].map(label_map).fillna(df['guesser_model'])
    
    return df

def main():
    print("开始BERT语义相似度分析...")
    
    # 1. 加载数据
    result_file = "bert_multi_model_similarity_analysis.csv"
    similarity_df = load_bert_similarity_results(result_file)
    
    if similarity_df is None:
        print("❌ 无法加载数据，请检查文件路径")
        return
    
    # 2. 清理模型名称
    similarity_df = clean_model_names(similarity_df)
    
    # 确保figures目录存在
    os.makedirs('figures', exist_ok=True)
    
    # 3. 图4.17: 四款模型担任Hinter时的平均相似度对比
    print("生成图4.17: Hinter模型平均相似度对比...")
    
    if 'hinter_model_clean' in similarity_df.columns:
        # 计算各Hinter模型的平均相似度
        hinter_stats = similarity_df.groupby('hinter_model_clean').agg({
            'similarity': ['mean', 'std', 'count']
        }).round(4)
        hinter_stats.columns = ['Mean_Similarity', 'Std_Similarity', 'Count']
        hinter_stats = hinter_stats.reset_index().sort_values('Mean_Similarity', ascending=False)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            hinter_stats['hinter_model_clean'], 
            hinter_stats['Mean_Similarity'],
            color=colors[1], 
            alpha=0.8,
            width=0.6
        )
        
        plt.xlabel("Hinter Model", fontsize=14)
        plt.ylabel("Average Similarity", fontsize=14)
        plt.ylim(0, 1.05)
        
        # 添加1.0参考线
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        
        # 添加数值标签
        for i, (bar, row) in enumerate(zip(bars, hinter_stats.itertuples())):
            plt.text(bar.get_x() + bar.get_width()/2, row.Mean_Similarity + 0.02, 
                    f'{row.Mean_Similarity:.3f}', ha='center', va='bottom', 
                    fontsize=12, weight='bold')
            # 添加样本数量标签
            plt.text(bar.get_x() + bar.get_width()/2, 0.05, f"n={row.Count}", 
                    ha='center', va='bottom', color='white', fontsize=12, fontweight='bold')
        
        plt.xticks(fontsize=12, rotation=0)
        plt.yticks(fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # 保存图4.17
        plt.savefig('figures/figure_4_17_hinter_similarity_comparison.pdf', 
                    dpi=300, bbox_inches='tight')
        plt.savefig('figures/figure_4_17_hinter_similarity_comparison.png', 
                    dpi=300, bbox_inches='tight')
        print("✓ 图4.17已保存")
        plt.show()
        
        # 打印Hinter模型排名
        print("\nHinter模型平均相似度排名:")
        for i, (_, row) in enumerate(hinter_stats.iterrows(), 1):
            print(f"  {i}. {row['hinter_model_clean']}: {row['Mean_Similarity']:.3f} "
                  f"({int(row['Count'])} 个样本)")
    
    # 4. 图4.18: 平均相似度随轮次变化趋势图
    print("生成图4.18: 相似度随轮次变化趋势...")
    
    if 'turn_number' in similarity_df.columns:
        # 计算各轮次的平均相似度
        turn_stats = similarity_df.groupby('turn_number').agg({
            'similarity': ['mean', 'std', 'count']
        }).round(4)
        turn_stats.columns = ['Mean_Similarity', 'Std_Similarity', 'Count']
        turn_stats = turn_stats.reset_index()
        
        plt.figure(figsize=(12, 8))
        
        # 绘制主趋势线
        plt.plot(turn_stats['turn_number'], turn_stats['Mean_Similarity'], 
                'o-', color=colors[2], linewidth=3, markersize=8, 
                label='Average Similarity')
        
        # 添加置信区间（标准误差）
        plt.fill_between(turn_stats['turn_number'], 
                        turn_stats['Mean_Similarity'] - turn_stats['Std_Similarity']/np.sqrt(turn_stats['Count']),
                        turn_stats['Mean_Similarity'] + turn_stats['Std_Similarity']/np.sqrt(turn_stats['Count']), 
                        color=colors[2], alpha=0.3, label='Standard Error')
        
        plt.xlabel("Turn Number", fontsize=14)
        plt.ylabel("Average Similarity", fontsize=14)
        plt.ylim(0, 1.05)
        
        # 添加1.0参考线
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        
        # 添加数值标签
        for _, row in turn_stats.iterrows():
            plt.text(row['turn_number'], row['Mean_Similarity'] + 0.03, 
                    f"{row['Mean_Similarity']:.3f}", ha='center', va='bottom', 
                    fontsize=11, weight='bold')
        
        plt.xticks(turn_stats['turn_number'], fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # 保存图4.18
        plt.savefig('figures/figure_4_18_turn_similarity_trend.pdf', 
                    dpi=300, bbox_inches='tight')
        plt.savefig('figures/figure_4_18_turn_similarity_trend.png', 
                    dpi=300, bbox_inches='tight')
        print("✓ 图4.18已保存")
        plt.show()
        
        # 打印轮次趋势分析
        print("\n轮次相似度变化趋势:")
        for _, row in turn_stats.iterrows():
            print(f"  • 第{int(row['turn_number'])}轮: {row['Mean_Similarity']:.3f} "
                  f"({int(row['Count'])} 个样本)")
        
        # 计算趋势
        first_turn = turn_stats.iloc[0]['Mean_Similarity']
        last_turn = turn_stats.iloc[-1]['Mean_Similarity']
        trend_change = last_turn - first_turn
        
        print(f"\n趋势分析:")
        print(f"  • 首轮相似度: {first_turn:.3f}")
        print(f"  • 末轮相似度: {last_turn:.3f}")
        print(f"  • 总体变化: {trend_change:+.3f}")
        
        if trend_change < -0.01:
            print(f"  • 结论: 相似度随轮次递减，存在'信息稀释'现象")
        elif trend_change > 0.01:
            print(f"  • 结论: 相似度随轮次递增")
        else:
            print(f"  • 结论: 相似度在各轮次间保持相对稳定")
    
    # 5. 打印综合分析结果
    print("\n" + "="*60)
    print("BERT语义相似度分析结果总结")
    print("="*60)
    
    overall_similarity = similarity_df['similarity'].mean()
    print(f"\n整体平均相似度: {overall_similarity:.3f}")
    
    if 'success' in similarity_df.columns:
        success_similarity = similarity_df[similarity_df['success'] == True]['similarity'].mean()
        failure_similarity = similarity_df[similarity_df['success'] == False]['similarity'].mean()
        print(f"成功游戏平均相似度: {success_similarity:.3f}")
        print(f"失败游戏平均相似度: {failure_similarity:.3f}")
        print(f"成功与失败相似度差异: {success_similarity - failure_similarity:+.3f}")
    
    print(f"\n✅ 分析完成！图表已保存到 figures/ 目录")

if __name__ == "__main__":
    main()
