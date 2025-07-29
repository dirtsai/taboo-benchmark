import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import Counter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('ggplot')

# 创建输出目录
output_dir = Path("presentation_figures")
output_dir.mkdir(exist_ok=True)

# 自定义颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

def load_data():
    """加载数据"""
    # 加载数据集
    with open('data/dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    dataset_df = pd.DataFrame(dataset)
    dataset_df['concreteness_score'] = dataset_df['metadata'].apply(lambda x: x.get('concreteness_score'))
    
    # 加载实验结果
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
    
    # 合并数据
    merged_df = results_df.merge(dataset_df, left_on='target_word', right_on='target', how='left')
    
    # 计算词频
    try:
        from wordfreq import word_frequency
        merged_df['word_frequency'] = merged_df['target_word'].apply(lambda x: word_frequency(x, 'en'))
        merged_df['frequency_log'] = np.log10(merged_df['word_frequency'].replace(0, 1e-10))
        
        # 创建词频类别
        merged_df['frequency_category'] = pd.cut(merged_df['frequency_log'], 
                                                bins=[-np.inf, -7, -6, -5, -4, np.inf],
                                                labels=['Very Rare', 'Rare', 'Uncommon', 'Common', 'Very Common'])
    except ImportError:
        print("Warning: wordfreq not available, using mock frequency data")
        merged_df['frequency_category'] = 'Common'
    
    return merged_df, dataset_df

def create_figure1_flowchart():
    """图形位置1: 实验设计流程图"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # 流程步骤
    steps = [
        "WordNet 3.1\nData Collection",
        "10,000 Cards\nGeneration",
        "4 LLM Models\nSelection",
        "4,800 Games\nExecution",
        "Multi-dimensional\nAnalysis"
    ]
    
    # 绘制流程图
    y_pos = 0.5
    step_width = 0.15
    start_x = 0.1
    
    for i, step in enumerate(steps):
        x_pos = start_x + i * (step_width + 0.05)
        
        # 绘制方框
        rect = plt.Rectangle((x_pos, y_pos-0.1), step_width, 0.2, 
                           facecolor=colors[i], alpha=0.7, edgecolor='black')
        ax.add_patch(rect)
        
        # 添加文字
        ax.text(x_pos + step_width/2, y_pos, step, ha='center', va='center', 
                fontsize=10, fontweight='bold', wrap=True)
        
        # 添加箭头
        if i < len(steps) - 1:
            ax.arrow(x_pos + step_width + 0.01, y_pos, 0.03, 0, 
                    head_width=0.02, head_length=0.01, fc='black', ec='black')
    
    # 添加关键信息
    ax.text(0.5, 0.8, 'Taboo Game Experiment Design Flow', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # 添加统计信息
    stats_text = "• 4 Models: GPT-4o, Claude Sonnet 4, Gemini 2.5 Pro, DeepSeek Chat V3\n"
    stats_text += "• 300 unique words across 5 domains\n"
    stats_text += "• 16 model pairs (4×4)\n"
    stats_text += "• Maximum 5 turns per game"
    
    ax.text(0.5, 0.2, stats_text, ha='center', va='center', 
            fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / "figure1_flowchart.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 1: Flowchart saved")

def create_figure2_dataset_distribution(dataset_df):
    """图形位置2: 数据集分布图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 词性分布饼图
    pos_counts = dataset_df['part_of_speech'].value_counts()
    ax1.pie(pos_counts.values, labels=pos_counts.index, autopct='%1.1f%%', 
            colors=colors[:len(pos_counts)], startangle=90)
    ax1.set_title('Part of Speech Distribution', fontsize=14, fontweight='bold')
    
    # 领域分布条形图
    category_counts = dataset_df['category'].value_counts()
    bars = ax2.bar(category_counts.index, category_counts.values, 
                   color=colors[:len(category_counts)])
    ax2.set_title('Domain Distribution', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Words')
    ax2.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 抽象程度分布
    concrete_data = dataset_df[dataset_df['concreteness_score'].notna()]
    if len(concrete_data) > 0:
        ax3.hist(concrete_data['concreteness_score'], bins=20, 
                color=colors[2], alpha=0.7, edgecolor='black')
        ax3.set_title('Concreteness Score Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Concreteness Score (1-5)')
        ax3.set_ylabel('Number of Words')
    else:
        ax3.text(0.5, 0.5, 'No concreteness data available', 
                ha='center', va='center', transform=ax3.transAxes)
    
    # 数据集概览
    ax4.axis('off')
    overview_text = f"""Dataset Overview:
    
    • Total Words: {len(dataset_df):,}
    • Domains: {len(dataset_df['category'].unique())}
    • Parts of Speech: {len(dataset_df['part_of_speech'].unique())}
    • Concreteness Scores: {dataset_df['concreteness_score'].notna().sum():,}
    • Source: WordNet 3.1
    """
    ax4.text(0.1, 0.9, overview_text, transform=ax4.transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_dir / "figure2_dataset_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 2: Dataset distribution saved")

def create_figure3_model_performance(merged_df):
    """图形位置3: 模型性能比较图"""
    # 计算模型性能
    model_success = merged_df.groupby('hinter_model_clean').agg({
        'success': ['count', 'sum', 'mean'],
        'turns_used': 'mean',
        'has_taboo_violation': 'mean'
    }).round(3)
    
    model_success.columns = ['Total Games', 'Successful Games', 'Success Rate', 'Average Turns', 'Violation Rate']
    model_success = model_success.sort_values('Success Rate', ascending=False)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 成功率对比
    bars1 = ax1.bar(model_success.index, model_success['Success Rate'], 
                    color=colors[:len(model_success)])
    ax1.set_title('Model Success Rate Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Success Rate')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 平均回合数对比
    bars2 = ax2.bar(model_success.index, model_success['Average Turns'], 
                    color=colors[:len(model_success)])
    ax2.set_title('Average Turns Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Turns')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 违规率对比
    bars3 = ax3.bar(model_success.index, model_success['Violation Rate'], 
                    color=colors[:len(model_success)])
    ax3.set_title('Violation Rate Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Violation Rate')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "figure3_model_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 3: Model performance comparison saved")

def create_figure4_cumulative_success(merged_df):
    """图形位置4: 累积成功率图"""
    successful_games = merged_df[merged_df['success'] == True]
    
    # 计算累积成功率
    cumulative_success = {}
    for turn in range(1, 6):
        cumulative_rates = successful_games.groupby('hinter_model_clean').apply(
            lambda x: (x['turns_used'] <= turn).sum() / len(x)
        )
        cumulative_success[f'Turn {turn}'] = cumulative_rates
    
    cumulative_df = pd.DataFrame(cumulative_success).fillna(0)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制累积成功率曲线
    for i, model in enumerate(cumulative_df.index):
        turns = range(1, 6)
        rates = [cumulative_df.loc[model, f'Turn {turn}'] for turn in turns]
        ax.plot(turns, rates, 'o-', linewidth=3, markersize=8, 
                label=model, color=colors[i])
    
    ax.set_title('Cumulative Success Rate by Turn Number', fontsize=16, fontweight='bold')
    ax.set_xlabel('Turn Number', fontsize=14)
    ax.set_ylabel('Cumulative Success Rate', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 6))
    ax.set_ylim(0, 1)
    
    # 添加数值标签
    for i, model in enumerate(cumulative_df.index):
        for turn in range(1, 6):
            rate = cumulative_df.loc[model, f'Turn {turn}']
            ax.text(turn, rate + 0.02, f'{rate:.3f}', 
                   ha='center', va='bottom', fontsize=9, color=colors[i])
    
    plt.tight_layout()
    plt.savefig(output_dir / "figure4_cumulative_success.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 4: Cumulative success rate saved")

def create_figure5_frequency_effect(merged_df):
    """图形位置5: 词频效应图"""
    # 按词频类别分析
    if 'frequency_category' in merged_df.columns:
        frequency_analysis = merged_df.groupby('frequency_category').agg({
            'success': ['count', 'mean'],
            'turns_used': 'mean'
        }).round(3)
        
        frequency_analysis.columns = ['Total Games', 'Success Rate', 'Average Turns']
        
        # 重新排序
        freq_order = ['Very Common', 'Common', 'Uncommon', 'Rare', 'Very Rare']
        frequency_analysis = frequency_analysis.reindex([f for f in freq_order if f in frequency_analysis.index])
        
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # 主Y轴：成功率
        color1 = colors[0]
        ax1.set_xlabel('Word Frequency Category', fontsize=14)
        ax1.set_ylabel('Success Rate', color=color1, fontsize=14)
        bars1 = ax1.bar(frequency_analysis.index, frequency_analysis['Success Rate'], 
                        color=color1, alpha=0.7, label='Success Rate')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim(0, 1)
        
        # 添加成功率标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 次Y轴：平均回合数
        ax2 = ax1.twinx()
        color2 = colors[1]
        ax2.set_ylabel('Average Turns', color=color2, fontsize=14)
        line2 = ax2.plot(frequency_analysis.index, frequency_analysis['Average Turns'], 
                        color=color2, marker='o', linewidth=3, markersize=8, label='Average Turns')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # 添加平均回合数标签
        for i, val in enumerate(frequency_analysis['Average Turns']):
            ax2.text(i, val + 0.05, f'{val:.1f}', 
                    ha='center', va='bottom', fontweight='bold', color=color2)
        
        ax1.set_title('Word Frequency Effect on Performance', fontsize=16, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加相关性信息
        if 'frequency_log' in merged_df.columns:
            correlation = merged_df[['frequency_log', 'success']].corr().iloc[0, 1]
            ax1.text(0.7, 0.9, f'Correlation: r = {correlation:.3f}', 
                    transform=ax1.transAxes, fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        
        plt.tight_layout()
        plt.savefig(output_dir / "figure5_frequency_effect.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Figure 5: Frequency effect saved")

def create_figure6_domain_performance(merged_df):
    """图形位置6: 领域性能图"""
    # 按领域分析
    domain_analysis = merged_df.groupby('category').agg({
        'success': ['count', 'mean'],
        'turns_used': 'mean'
    }).round(3)
    
    domain_analysis.columns = ['Total Games', 'Success Rate', 'Average Turns']
    domain_analysis = domain_analysis.sort_values('Success Rate', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 水平条形图
    bars = ax.barh(domain_analysis.index, domain_analysis['Success Rate'], 
                   color=colors[:len(domain_analysis)])
    
    ax.set_title('Performance by Domain', fontsize=16, fontweight='bold')
    ax.set_xlabel('Success Rate', fontsize=14)
    ax.set_ylabel('Domain', fontsize=14)
    ax.set_xlim(0, 1)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                f'{width:.3f}', ha='left', va='center', fontweight='bold')
        
        # 添加游戏数量
        games = domain_analysis.iloc[i]['Total Games']
        ax.text(0.02, bar.get_y() + bar.get_height()/2.,
                f'n={games}', ha='left', va='center', fontsize=10, color='white')
    
    plt.tight_layout()
    plt.savefig(output_dir / "figure6_domain_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 6: Domain performance saved")

def main():
    """主函数"""
    print("Generating presentation figures...")
    print("=" * 50)
    
    # 加载数据
    merged_df, dataset_df = load_data()
    print(f"Data loaded: {len(merged_df)} games, {len(dataset_df)} words")
    
    # 生成图表
    create_figure1_flowchart()
    create_figure2_dataset_distribution(dataset_df)
    create_figure3_model_performance(merged_df)
    create_figure4_cumulative_success(merged_df)
    create_figure5_frequency_effect(merged_df)
    create_figure6_domain_performance(merged_df)
    
    print("\n" + "=" * 50)
    print("All figures saved to:", output_dir.absolute())
    print("Files generated:")
    for file in output_dir.glob("*.png"):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main() 