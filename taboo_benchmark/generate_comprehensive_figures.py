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
output_dir = Path("comprehensive_figures")
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

def create_figure1_overview(merged_df):
    """图1: 实验概览 - 基础性能指标"""
    # 计算模型性能
    model_success = merged_df.groupby('hinter_model_clean').agg({
        'success': ['count', 'sum', 'mean'],
        'turns_used': 'mean',
        'has_taboo_violation': 'mean'
    }).round(3)
    
    model_success.columns = ['Total Games', 'Successful Games', 'Success Rate', 'Average Turns', 'Violation Rate']
    model_success = model_success.sort_values('Success Rate', ascending=False)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 子图1: 成功率对比
    bars1 = ax1.bar(model_success.index, model_success['Success Rate'], 
                    color=colors[:len(model_success)])
    ax1.set_title('Model Success Rate Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Success Rate')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 子图2: 平均回合数对比
    bars2 = ax2.bar(model_success.index, model_success['Average Turns'], 
                    color=colors[:len(model_success)])
    ax2.set_title('Average Turns Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Turns')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 子图3: 违规率对比
    bars3 = ax3.bar(model_success.index, model_success['Violation Rate'], 
                    color=colors[:len(model_success)])
    ax3.set_title('Violation Rate Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Violation Rate')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 子图4: 实验规模信息
    ax4.axis('off')
    stats_text = f"""Experiment Overview:
    
    • Total Games: {len(merged_df):,}
    • Models Tested: {len(merged_df['hinter_model_clean'].unique())}
    • Unique Words: {merged_df['target_word'].nunique()}
    • Overall Success Rate: {merged_df['success'].mean():.1%}
    • Average Game Length: {merged_df[merged_df['success']]['turns_used'].mean():.1f} turns
    • Total Violations: {merged_df['has_taboo_violation'].sum()}
    """
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_dir / "figure1_overview.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 1: Experiment Overview saved")

def create_figure2_cumulative_efficiency(merged_df):
    """图2: 效率分析 - 累积成功率"""
    successful_games = merged_df[merged_df['success'] == True]
    
    # 计算累积成功率
    cumulative_success = {}
    for turn in range(1, 6):
        cumulative_rates = successful_games.groupby('hinter_model_clean').apply(
            lambda x: (x['turns_used'] <= turn).sum() / len(x)
        )
        cumulative_success[f'Turn {turn}'] = cumulative_rates
    
    cumulative_df = pd.DataFrame(cumulative_success).fillna(0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 累积成功率曲线
    for i, model in enumerate(cumulative_df.index):
        turns = range(1, 6)
        rates = [cumulative_df.loc[model, f'Turn {turn}'] for turn in turns]
        ax1.plot(turns, rates, 'o-', linewidth=3, markersize=8, 
                label=model, color=colors[i])
    
    ax1.set_title('Cumulative Success Rate by Turn Number', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Turn Number')
    ax1.set_ylabel('Cumulative Success Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, 6))
    ax1.set_ylim(0, 1)
    
    # 第1轮成功率对比
    first_turn_rates = {model: cumulative_df.loc[model, 'Turn 1'] for model in cumulative_df.index}
    first_turn_df = pd.Series(first_turn_rates).sort_values(ascending=False)
    
    bars = ax2.bar(first_turn_df.index, first_turn_df.values, 
                   color=colors[:len(first_turn_df)])
    ax2.set_title('First Turn Success Rate', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Success Rate')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "figure2_efficiency.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 2: Efficiency Analysis saved")

def create_figure3_linguistic_factors(merged_df):
    """图3: 语言学因素分析 - 词性、抽象程度、词义数量"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 子图1: 词性分析
    if 'part_of_speech' in merged_df.columns:
        pos_success = merged_df.groupby('part_of_speech')['success'].mean().sort_values(ascending=False)
        bars1 = ax1.bar(pos_success.index, pos_success.values, color=colors[:len(pos_success)])
        ax1.set_title('Success Rate by Part of Speech', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Success Rate')
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 子图2: 抽象程度分析
    concrete_data = merged_df[merged_df['concreteness_score'].notna()].copy()
    if len(concrete_data) > 0:
        concrete_data['concreteness_level'] = pd.cut(concrete_data['concreteness_score'], 
                                                   bins=[0, 2, 3, 4, 5], 
                                                   labels=['High Abstract', 'Mid Abstract', 'Mid Concrete', 'High Concrete'])
        
        concrete_success = concrete_data.groupby('concreteness_level')['success'].mean().sort_values(ascending=False)
        bars2 = ax2.bar(concrete_success.index, concrete_success.values, color=colors[:len(concrete_success)])
        ax2.set_title('Success Rate by Concreteness Level', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Success Rate')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 子图3: 词义数量分析
    if 'senses' in merged_df.columns:
        merged_df['sense_count'] = merged_df['senses'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        if merged_df['sense_count'].max() > 0:
            # 将词义数量分组避免过于分散
            merged_df['sense_group'] = pd.cut(merged_df['sense_count'], 
                                            bins=[0, 2, 5, 10, np.inf], 
                                            labels=['1-2', '3-5', '6-10', '10+'])
            
            sense_success = merged_df.groupby('sense_group')['success'].mean()
            bars3 = ax3.bar(sense_success.index, sense_success.values, color=colors[:len(sense_success)])
            ax3.set_title('Success Rate by Number of Senses', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Success Rate')
            
            for bar in bars3:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 子图4: 语言学因素总结
    ax4.axis('off')
    summary_text = """Linguistic Factors Summary:
    
    • Part of Speech Effect:
      - Nouns generally easier to guess
      - Adjectives more challenging
    
    • Concreteness Effect:
      - Mid-concrete words optimal
      - Abstract concepts challenging
    
    • Polysemy Effect:
      - Multiple senses add complexity
      - Context disambiguation important
    """
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_dir / "figure3_linguistic.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 3: Linguistic Factors saved")

def create_figure4_frequency_analysis(merged_df):
    """图4: 词频分析"""
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
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 主图：成功率vs词频（双Y轴）
        color1 = colors[0]
        ax1.set_xlabel('Word Frequency Category')
        ax1.set_ylabel('Success Rate', color=color1)
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
        ax1_twin = ax1.twinx()
        color2 = colors[1]
        ax1_twin.set_ylabel('Average Turns', color=color2)
        line2 = ax1_twin.plot(frequency_analysis.index, frequency_analysis['Average Turns'], 
                        color=color2, marker='o', linewidth=3, markersize=8, label='Average Turns')
        ax1_twin.tick_params(axis='y', labelcolor=color2)
        
        ax1.set_title('Word Frequency Effect on Performance', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # 游戏数量分布
        bars2 = ax2.bar(frequency_analysis.index, frequency_analysis['Total Games'], 
                        color=colors[2], alpha=0.7)
        ax2.set_title('Game Distribution by Frequency Category', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Games')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 20,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / "figure4_frequency.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Figure 4: Frequency Analysis saved")

def create_figure5_domain_analysis(merged_df):
    """图5: 领域分析"""
    domain_analysis = merged_df.groupby('category').agg({
        'success': ['count', 'mean'],
        'turns_used': 'mean'
    }).round(3)
    
    domain_analysis.columns = ['Total Games', 'Success Rate', 'Average Turns']
    domain_analysis = domain_analysis.sort_values('Success Rate', ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 领域成功率对比
    bars1 = ax1.bar(domain_analysis.index, domain_analysis['Success Rate'], 
                   color=colors[:len(domain_analysis)])
    ax1.set_title('Performance by Domain', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Success Rate')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylim(0, 1)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 各模型在不同领域的表现
    domain_model_pivot = merged_df.groupby(['category', 'hinter_model_clean'])['success'].mean().unstack()
    domain_model_pivot.plot(kind='bar', ax=ax2, color=colors[:len(domain_model_pivot.columns)])
    ax2.set_title('Model Performance Across Domains', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Success Rate')
    ax2.set_xlabel('Domain')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / "figure5_domain.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 5: Domain Analysis saved")

def create_figure6_error_analysis(merged_df):
    """图6: 错误分析 - 使用堆叠图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 失败原因按模型的堆叠柱状图 (绝对数量)
    failed_games = merged_df[merged_df['success'] == False]
    
    if len(failed_games) > 0:
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
        ax1.set_title('Failure Reasons by Model (Absolute)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Number of Failed Games')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(title='Failure Reason', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        # 添加总失败数标签
        for i, (model, row) in enumerate(failure_stack_data.iterrows()):
            total_failures = row.sum()
            ax1.text(i, total_failures + 2, f'{total_failures}', ha='center', va='bottom', fontweight='bold')
    
    # 2. 失败原因按模型的堆叠柱状图 (百分比)
    if len(failed_games) > 0:
        failure_pct_data = failure_stack_data.div(failure_stack_data.sum(axis=1), axis=0) * 100
        failure_pct_data.plot(kind='bar', stacked=True, ax=ax2, color=chart_colors, width=0.7)
        ax2.set_title('Failure Reasons by Model (Percentage)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Percentage of Failed Games')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend(title='Failure Reason', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax2.set_ylim(0, 100)
        
        # 添加百分比标签
        for i, (model, row) in enumerate(failure_pct_data.iterrows()):
            cumulative = 0
            for j, (reason, pct) in enumerate(row.items()):
                if pct > 10:  # 只显示超过10%的标签
                    ax2.text(i, cumulative + pct/2, f'{pct:.0f}%', ha='center', va='center', 
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
                    ha='center', va='center', fontweight='bold', color='white', fontsize=9)
    
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
            ax4.legend(title='Violation Turn', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            
            # 添加总违规数标签
            for i, (model, row) in enumerate(violation_turn_data.iterrows()):
                total_violations = row.sum()
                if total_violations > 0:
                    ax4.text(i, total_violations + 0.5, f'{total_violations}', 
                            ha='center', va='bottom', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No Taboo Violations Found', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=14)
            ax4.set_title('Taboo Violations by Turn and Model', fontsize=14, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No Taboo Violations Found', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Taboo Violations by Turn and Model', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "figure6_error_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 6: Error Analysis saved")

def create_figure7_radar_comparison(merged_df):
    """图7: 综合性能雷达图"""
    from math import pi
    
    # 计算各模型的多维性能指标
    model_metrics = merged_df.groupby('hinter_model_clean').agg({
        'success': 'mean',
        'turns_used': lambda x: 1/(x[merged_df.loc[x.index, 'success']].mean()),  # 效率指标
        'has_taboo_violation': lambda x: 1-x.mean()  # 规则遵守指标
    }).round(3)
    
    # 添加第1轮成功率
    first_success_rate = merged_df[merged_df['success'] == True].groupby('hinter_model_clean').apply(
        lambda x: (x['turns_used'] == 1).sum() / len(x)
    )
    model_metrics['first_turn_success'] = first_success_rate
    
    model_metrics.columns = ['Success Rate', 'Efficiency', 'Rule Compliance', 'First Turn Success']
    
    # 标准化指标到0-1范围
    model_metrics_scaled = model_metrics.copy()
    for col in model_metrics.columns:
        col_min = model_metrics[col].min()
        col_max = model_metrics[col].max()
        if col_max > col_min:
            model_metrics_scaled[col] = (model_metrics[col] - col_min) / (col_max - col_min)
        else:
            model_metrics_scaled[col] = 0.5
    
    # 绘制雷达图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 雷达图
    ax1 = plt.subplot(121, projection='polar')
    angles = [n / float(len(model_metrics.columns)) * 2 * pi for n in range(len(model_metrics.columns))]
    angles += angles[:1]
    
    for i, (model, values) in enumerate(model_metrics_scaled.iterrows()):
        values_list = values.tolist()
        values_list += values_list[:1]
        
        ax1.plot(angles, values_list, 'o-', linewidth=2, label=model, color=colors[i])
        ax1.fill(angles, values_list, alpha=0.25, color=colors[i])
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(model_metrics.columns)
    ax1.set_ylim(0, 1)
    ax1.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax1.grid(True)
    
    # 原始数据表格
    ax2.axis('off')
    table_data = model_metrics.round(3)
    table = ax2.table(cellText=table_data.values,
                     rowLabels=table_data.index,
                     colLabels=table_data.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax2.set_title('Performance Metrics Table', fontsize=14, fontweight='bold', y=0.9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "figure7_radar.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 7: Radar Comparison saved")

def main():
    """主函数"""
    print("Generating comprehensive presentation figures...")
    print("=" * 60)
    
    # 加载数据
    merged_df, dataset_df = load_data()
    print(f"Data loaded: {len(merged_df)} games, {len(dataset_df)} words")
    
    # 生成所有图表
    create_figure1_overview(merged_df)
    create_figure2_cumulative_efficiency(merged_df)
    create_figure3_linguistic_factors(merged_df)
    create_figure4_frequency_analysis(merged_df)
    create_figure5_domain_analysis(merged_df)
    create_figure6_error_analysis(merged_df)
    create_figure7_radar_comparison(merged_df)
    
    print("\n" + "=" * 60)
    print("All comprehensive figures saved to:", output_dir.absolute())
    print("Files generated:")
    for file in output_dir.glob("*.png"):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main() 