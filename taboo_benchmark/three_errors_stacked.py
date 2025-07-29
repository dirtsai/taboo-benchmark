#!/usr/bin/env python3
"""
显示三种错误类型的堆叠图
"""

import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置图表样式
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

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

# 计算三种错误类型的绝对数量
error_counts = failed_games.groupby('hinter_model_clean').apply(
    lambda x: pd.Series({
        'Taboo_Violation': (x['has_taboo_violation'] == True).sum(),
        'Max_Turns_Exceeded': ((x['has_taboo_violation'] == False) & (x['turns_used'] >= 10)).sum(),
        'Other_Failure': ((x['has_taboo_violation'] == False) & (x['turns_used'] < 10)).sum()
    })
).fillna(0).astype(int)

print("三种错误类型统计:")
print(error_counts)
print(f"\n各模型总失败数:")
print(error_counts.sum(axis=1))

# 创建堆叠图
fig, ax = plt.subplots(figsize=(12, 8))

# 三种颜色分别代表三种错误
colors = ['#d62728', '#ff7f0e', '#9467bd']  # 红色违规，橙色超时，紫色其他

error_counts.plot(kind='bar', stacked=True, ax=ax, 
                 color=colors, edgecolor='white', linewidth=0.8)

ax.set_title('Error Type Breakdown by Model', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Model', fontsize=14)
ax.set_ylabel('Number of Errors', fontsize=14)
ax.legend(['Taboo Violation', 'Max Turns Exceeded', 'Other Failures'], 
          title='Error Type', fontsize=12, title_fontsize=12)
ax.grid(axis='y', alpha=0.3)
ax.tick_params(axis='x', rotation=45)

# 添加总错误数标签
totals = error_counts.sum(axis=1)
for i, (model, total) in enumerate(totals.items()):
    ax.text(i, total + 5, f'Total: {total}', ha='center', va='bottom', 
            fontweight='bold', fontsize=11)

# 添加各段的数值标签
for container in ax.containers:
    labels = [f'{int(v)}' if v > 10 else '' for v in container.datavalues]  # 只显示>10的标签
    ax.bar_label(container, labels=labels, label_type='center', 
                fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('comprehensive_figures/three_error_types_stacked.png', dpi=300, bbox_inches='tight')
plt.show()

# 详细分析
print("\n" + "="*50)
print("📊 三种错误类型详细分析")
print("="*50)

for model in error_counts.index:
    row = error_counts.loc[model]
    total_errors = row.sum()
    
    print(f"\n• {model} (总计 {total_errors} 个错误):")
    print(f"  - Taboo Violations: {row['Taboo_Violation']} ({row['Taboo_Violation']/total_errors*100:.1f}%)")
    print(f"  - Max Turns Exceeded: {row['Max_Turns_Exceeded']} ({row['Max_Turns_Exceeded']/total_errors*100:.1f}%)")
    print(f"  - Other Failures: {row['Other_Failure']} ({row['Other_Failure']/total_errors*100:.1f}%)")

print(f"\n✨ 图表已保存: comprehensive_figures/three_error_types_stacked.png") 