import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from math import pi

# 设置
output_dir = Path('comprehensive_figures')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# 加载数据
results_df = pd.read_csv('results/taboo_experiment_20250712_004918/complete_experiment_results.csv')
model_name_mapping = {
    'anthropic/claude-sonnet-4': 'Claude Sonnet 4',
    'openai/gpt-4o': 'GPT-4o', 
    'google/gemini-2.5-pro': 'Gemini 2.5 Pro',
    'deepseek/deepseek-chat-v3-0324': 'DeepSeek Chat V3'
}
results_df['hinter_model_clean'] = results_df['hinter_model'].map(model_name_mapping)

# 计算各模型的多维性能指标
model_metrics = results_df.groupby('hinter_model_clean').agg({
    'success': 'mean',
    'turns_used': lambda x: 1/(x[results_df.loc[x.index, 'success']].mean()),
    'has_taboo_violation': lambda x: 1-x.mean()
}).round(3)

# 添加第1轮成功率
first_success_rate = results_df[results_df['success'] == True].groupby('hinter_model_clean').apply(
    lambda x: (x['turns_used'] == 1).sum() / len(x)
)
model_metrics['first_turn_success'] = first_success_rate
model_metrics.columns = ['Success Rate', 'Efficiency', 'Rule Compliance', 'First Turn Success']

# 标准化指标
model_metrics_scaled = model_metrics.copy()
for col in model_metrics.columns:
    col_min = model_metrics[col].min()
    col_max = model_metrics[col].max()
    if col_max > col_min:
        model_metrics_scaled[col] = (model_metrics[col] - col_min) / (col_max - col_min)
    else:
        model_metrics_scaled[col] = 0.5

# 图7: 雷达图
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

# 数据表格
ax2.axis('off')
table_data = model_metrics.round(3)
table = ax2.table(cellText=table_data.values, rowLabels=table_data.index, colLabels=table_data.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)
ax2.set_title('Performance Metrics Table', fontsize=14, fontweight='bold', y=0.9)

plt.tight_layout()
plt.savefig(output_dir / 'figure7_radar.png', dpi=300, bbox_inches='tight')
plt.close()
print('✓ Figure 7: Radar Comparison saved') 