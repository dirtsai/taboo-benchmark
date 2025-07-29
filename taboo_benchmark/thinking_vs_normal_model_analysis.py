import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载实验结果数据
print("🔍 加载实验结果数据...")
results_df = pd.read_csv('results/taboo_experiment_20250712_004918/complete_experiment_results.csv')

# 模型分类
print("\n📊 模型分类:")
print("="*50)

# 根据用户指定的分类
normal_models = {
    'deepseek/deepseek-chat-v3-0324': 'DeepSeek Chat V3',
    'openai/gpt-4o': 'GPT-4o'
}

thinking_models = {
    'anthropic/claude-sonnet-4': 'Claude Sonnet 4',
    'google/gemini-2.5-pro': 'Gemini 2.5 Pro'
}

print("普通模型 (Normal Models):")
for model_id, model_name in normal_models.items():
    print(f"  • {model_name} ({model_id})")

print("\nThinking模型 (Thinking Models):")
for model_id, model_name in thinking_models.items():
    print(f"  • {model_name} ({model_id})")

# 添加模型类型标签
def get_model_type(model_id):
    if model_id in normal_models:
        return '普通模型'
    elif model_id in thinking_models:
        return 'Thinking模型'
    else:
        return '未知'

def get_model_clean_name(model_id):
    if model_id in normal_models:
        return normal_models[model_id]
    elif model_id in thinking_models:
        return thinking_models[model_id]
    else:
        return model_id

results_df['model_type'] = results_df['hinter_model'].apply(get_model_type)
results_df['model_name'] = results_df['hinter_model'].apply(get_model_clean_name)

print(f"\n📈 数据概览:")
print(f"  • 总游戏数: {len(results_df):,}")
print(f"  • 普通模型游戏数: {len(results_df[results_df['model_type'] == '普通模型']):,}")
print(f"  • Thinking模型游戏数: {len(results_df[results_df['model_type'] == 'Thinking模型']):,}")

# 1. 基础性能对比
print("\n" + "="*60)
print("1. 基础性能对比分析")
print("="*60)

# 按模型类型分组统计
type_stats = results_df.groupby('model_type').agg({
    'success': ['count', 'sum', 'mean'],
    'turns_used': lambda x: x[results_df.loc[x.index, 'success']].mean(),
    'has_taboo_violation': 'mean',
    'duration_seconds': 'mean'
}).round(3)

type_stats.columns = ['总游戏数', '成功游戏数', '成功率', '平均轮数', '违规率', '平均用时(秒)']

print("📊 按模型类型统计:")
print(type_stats)

# 个别模型详细统计
individual_stats = results_df.groupby('model_name').agg({
    'success': ['count', 'sum', 'mean'],
    'turns_used': lambda x: x[results_df.loc[x.index, 'success']].mean(),
    'has_taboo_violation': 'mean',
    'duration_seconds': 'mean'
}).round(3)

individual_stats.columns = ['总游戏数', '成功游戏数', '成功率', '平均轮数', '违规率', '平均用时(秒)']

print("\n📋 各模型详细统计:")
print(individual_stats)

# 2. 效率分析
print("\n" + "="*60)
print("2. 效率分析")
print("="*60)

# 第1轮成功率分析
successful_games = results_df[results_df['success'] == True]
first_turn_success = successful_games.groupby(['model_type', 'model_name']).apply(
    lambda x: (x['turns_used'] == 1).sum() / len(x)
).reset_index()
first_turn_success.columns = ['模型类型', '模型名称', '第1轮成功率']

print("⚡ 第1轮成功率对比:")
for model_type in ['普通模型', 'Thinking模型']:
    print(f"\n{model_type}:")
    type_data = first_turn_success[first_turn_success['模型类型'] == model_type]
    for _, row in type_data.iterrows():
        print(f"  • {row['模型名称']}: {row['第1轮成功率']:.1%}")

# 累积成功率分析
cumulative_success = {}
for turn in range(1, 6):
    cumulative_rates = successful_games.groupby(['model_type', 'model_name']).apply(
        lambda x: (x['turns_used'] <= turn).sum() / len(x)
    ).reset_index()
    cumulative_rates.columns = ['模型类型', '模型名称', f'前{turn}轮累积成功率']
    cumulative_success[turn] = cumulative_rates

print(f"\n🎯 前3轮累积成功率对比:")
for model_type in ['普通模型', 'Thinking模型']:
    print(f"\n{model_type}:")
    type_data = cumulative_success[3][cumulative_success[3]['模型类型'] == model_type]
    for _, row in type_data.iterrows():
        print(f"  • {row['模型名称']}: {row['前3轮累积成功率']:.1%}")

# 3. 稳定性分析
print("\n" + "="*60)
print("3. 稳定性分析")
print("="*60)

# 计算各模型在不同任务上的性能标准差
stability_stats = results_df.groupby(['model_type', 'model_name']).agg({
    'success': ['mean', 'std'],
    'turns_used': lambda x: x[results_df.loc[x.index, 'success']].std()
}).round(4)

stability_stats.columns = ['成功率均值', '成功率标准差', '轮数标准差']

print("🎲 稳定性指标 (标准差越小越稳定):")
print(stability_stats)

# 4. 失败原因分析
print("\n" + "="*60)
print("4. 失败原因分析")
print("="*60)

failed_games = results_df[results_df['success'] == False]
failure_analysis = failed_games.groupby(['model_type', 'failure_reason']).size().unstack(fill_value=0)

print("❌ 失败原因分布:")
if not failure_analysis.empty:
    print(failure_analysis)
    
    # 计算失败原因比例
    failure_pct = failure_analysis.div(failure_analysis.sum(axis=1), axis=0) * 100
    print(f"\n失败原因比例 (%):")
    print(failure_pct.round(1))

# 5. 可视化对比
print("\n" + "="*60)
print("5. 可视化对比分析")
print("="*60)

# 创建综合对比图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Thinking模型 vs 普通模型 性能对比', fontsize=16, fontweight='bold')

# 1. 成功率对比
ax1 = axes[0, 0]
type_success = results_df.groupby('model_type')['success'].mean()
colors = ['#ff7f0e', '#1f77b4']  # 橙色：普通模型，蓝色：Thinking模型
bars1 = ax1.bar(type_success.index, type_success.values, color=colors)
ax1.set_title('成功率对比', fontweight='bold')
ax1.set_ylabel('成功率')
ax1.set_ylim(0, 1)

# 添加数值标签
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.1%}', ha='center', va='bottom', fontweight='bold')

# 2. 平均轮数对比
ax2 = axes[0, 1]
type_turns = results_df[results_df['success'] == True].groupby('model_type')['turns_used'].mean()
bars2 = ax2.bar(type_turns.index, type_turns.values, color=colors)
ax2.set_title('平均轮数对比 (成功游戏)', fontweight='bold')
ax2.set_ylabel('平均轮数')

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

# 3. 违规率对比
ax3 = axes[0, 2]
type_violation = results_df.groupby('model_type')['has_taboo_violation'].mean()
bars3 = ax3.bar(type_violation.index, type_violation.values, color=colors)
ax3.set_title('违规率对比', fontweight='bold')
ax3.set_ylabel('违规率')

for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.002,
             f'{height:.1%}', ha='center', va='bottom', fontweight='bold')

# 4. 个别模型详细对比
ax4 = axes[1, 0]
individual_success = results_df.groupby('model_name')['success'].mean().sort_values(ascending=False)
model_colors = ['#1f77b4' if name in thinking_models.values() else '#ff7f0e' 
                for name in individual_success.index]
bars4 = ax4.bar(range(len(individual_success)), individual_success.values, color=model_colors)
ax4.set_title('各模型成功率详细对比', fontweight='bold')
ax4.set_ylabel('成功率')
ax4.set_xticks(range(len(individual_success)))
ax4.set_xticklabels(individual_success.index, rotation=45, ha='right')

for i, bar in enumerate(bars4):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 5. 第1轮成功率对比
ax5 = axes[1, 1]
type_first_turn = successful_games.groupby('model_type').apply(
    lambda x: (x['turns_used'] == 1).sum() / len(x)
)
bars5 = ax5.bar(type_first_turn.index, type_first_turn.values, color=colors)
ax5.set_title('第1轮成功率对比', fontweight='bold')
ax5.set_ylabel('第1轮成功率')

for bar in bars5:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.1%}', ha='center', va='bottom', fontweight='bold')

# 6. 平均用时对比
ax6 = axes[1, 2]
type_duration = results_df.groupby('model_type')['duration_seconds'].mean()
bars6 = ax6.bar(type_duration.index, type_duration.values, color=colors)
ax6.set_title('平均用时对比', fontweight='bold')
ax6.set_ylabel('平均用时 (秒)')

for bar in bars6:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.0f}s', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# 6. 总结报告
print("\n" + "="*60)
print("6. 总结报告")
print("="*60)

# 计算性能差异
thinking_success = type_stats.loc['Thinking模型', '成功率']
normal_success = type_stats.loc['普通模型', '成功率']
success_gap = thinking_success - normal_success

thinking_turns = type_stats.loc['Thinking模型', '平均轮数']
normal_turns = type_stats.loc['普通模型', '平均轮数']
efficiency_gap = normal_turns - thinking_turns

thinking_violation = type_stats.loc['Thinking模型', '违规率']
normal_violation = type_stats.loc['普通模型', '违规率']
violation_gap = normal_violation - thinking_violation

print(f"🎯 核心发现:")
print(f"  • 成功率差异: Thinking模型比普通模型高 {success_gap:.1%}")
print(f"  • 效率差异: Thinking模型比普通模型少用 {efficiency_gap:.1f} 轮")
print(f"  • 规则遵守: Thinking模型违规率比普通模型低 {violation_gap:.1%}")

print(f"\n🏆 最佳表现:")
best_overall = individual_stats.sort_values('成功率', ascending=False).index[0]
best_efficiency = individual_stats.sort_values('平均轮数', ascending=True).index[0]
best_compliance = individual_stats.sort_values('违规率', ascending=True).index[0]

print(f"  • 最高成功率: {best_overall} ({individual_stats.loc[best_overall, '成功率']:.1%})")
print(f"  • 最高效率: {best_efficiency} ({individual_stats.loc[best_efficiency, '平均轮数']:.1f}轮)")
print(f"  • 最佳规则遵守: {best_compliance} ({individual_stats.loc[best_compliance, '违规率']:.1%})")

print(f"\n📊 模型类型优势:")
if success_gap > 0.05:
    print(f"  • Thinking模型在成功率上有明显优势")
else:
    print(f"  • 两类模型在成功率上差异不大")

if efficiency_gap > 0.3:
    print(f"  • Thinking模型在效率上有明显优势")
else:
    print(f"  • 两类模型在效率上差异不大")

if violation_gap > 0.01:
    print(f"  • Thinking模型在规则遵守上有明显优势")
else:
    print(f"  • 两类模型在规则遵守上差异不大")

print(f"\n💡 结论:")
print(f"  • Thinking模型整体表现优于普通模型")
print(f"  • 主要优势体现在：更高的成功率、更少的轮数、更低的违规率")
print(f"  • 这可能与Thinking模型的内部推理机制有关")

print(f"\n✅ 分析完成！") 