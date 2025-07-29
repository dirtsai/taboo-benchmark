import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pathlib

# ---------- 加载数据 ----------
# 加载中文实验结果（无字数提示）
results_path_no_hint = "results/chinese_full_experiment_20250717_222959/chinese_full_results_20250717_222959.csv"
no_hint_results = pd.read_csv(results_path_no_hint)

# 加载中文实验结果（有字数提示）
results_path_with_hint = "taboo_benchmark/results/chinese_merged_results_fixed_20250723_013406.csv"
with_hint_results = pd.read_csv(results_path_with_hint)

# 打印数据基本信息
print(f"无字数提示: {len(no_hint_results)} 条实验记录")
print(f"有字数提示: {len(with_hint_results)} 条实验记录")
print(f"目标词数量 (无提示): {no_hint_results['target_word'].nunique()}")
print(f"目标词数量 (有提示): {with_hint_results['target_word'].nunique()}")
print(f"Hinter 模型: {no_hint_results['hinter_model'].unique()}")
print(f"Guesser 模型: {no_hint_results['guesser_model'].unique()}")

# ---------- 成功率比较：整体成功率 ----------
# "全名 → 简写"映射
label_map = {
    "openai/gpt-4o":                  "gpt-4o",
    "google/gemini-2.5-flash":        "gemini-2.5-flash",
    "deepseek/deepseek-chat-v3-0324": "deepseek-v3",
    "moonshotai/kimi-k2":             "kimi-k2",
}

sns.set_theme(style="white", font_scale=1.2)
out_dir = pathlib.Path("figures")
out_dir.mkdir(exist_ok=True)

def plot_comparison_bar(df_no_hint, df_with_hint, group_key, fname, palette="magma"):
    """比较有无字数提示的成功率并绘制并排柱状图"""
    # 计算无提示成功率
    rate_no_hint = (
        df_no_hint.groupby(group_key)["success"]
          .mean()
          .sort_values(ascending=False)
    )
    
    # 计算有提示成功率
    rate_with_hint = (
        df_with_hint.groupby(group_key)["success"]
          .mean()
          .reindex(rate_no_hint.index)  # 保持相同顺序
    )
    
    # 替换索引为简写
    short_idx = rate_no_hint.index.map(lambda x: label_map.get(x, x))
    
    # 准备数据用于并排柱状图
    x = np.arange(len(short_idx))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    
    # 绘制并排柱状图
    bars1 = plt.bar(x - width/2, rate_no_hint.values, width, 
                   label='无字数提示', color='#440154', alpha=0.8)
    bars2 = plt.bar(x + width/2, rate_with_hint.values, width,
                   label='有字数提示', color='#fde725', alpha=0.8)
    
    plt.ylim(0, 1.05)
    plt.axhline(1, ls="--", c="gray", lw=1)
    plt.ylabel("Success Rate", fontsize=20)
    plt.xlabel(f"{group_key.split('_')[0].title()} Model", fontsize=20)
    plt.xticks(x, short_idx, rotation=0, fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=14)
    
    # 添加数值标签
    for i, (val1, val2) in enumerate(zip(rate_no_hint.values, rate_with_hint.values)):
        plt.text(i - width/2, val1 + 0.02, f"{val1:.2f}", ha="center", va="bottom",
                fontsize=11, weight="bold")
        plt.text(i + width/2, val2 + 0.02, f"{val2:.2f}", ha="center", va="bottom",
                fontsize=11, weight="bold")
    
    plt.tight_layout()
    plt.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
    plt.savefig(out_dir / f"{fname}.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    return rate_no_hint, rate_with_hint

# 1) 按 Hinter 模型比较
print("\n按 Hinter 模型的成功率比较:")
rate_hinter_no, rate_hinter_with = plot_comparison_bar(
    no_hint_results, with_hint_results,
    group_key="hinter_model",
    fname="Chinese_CharCount_Comparison_by_Hinter"
)

print("无字数提示:")
for model, rate in rate_hinter_no.items():
    print(f"  {label_map.get(model, model)}: {rate:.3f}")
print("有字数提示:")
for model, rate in rate_hinter_with.items():
    print(f"  {label_map.get(model, model)}: {rate:.3f}")

# 2) 按 Guesser 模型比较
print("\n按 Guesser 模型的成功率比较:")
rate_guesser_no, rate_guesser_with = plot_comparison_bar(
    no_hint_results, with_hint_results,
    group_key="guesser_model", 
    fname="Chinese_CharCount_Comparison_by_Guesser"
)

print("无字数提示:")
for model, rate in rate_guesser_no.items():
    print(f"  {label_map.get(model, model)}: {rate:.3f}")
print("有字数提示:")
for model, rate in rate_guesser_with.items():
    print(f"  {label_map.get(model, model)}: {rate:.3f}")

# 3) 整体成功率比较
overall_no_hint = no_hint_results['success'].mean()
overall_with_hint = with_hint_results['success'].mean()

print(f"\n整体成功率比较:")
print(f"无字数提示: {overall_no_hint:.3f}")
print(f"有字数提示: {overall_with_hint:.3f}")
print(f"提升幅度: {(overall_with_hint - overall_no_hint):.3f} ({((overall_with_hint - overall_no_hint)/overall_no_hint*100):.1f}%)")

# 4) 绘制整体成功率对比
plt.figure(figsize=(8, 6))
conditions = ['无字数提示', '有字数提示']
rates = [overall_no_hint, overall_with_hint]
colors = ['#440154', '#fde725']

bars = plt.bar(conditions, rates, color=colors, alpha=0.8)
plt.ylim(0, 1.05)
plt.axhline(1, ls="--", c="gray", lw=1)
plt.ylabel("Overall Success Rate", fontsize=20)
plt.xlabel("Condition", fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# 添加数值标签
for bar, rate in zip(bars, rates):
    plt.text(bar.get_x() + bar.get_width()/2, rate + 0.02, f"{rate:.3f}", 
            ha="center", va="bottom", fontsize=14, weight="bold")

plt.tight_layout()
plt.savefig(out_dir / "Chinese_CharCount_Overall_Comparison.pdf", bbox_inches="tight")
plt.savefig(out_dir / "Chinese_CharCount_Overall_Comparison.png", dpi=300, bbox_inches="tight")
plt.show()

print("→ 字数提示对比图表已保存至 figures/ 目录")
