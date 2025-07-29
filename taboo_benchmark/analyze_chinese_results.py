import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
from datetime import datetime

# Set global style
sns.set_theme(style="white", font_scale=1.2)

# Load Chinese experiment results data
results_path = "results/chinese_full_experiment_20250717_222959/chinese_full_results_20250717_222959.csv"
complete_experiment_results = pd.read_csv(results_path)

# Print basic information about the data
print(f"Loaded {len(complete_experiment_results)} experiment records")
print(f"Number of target words: {complete_experiment_results['target_word'].nunique()}")
print(f"Hinter models: {complete_experiment_results['hinter_model'].unique()}")
print(f"Guesser models: {complete_experiment_results['guesser_model'].unique()}")

# Model name mapping
label_map = {
    "openai/gpt-4o": "gpt-4o",
    "google/gemini-2.5-flash": "gemini-2.5-flash",
    "deepseek/deepseek-chat-v3-0324": "deepseek-v3",
    "moonshotai/kimi-k2": "kimi-k2"
}

# Create output directory
out_dir = pathlib.Path("figures")
out_dir.mkdir(exist_ok=True)

# Function to plot success rate bar chart
def plot_success_bar(df, group_key, fname, palette="magma"):
    """Plot success rate bar chart grouped by group_key and save to file"""
    rate = (
        df.groupby(group_key)["success"]
          .mean()
          .sort_values(ascending=False)
    )

    # Replace index with short model names
    short_idx = rate.index.map(lambda x: label_map.get(x, x))

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=short_idx, y=rate.values,
        palette=sns.color_palette(palette, n_colors=len(rate))
    )
    plt.ylim(0, 1.05)                        # Leave 0.05 space above 1.0
    plt.axhline(1, ls="--", c="gray", lw=1)  # Use gray dashed line for 1.0 mark
    plt.ylabel("Success Rate", fontsize=20)
    plt.xlabel(f"{group_key.split('_')[0].title()} Model", fontsize=20)
    # Uniform ticks: horizontal, font size 18
    plt.xticks(rotation=0, fontsize=18)
    plt.yticks(fontsize=18)
    for idx, val in enumerate(rate.values):
        plt.text(idx, val + 0.02, f"{val:.2f}", ha="center", va="bottom",
                 fontsize=12, weight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
    plt.savefig(out_dir / f"{fname}.png", dpi=300, bbox_inches="tight")
    plt.show()
    return rate

# Plot success rate bar chart by Hinter model
print("\nSuccess rate by Hinter model:")
rate_hinter = plot_success_bar(
    complete_experiment_results,
    group_key="hinter_model",
    fname="Chinese-40_SuccessRate_by_Hinter"
)

# Plot success rate bar chart by Guesser model
print("\nSuccess rate by Guesser model:")
rate_guesser = plot_success_bar(
    complete_experiment_results,
    group_key="guesser_model",
    fname="Chinese-40_SuccessRate_by_Guesser"
)

print("→ Two bar charts saved to figures/ directory")

# =============================================
# Figure 1: Hinter model success rate heatmap
# =============================================
def generate_hinter_success_heatmap():
    # Calculate success rate grouped by hinter and guesser
    pivot = (
        complete_experiment_results
        .groupby(["guesser_model", "hinter_model"])["success"]
        .mean()
        .unstack()  # Rows=Guesser, Columns=Hinter
    )
    
    # Reorder columns in reverse
    pivot_sorted = pivot[pivot.columns[::-1]]
    pivot_display = pivot_sorted.rename(index=label_map, columns=label_map)
    
    plt.figure(figsize=(12, 9))
    sns.heatmap(
        pivot_display,
        annot=True, fmt=".3f",
        cmap=sns.color_palette("magma", as_cmap=True),
        vmin=0, vmax=1,
        linewidths=.6, linecolor="white",
        annot_kws={"size": 20, "weight": "bold"}
    )
    
    plt.title("Figure 1: Success Rate of Different Models as Hinters in Chinese Taboo Experiment", fontsize=22, weight="bold", pad=16)
    plt.xlabel("Hinter Model", fontsize=20)
    plt.ylabel("Guesser Model", fontsize=20)
    plt.xticks(rotation=0, fontsize=18)
    plt.yticks(rotation=0, fontsize=18)
    plt.tight_layout()
    
    # Save images
    fname = "Chinese_Guesser-Hinter_SuccessRate"
    plt.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
    plt.savefig(out_dir / f"{fname}.png", dpi=300, bbox_inches="tight")
    plt.show()

# =============================================
# Figure 2: Failure reason analysis stacked chart
# =============================================
def generate_failure_reason_stacked_chart():
    # Filter failed cases
    failed_cases = complete_experiment_results[complete_experiment_results['success'] == False]
    
    # Count failure reasons by model
    failure_counts = failed_cases.groupby(['hinter_model', 'failure_reason']).size().unstack(fill_value=0)
    
    # Calculate percentage of each failure reason
    failure_percentages = failure_counts.div(failure_counts.sum(axis=1), axis=0)
    
    # Rename model names
    failure_percentages.index = [label_map[model] for model in failure_percentages.index]
    
    # Draw stacked bar chart
    plt.figure(figsize=(14, 8))
    failure_percentages.plot(
        kind='bar', 
        stacked=True,
        colormap='magma',
        figsize=(14, 8)
    )
    
    plt.title("Figure 2: Failure Reason Analysis of Different Hinter Models in Chinese Taboo Experiment", fontsize=22, weight="bold", pad=16)
    plt.xlabel("Hinter Model", fontsize=20)
    plt.ylabel("Failure Reason Percentage", fontsize=20)
    plt.xticks(rotation=0, fontsize=18)
    plt.legend(title="Failure Reason", fontsize=14, title_fontsize=16)
    plt.tight_layout()
    
    # Save images
    fname = "Chinese_Hinter_FailureReasons"
    plt.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
    plt.savefig(out_dir / f"{fname}.png", dpi=300, bbox_inches="tight")
    plt.show()

# =============================================
# Figure 3: Success rate by part of speech bar chart
# =============================================
def generate_pos_success_bar_chart():
    # Calculate success rate by model and part of speech
    pos_success = (
        complete_experiment_results
        .groupby(['hinter_model', 'part_of_speech'])['success']
        .mean()
        .unstack()
    )
    
    # Rename model names
    pos_success.index = [label_map[model] for model in pos_success.index]
    
    # Draw grouped bar chart
    plt.figure(figsize=(14, 8))
    pos_success.plot(
        kind='bar',
        colormap='magma',
        figsize=(14, 8)
    )
    
    plt.title("Figure 3: Success Rate by Part of Speech in Chinese Taboo Experiment", fontsize=22, weight="bold", pad=16)
    plt.xlabel("Hinter Model", fontsize=20)
    plt.ylabel("Success Rate", fontsize=20)
    plt.xticks(rotation=0, fontsize=18)
    plt.legend(title="Part of Speech", fontsize=14, title_fontsize=16)
    plt.tight_layout()
    
    # Save images
    fname = "Chinese_POS_SuccessRate"
    plt.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
    plt.savefig(out_dir / f"{fname}.png", dpi=300, bbox_inches="tight")
    plt.show()

# Execute analysis and generate charts
if __name__ == "__main__":
    print("Generating Figure 1: Success rate heatmap of different models as hinters...")
    generate_hinter_success_heatmap()
    
    print("Generating Figure 2: Failure reason analysis stacked chart...")
    generate_failure_reason_stacked_chart()
    
    print("Generating Figure 3: Success rate by part of speech bar chart...")
    generate_pos_success_bar_chart()
    
    print("Analysis complete, charts saved to figures/ directory")
