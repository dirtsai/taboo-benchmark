
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
baseline_path = "results/chinese_full_experiment_20250717_222959/chinese_full_results_20250717_222959.csv"
optimized_path = "results/chinese_full_experiment_20250717_231541/chinese_full_char_count_results_20250717_231541.csv"

baseline_results = pd.read_csv(baseline_path)
optimized_results = pd.read_csv(optimized_path)

print(f"Baseline results shape: {baseline_results.shape}")
print(f"Optimized results shape: {optimized_results.shape}")

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

# =============================================
# Prepare data for comparison
# =============================================

# Extract unique identifiers for joining
baseline_results['unique_id'] = baseline_results['target_word'] + '_' + baseline_results['hinter_model'] + '_' + baseline_results['guesser_model']
optimized_results['unique_id'] = optimized_results['target_word'] + '_' + optimized_results['hinter_model'] + '_' + optimized_results['guesser_model']

# Filter to only include the two models we're comparing (gemini-2.5-flash and deepseek-v3)
baseline_filtered = baseline_results[
    (baseline_results['hinter_model'] == 'google/gemini-2.5-flash') | 
    (baseline_results['hinter_model'] == 'deepseek/deepseek-chat-v3-0324')
]

optimized_filtered = optimized_results[
    (optimized_results['hinter_model'] == 'google/gemini-2.5-flash') | 
    (optimized_results['hinter_model'] == 'deepseek/deepseek-chat-v3-0324')
]

print(f"Filtered baseline results shape: {baseline_filtered.shape}")
print(f"Filtered optimized results shape: {optimized_filtered.shape}")

# Perform left join on optimized results
merged_results = pd.merge(
    optimized_filtered[['unique_id', 'target_word', 'part_of_speech', 'category', 'hinter_model', 'guesser_model', 'success', 'turns_used', 'final_guess', 'failure_reason', 'duration_seconds']],
    baseline_filtered[['unique_id', 'success', 'turns_used', 'final_guess', 'failure_reason', 'duration_seconds']],
    on='unique_id',
    how='left',
    suffixes=('_optimized', '_baseline')
)

print(f"Merged results shape: {merged_results.shape}")
print(f"Sample of merged data:\n{merged_results.head()}")

# =============================================
# Calculate overall metrics
# =============================================
def calculate_overall_metrics():
    # Overall success rates
    baseline_success_rate = baseline_filtered['success'].mean()
    optimized_success_rate = optimized_filtered['success'].mean()
    
    # Average turns for successful games
    baseline_turns = baseline_filtered[baseline_filtered['success']]['turns_used'].mean()
    optimized_turns = optimized_filtered[optimized_filtered['success']]['turns_used'].mean()
    
    # Average duration
    baseline_duration = baseline_filtered['duration_seconds'].mean()
    optimized_duration = optimized_filtered['duration_seconds'].mean()
    
    # Success rate improvement
    improvement = (optimized_success_rate - baseline_success_rate) * 100
    
    print("\n===== Overall Metrics =====")
    print(f"Baseline Success Rate: {baseline_success_rate:.4f} ({baseline_success_rate*100:.2f}%)")
    print(f"Optimized Success Rate: {optimized_success_rate:.4f} ({optimized_success_rate*100:.2f}%)")
    print(f"Improvement: {improvement:.2f}%")
    print(f"Average Turns (Success Cases) - Baseline: {baseline_turns:.2f}, Optimized: {optimized_turns:.2f}")
    print(f"Average Duration - Baseline: {baseline_duration:.2f}s, Optimized: {optimized_duration:.2f}s")
    
    # Create a summary dataframe for visualization
    summary_data = pd.DataFrame({
        'Metric': ['Success Rate', 'Avg Turns (Success)', 'Avg Duration (s)'],
        'Baseline': [baseline_success_rate, baseline_turns, baseline_duration],
        'Optimized': [optimized_success_rate, optimized_turns, optimized_duration]
    })
    
    return summary_data

# =============================================
# Calculate model accuracy by hinter
# =============================================
def calculate_model_accuracy(data):
    # Group by hinter model and calculate success rates
    hinter_accuracy = data.groupby('hinter_model').agg({
        'success_optimized': 'mean',
        'success_baseline': 'mean'
    }).reset_index()
    
    # Calculate improvement
    hinter_accuracy['improvement'] = (hinter_accuracy['success_optimized'] - hinter_accuracy['success_baseline']) * 100
    
    # Rename models using label map
    hinter_accuracy['hinter_model'] = hinter_accuracy['hinter_model'].map(lambda x: label_map.get(x, x))
    
    return hinter_accuracy

# =============================================
# Calculate accuracy by guesser model
# =============================================
def calculate_guesser_accuracy(data):
    # Group by guesser model and calculate success rates
    guesser_accuracy = data.groupby('guesser_model').agg({
        'success_optimized': 'mean',
        'success_baseline': 'mean'
    }).reset_index()
    
    # Calculate improvement
    guesser_accuracy['improvement'] = (guesser_accuracy['success_optimized'] - guesser_accuracy['success_baseline']) * 100
    
    # Rename models using label map
    guesser_accuracy['guesser_model'] = guesser_accuracy['guesser_model'].map(lambda x: label_map.get(x, x))
    
    return guesser_accuracy

# =============================================
# Calculate metrics by part of speech
# =============================================
def calculate_pos_metrics(data):
    # Group by part of speech and calculate success rates
    pos_metrics = data.groupby('part_of_speech').agg({
        'success_optimized': 'mean',
        'success_baseline': 'mean'
    }).reset_index()
    
    # Calculate improvement
    pos_metrics['improvement'] = (pos_metrics['success_optimized'] - pos_metrics['success_baseline']) * 100
    
    return pos_metrics

# =============================================
# Analyze failure reasons
# =============================================
def analyze_failure_reasons():
    # Count failure reasons in baseline
    baseline_failures = baseline_filtered[~baseline_filtered['success']]
    baseline_failure_counts = baseline_failures['failure_reason'].value_counts().reset_index()
    baseline_failure_counts.columns = ['failure_reason', 'count_baseline']
    baseline_failure_counts['percentage_baseline'] = baseline_failure_counts['count_baseline'] / len(baseline_failures) * 100
    
    # Count failure reasons in optimized
    optimized_failures = optimized_filtered[~optimized_filtered['success']]
    optimized_failure_counts = optimized_failures['failure_reason'].value_counts().reset_index()
    optimized_failure_counts.columns = ['failure_reason', 'count_optimized']
    optimized_failure_counts['percentage_optimized'] = optimized_failure_counts['count_optimized'] / len(optimized_failures) * 100
    
    # Merge failure reasons
    failure_comparison = pd.merge(
        baseline_failure_counts, 
        optimized_failure_counts, 
        on='failure_reason', 
        how='outer'
    ).fillna(0)
    
    # Calculate difference
    failure_comparison['count_diff'] = failure_comparison['count_optimized'] - failure_comparison['count_baseline']
    failure_comparison['percentage_diff'] = failure_comparison['percentage_optimized'] - failure_comparison['percentage_baseline']
    
    return failure_comparison

# =============================================
# Analyze turns distribution
# =============================================
def analyze_turns_distribution():
    # Turns distribution for successful games
    baseline_turns = baseline_filtered[baseline_filtered['success']]['turns_used'].value_counts().sort_index().reset_index()
    baseline_turns.columns = ['turns', 'count_baseline']
    baseline_turns['percentage_baseline'] = baseline_turns['count_baseline'] / baseline_turns['count_baseline'].sum() * 100
    
    optimized_turns = optimized_filtered[optimized_filtered['success']]['turns_used'].value_counts().sort_index().reset_index()
    optimized_turns.columns = ['turns', 'count_optimized']
    optimized_turns['percentage_optimized'] = optimized_turns['count_optimized'] / optimized_turns['count_optimized'].sum() * 100
    
    # Merge turns data
    turns_comparison = pd.merge(
        baseline_turns, 
        optimized_turns, 
        on='turns', 
        how='outer'
    ).fillna(0)
    
    return turns_comparison

# =============================================
# Generate bar charts
# =============================================
def generate_hinter_comparison_chart(hinter_accuracy):
    plt.figure(figsize=(10, 6))
    
    # Reshape data for seaborn
    hinter_data = pd.melt(
        hinter_accuracy, 
        id_vars=['hinter_model'],
        value_vars=['success_baseline', 'success_optimized'],
        var_name='version', 
        value_name='accuracy'
    )
    
    # Map version names to more readable labels
    hinter_data['version'] = hinter_data['version'].map({
        'success_baseline': 'Baseline',
        'success_optimized': 'Optimized'
    })
    
    # Create grouped bar chart
    sns.barplot(
        data=hinter_data,
        x='hinter_model',
        y='accuracy',
        hue='version',
        palette='magma'
    )
    
    plt.title("Hinter Model Accuracy: Before vs After Prompt Optimization", fontsize=16, weight="bold")
    plt.xlabel("Hinter Model", fontsize=14)
    plt.ylabel("Success Rate", fontsize=14)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for i, bar in enumerate(plt.gca().patches):
        plt.gca().text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f'{bar.get_height():.3f}',
            ha='center',
            fontsize=10
        )
    
    plt.tight_layout()
    
    # Save images
    fname = "Chinese_Hinter_Accuracy_Comparison"
    plt.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
    plt.savefig(out_dir / f"{fname}.png", dpi=300, bbox_inches="tight")
    plt.show()

def generate_guesser_comparison_chart(guesser_accuracy):
    plt.figure(figsize=(10, 6))
    
    # Reshape data for seaborn
    guesser_data = pd.melt(
        guesser_accuracy, 
        id_vars=['guesser_model'],
        value_vars=['success_baseline', 'success_optimized'],
        var_name='version', 
        value_name='accuracy'
    )
    
    # Map version names to more readable labels
    guesser_data['version'] = guesser_data['version'].map({
        'success_baseline': 'Baseline',
        'success_optimized': 'Optimized'
    })
    
    # Create grouped bar chart
    sns.barplot(
        data=guesser_data,
        x='guesser_model',
        y='accuracy',
        hue='version',
        palette='magma'
    )
    
    plt.title("Guesser Model Accuracy: Before vs After Prompt Optimization", fontsize=16, weight="bold")
    plt.xlabel("Guesser Model", fontsize=14)
    plt.ylabel("Success Rate", fontsize=14)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for i, bar in enumerate(plt.gca().patches):
        plt.gca().text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f'{bar.get_height():.3f}',
            ha='center',
            fontsize=10
        )
    
    plt.tight_layout()
    
    # Save images
    fname = "Chinese_Guesser_Accuracy_Comparison"
    plt.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
    plt.savefig(out_dir / f"{fname}.png", dpi=300, bbox_inches="tight")
    plt.show()

# =============================================
# Generate failure reason chart
# =============================================
def generate_failure_reason_chart(failure_comparison):
    plt.figure(figsize=(12, 7))
    
    # Reshape data for seaborn
    failure_data = pd.melt(
        failure_comparison, 
        id_vars=['failure_reason'],
        value_vars=['percentage_baseline', 'percentage_optimized'],
        var_name='version', 
        value_name='percentage'
    )
    
    # Map version names to more readable labels
    failure_data['version'] = failure_data['version'].map({
        'percentage_baseline': 'Baseline',
        'percentage_optimized': 'Optimized'
    })
    
    # Create grouped bar chart
    sns.barplot(
        data=failure_data,
        x='failure_reason',
        y='percentage',
        hue='version',
        palette='magma'
    )
    
    plt.title("Failure Reason Distribution: Before vs After Prompt Optimization", fontsize=16, weight="bold")
    plt.xlabel("Failure Reason", fontsize=14)
    plt.ylabel("Percentage of Failures (%)", fontsize=14)
    
    # Add value labels on bars
    for i, bar in enumerate(plt.gca().patches):
        if bar.get_height() > 0:
            plt.gca().text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f'{bar.get_height():.1f}%',
                ha='center',
                fontsize=9
            )
    
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    
    # Save images
    fname = "Chinese_Failure_Reason_Comparison"
    plt.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
    plt.savefig(out_dir / f"{fname}.png", dpi=300, bbox_inches="tight")
    plt.show()

# =============================================
# Generate turns distribution chart
# =============================================
def generate_turns_distribution_chart(turns_comparison):
    plt.figure(figsize=(10, 6))
    
    # Reshape data for seaborn
    turns_data = pd.melt(
        turns_comparison, 
        id_vars=['turns'],
        value_vars=['percentage_baseline', 'percentage_optimized'],
        var_name='version', 
        value_name='percentage'
    )
    
    # Map version names to more readable labels
    turns_data['version'] = turns_data['version'].map({
        'percentage_baseline': 'Baseline',
        'percentage_optimized': 'Optimized'
    })
    
    # Create grouped bar chart
    sns.barplot(
        data=turns_data,
        x='turns',
        y='percentage',
        hue='version',
        palette='magma'
    )
    
    plt.title("Turns Distribution for Successful Games: Before vs After Prompt Optimization", fontsize=16, weight="bold")
    plt.xlabel("Number of Turns", fontsize=14)
    plt.ylabel("Percentage of Successful Games (%)", fontsize=14)
    
    # Add value labels on bars
    for i, bar in enumerate(plt.gca().patches):
        if bar.get_height() > 0:
            plt.gca().text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f'{bar.get_height():.1f}%',
                ha='center',
                fontsize=9
            )
    
    plt.tight_layout()
    
    # Save images
    fname = "Chinese_Turns_Distribution_Comparison"
    plt.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
    plt.savefig(out_dir / f"{fname}.png", dpi=300, bbox_inches="tight")
    plt.show()

# =============================================
# Generate part of speech comparison chart
# =============================================
def generate_pos_comparison_chart(pos_metrics):
    plt.figure(figsize=(12, 7))
    
    # Reshape data for seaborn
    pos_data = pd.melt(
        pos_metrics, 
        id_vars=['part_of_speech'],
        value_vars=['success_baseline', 'success_optimized'],
        var_name='version', 
        value_name='success_rate'
    )
    
    # Map version names to more readable labels
    pos_data['version'] = pos_data['version'].map({
        'success_baseline': 'Baseline',
        'success_optimized': 'Optimized'
    })
    
    # Create grouped bar chart
    sns.barplot(
        data=pos_data,
        x='part_of_speech',
        y='success_rate',
        hue='version',
        palette='magma'
    )
    
    plt.title("Success Rate by Part of Speech: Before vs After Prompt Optimization", fontsize=16, weight="bold")
    plt.xlabel("Part of Speech", fontsize=14)
    plt.ylabel("Success Rate", fontsize=14)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for i, bar in enumerate(plt.gca().patches):
        plt.gca().text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f'{bar.get_height():.3f}',
            ha='center',
            fontsize=9
        )
    
    plt.tight_layout()
    
    # Save images
    fname = "Chinese_POS_Comparison"
    plt.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
    plt.savefig(out_dir / f"{fname}.png", dpi=300, bbox_inches="tight")
    plt.show()

# =============================================
# Generate summary metrics chart
# =============================================
def generate_summary_metrics_chart(summary_data):
    plt.figure(figsize=(10, 6))
    
    # Reshape data for seaborn
    summary_melted = pd.melt(
        summary_data, 
        id_vars=['Metric'],
        value_vars=['Baseline', 'Optimized'],
        var_name='Version', 
        value_name='Value'
    )
    
    # Create grouped bar chart
    g = sns.catplot(
        data=summary_melted,
        kind="bar",
        x="Metric",
        y="Value",
        hue="Version",
        palette="magma",
        height=5,
        aspect=1.5
    )
    
    g.set_axis_labels("Metric", "Value")
    g.legend.set_title("")
    
    # Add value labels on bars
    ax = g.axes[0, 0]
    for i, bar in enumerate(ax.patches):
        if i < 2:  # Success rate (format as percentage)
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02,
                f'{bar.get_height():.1%}',
                ha='center',
                fontsize=9
            )
        else:  # Other metrics (format as decimal)
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                f'{bar.get_height():.2f}',
                ha='center',
                fontsize=9
            )
    
    plt.title("Summary Metrics: Before vs After Prompt Optimization", fontsize=16, weight="bold")
    plt.tight_layout()
    
    # Save images
    fname = "Chinese_Summary_Metrics"
    plt.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
    plt.savefig(out_dir / f"{fname}.png", dpi=300, bbox_inches="tight")
    plt.show()

# =============================================
# Save detailed analysis to CSV
# =============================================
def save_detailed_analysis(hinter_accuracy, guesser_accuracy, pos_metrics, failure_comparison, turns_comparison):
    # Create a timestamp for the output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save all analysis dataframes to CSV
    hinter_accuracy.to_csv(out_dir / f"hinter_model_comparison_{timestamp}.csv", index=False)
    guesser_accuracy.to_csv(out_dir / f"guesser_model_comparison_{timestamp}.csv", index=False)
    pos_metrics.to_csv(out_dir / f"pos_comparison_{timestamp}.csv", index=False)
    failure_comparison.to_csv(out_dir / f"failure_reason_comparison_{timestamp}.csv", index=False)
    turns_comparison.to_csv(out_dir / f"turns_distribution_comparison_{timestamp}.csv", index=False)
    
    print(f"\nDetailed analysis saved to CSV files in {out_dir} directory")

# Execute analysis and generate charts
if __name__ == "__main__":
    print("Calculating overall metrics...")
    summary_data = calculate_overall_metrics()
    
    print("\nCalculating model accuracy...")
    hinter_accuracy = calculate_model_accuracy(merged_results)
    guesser_accuracy = calculate_guesser_accuracy(merged_results)
    
    print("\nCalculating part of speech metrics...")
    pos_metrics = calculate_pos_metrics(merged_results)
    
    print("\nAnalyzing failure reasons...")
    failure_comparison = analyze_failure_reasons()
    
    print("\nAnalyzing turns distribution...")
    turns_comparison = analyze_turns_distribution()
    
    print("\nHinter model accuracy comparison:")
    print(hinter_accuracy)
    
    print("\nGuesser model accuracy comparison:")
    print(guesser_accuracy)
    
    print("\nPart of speech comparison:")
    print(pos_metrics)
    
    print("\nFailure reason comparison:")
    print(failure_comparison)
    
    print("\nTurns distribution comparison:")
    print(turns_comparison)
    
    print("\nGenerating summary metrics chart...")
    generate_summary_metrics_chart(summary_data)
    
    print("\nGenerating hinter model comparison chart...")
    generate_hinter_comparison_chart(hinter_accuracy)
    
    print("\nGenerating guesser model comparison chart...")
    generate_guesser_comparison_chart(guesser_accuracy)
    
    print("\nGenerating failure reason chart...")
    generate_failure_reason_chart(failure_comparison)
    
    print("\nGenerating turns distribution chart...")
    generate_turns_distribution_chart(turns_comparison)
    
    print("\nGenerating part of speech comparison chart...")
    generate_pos_comparison_chart(pos_metrics)
    
    print("\nSaving detailed analysis to CSV...")
    save_detailed_analysis(hinter_accuracy, guesser_accuracy, pos_metrics, failure_comparison, turns_comparison)
    
    print("\nAnalysis complete, charts saved to figures/ directory")