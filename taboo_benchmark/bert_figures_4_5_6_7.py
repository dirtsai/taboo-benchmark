import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pathlib

# ---------- Load BERT Similarity Results ----------
def load_bert_results(filepath):
    """Load BERT similarity analysis results"""
    bert_results = pd.read_csv(filepath, encoding='utf-8')
    print(f"Loaded {len(bert_results)} BERT similarity records")
    print(f"Overall average similarity: {bert_results['similarity'].mean():.3f}")
    print(f"Standard deviation: {bert_results['similarity'].std():.3f}")
    return bert_results

# ---------- Model Name Mapping ----------
# "Full name → Short name" mapping
label_map = {
    "openai/gpt-4o":                  "GPT-4o",
    "google/gemini-2.5-flash":        "Gemini-2.5-Pro", 
    "deepseek/deepseek-chat-v3-0324": "DeepSeek-Chat-V3",
    "moonshotai/kimi-k2":             "Claude-Sonnet-4",  # Based on your text description
    "gpt-4o":                         "GPT-4o",
    "gemini-2.5-flash":               "Gemini-2.5-Pro",
    "deepseek-v3":                    "DeepSeek-Chat-V3", 
    "kimi-k2":                        "Claude-Sonnet-4"
}

# Set theme
sns.set_theme(style="white", font_scale=1.2)
out_dir = pathlib.Path("figures")
out_dir.mkdir(exist_ok=True)

def plot_figure_4_5_hinter_similarity(df):
    """Figure 4-5: Average similarity by Hinter model"""
    # Calculate average similarity by hinter model
    hinter_similarity = (
        df.groupby('hinter_model')['similarity']
          .mean()
          .sort_values(ascending=False)
    )
    
    # Map to short names
    short_names = hinter_similarity.index.map(lambda x: label_map.get(x, x))
    
    plt.figure(figsize=(10, 6))
    bars = sns.barplot(
        x=short_names, y=hinter_similarity.values,
        palette=sns.color_palette("magma", n_colors=len(hinter_similarity))
    )
    
    plt.ylim(0.90, 1.0)  # Focus on the high similarity range
    plt.axhline(0.95, ls="--", c="gray", lw=1, alpha=0.7)  # Reference line
    plt.ylabel("Average BERT Similarity", fontsize=20)
    plt.xlabel("Hinter Model", fontsize=20)
    plt.xticks(rotation=0, fontsize=18)
    plt.yticks(fontsize=18)
    
    # Add value labels on bars
    for idx, val in enumerate(hinter_similarity.values):
        plt.text(idx, val + 0.005, f"{val:.3f}", ha="center", va="bottom",
                 fontsize=12, weight="bold")
    
    plt.tight_layout()
    plt.savefig(out_dir / "Figure_4-5_Hinter_Similarity.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "Figure_4-5_Hinter_Similarity.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    print("Figure 4-5: Hinter Model Average Similarity")
    for model, sim in zip(short_names, hinter_similarity.values):
        print(f"  {model}: {sim:.3f}")
    
    return hinter_similarity

def plot_figure_4_6_turn_similarity(df):
    """Figure 4-6: Similarity trend by turn number"""
    # Calculate average similarity by turn
    turn_similarity = (
        df.groupby('turn_number')['similarity']
          .mean()
          .sort_index()
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(turn_similarity.index, turn_similarity.values, 'o-', 
             linewidth=3, markersize=8, color='#440154')  # Dark purple from magma
    
    # Add error bars (standard deviation)
    turn_std = df.groupby('turn_number')['similarity'].std()
    plt.fill_between(turn_similarity.index, 
                     turn_similarity.values - turn_std.values,
                     turn_similarity.values + turn_std.values, 
                     alpha=0.3, color='#440154')
    
    plt.ylim(0.90, 1.0)  # Focus on the high similarity range
    plt.axhline(0.95, ls="--", c="gray", lw=1, alpha=0.7)  # Reference line
    plt.ylabel("Average BERT Similarity", fontsize=20)
    plt.xlabel("Turn Number", fontsize=20)
    plt.xticks(turn_similarity.index, fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on points
    for turn, val in zip(turn_similarity.index, turn_similarity.values):
        plt.text(turn, val + 0.005, f"{val:.3f}", ha="center", va="bottom",
                 fontsize=12, weight="bold")
    
    plt.tight_layout()
    plt.savefig(out_dir / "Figure_4-6_Turn_Similarity.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "Figure_4-6_Turn_Similarity.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    print("Figure 4-6: Similarity by Turn Number")
    for turn, sim in turn_similarity.items():
        print(f"  Turn {turn}: {sim:.3f}")
    
    return turn_similarity

def plot_figure_4_7_heatmap(df):
    """Figure 4-7: Hinter-Guesser combination similarity heatmap"""
    # Calculate similarity by hinter-guesser combination
    heatmap_data = (
        df.groupby(['hinter_model', 'guesser_model'])['similarity']
          .mean()
          .unstack()
    )
    
    # Map to short names
    heatmap_data.index = heatmap_data.index.map(lambda x: label_map.get(x, x))
    heatmap_data.columns = heatmap_data.columns.map(lambda x: label_map.get(x, x))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, 
                annot=True, 
                fmt='.3f', 
                cmap='magma',
                cbar_kws={'label': 'Average BERT Similarity'},
                square=True,
                linewidths=0.5)
    
    plt.ylabel("Hinter Model", fontsize=20)
    plt.xlabel("Guesser Model", fontsize=20)
    plt.xticks(fontsize=16, rotation=45, ha='right')
    plt.yticks(fontsize=16, rotation=0)
    
    # Adjust colorbar
    cbar = plt.gca().collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Average BERT Similarity', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(out_dir / "Figure_4-7_Hinter_Guesser_Heatmap.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "Figure_4-7_Hinter_Guesser_Heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    print("Figure 4-7: Hinter-Guesser Combination Similarity")
    print(heatmap_data.round(3))
    
    return heatmap_data

def generate_bert_analysis_figures(filepath):
    """Generate all three BERT analysis figures"""
    print("🔍 Generating BERT Similarity Analysis Figures...")
    
    # Load data
    bert_results = load_bert_results(filepath)
    
    # Convert similarity to percentage scale (0-1 to 0-100) if needed
    if bert_results['similarity'].max() <= 1.0:
        print("Converting similarity to percentage scale...")
        bert_results['similarity'] = bert_results['similarity'] * 100
    
    print(f"\nOverall Statistics:")
    print(f"  Average similarity: {bert_results['similarity'].mean():.1f}%")
    print(f"  Standard deviation: {bert_results['similarity'].std():.3f}")
    print(f"  Range: {bert_results['similarity'].min():.1f}% - {bert_results['similarity'].max():.1f}%")
    
    # Generate Figure 4-5: Hinter Model Similarity
    print("\n" + "="*50)
    hinter_sim = plot_figure_4_5_hinter_similarity(bert_results)
    
    # Generate Figure 4-6: Turn-wise Similarity
    print("\n" + "="*50)
    turn_sim = plot_figure_4_6_turn_similarity(bert_results)
    
    # Generate Figure 4-7: Hinter-Guesser Heatmap
    print("\n" + "="*50)
    heatmap_data = plot_figure_4_7_heatmap(bert_results)
    
    print("\n" + "="*50)
    print("→ All three figures saved to figures/ directory")
    print("  - Figure_4-5_Hinter_Similarity.pdf/png")
    print("  - Figure_4-6_Turn_Similarity.pdf/png") 
    print("  - Figure_4-7_Hinter_Guesser_Heatmap.pdf/png")
    
    return {
        'hinter_similarity': hinter_sim,
        'turn_similarity': turn_sim,
        'heatmap_data': heatmap_data
    }

# Usage
if __name__ == "__main__":
    # Replace with your BERT similarity result file path
    bert_file = "bert_multi_model_similarity_analysis.csv"
    results = generate_bert_analysis_figures(bert_file)
