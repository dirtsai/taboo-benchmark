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
    print(f"Target words: {bert_results['target'].nunique()}")
    print(f"Hinter models: {bert_results['hinter_model'].unique()}")
    print(f"Guesser models: {bert_results['guesser_model'].unique()}")
    return bert_results

# ---------- Model Name Mapping ----------
# "Full name → Short name" mapping
label_map = {
    "openai/gpt-4o":                  "gpt-4o",
    "google/gemini-2.5-flash":        "gemini-2.5-flash", 
    "deepseek/deepseek-chat-v3-0324": "deepseek-v3",
    "moonshotai/kimi-k2":             "kimi-k2",
    "gpt-4o":                         "gpt-4o",
    "gemini-2.5-flash":               "gemini-2.5-flash",
    "deepseek-v3":                    "deepseek-v3",
    "kimi-k2":                        "kimi-k2"
}

# Set theme
sns.set_theme(style="white", font_scale=1.2)
out_dir = pathlib.Path("figures")
out_dir.mkdir(exist_ok=True)

def plot_bert_similarity_bar(df, group_key, fname, palette="magma"):
    """Plot BERT similarity by group_key and save (with short model names)"""
    similarity_mean = (
        df.groupby(group_key)["similarity"]
          .mean()
          .sort_values(ascending=False)
    )
    
    # Replace index with short names
    short_idx = similarity_mean.index.map(lambda x: label_map.get(x, x))
    
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=short_idx, y=similarity_mean.values,
        palette=sns.color_palette(palette, n_colors=len(similarity_mean))
    )
    
    # Set y-axis limits with some headroom
    max_val = similarity_mean.max()
    plt.ylim(0, max_val * 1.1)
    plt.axhline(max_val, ls="--", c="gray", lw=1, alpha=0.5)
    
    plt.ylabel("Average BERT Similarity", fontsize=20)
    plt.xlabel(f"{group_key.split('_')[0].title()} Model", fontsize=20)
    
    # Tick settings: horizontal, font size 18
    plt.xticks(rotation=0, fontsize=18)
    plt.yticks(fontsize=18)
    
    # Add value labels on bars
    for idx, val in enumerate(similarity_mean.values):
        plt.text(idx, val + max_val * 0.02, f"{val:.3f}", ha="center", va="bottom",
                 fontsize=12, weight="bold")
    
    plt.tight_layout()
    plt.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
    plt.savefig(out_dir / f"{fname}.png", dpi=300, bbox_inches="tight")
    plt.show()
    return similarity_mean

def analyze_bert_similarity_by_models(filepath):
    """Main analysis function for BERT similarity by models"""
    # Load data
    bert_results = load_bert_results(filepath)
    
    # 1) By Hinter Model (Clue Quality Perspective)
    print("\nBERT Similarity by Hinter Model (Clue Quality):")
    similarity_hinter = plot_bert_similarity_bar(
        bert_results,
        group_key="hinter_model", 
        fname="BERT_Similarity_by_Hinter"
    )
    print(similarity_hinter)
    
    # 2) By Guesser Model (Guessing Accuracy Perspective)  
    print("\nBERT Similarity by Guesser Model (Guessing Accuracy):")
    similarity_guesser = plot_bert_similarity_bar(
        bert_results,
        group_key="guesser_model",
        fname="BERT_Similarity_by_Guesser" 
    )
    print(similarity_guesser)
    
    # 3) Combined analysis: Similarity vs Success Rate
    if 'success' in bert_results.columns:
        print("\nCorrelation Analysis: BERT Similarity vs Success Rate")
        
        # By Hinter
        hinter_combined = bert_results.groupby('hinter_model').agg({
            'similarity': 'mean',
            'success': 'mean'
        }).round(4)
        print("\nHinter Model - Similarity vs Success:")
        print(hinter_combined)
        
        # By Guesser
        guesser_combined = bert_results.groupby('guesser_model').agg({
            'similarity': 'mean', 
            'success': 'mean'
        }).round(4)
        print("\nGuesser Model - Similarity vs Success:")
        print(guesser_combined)
    
    print("→ BERT similarity bar charts saved to figures/ directory")
    
    return {
        'hinter_similarity': similarity_hinter,
        'guesser_similarity': similarity_guesser
    }

# Usage
if __name__ == "__main__":
    # Replace with your BERT similarity result file path
    bert_file = "bert_multi_model_similarity_analysis.csv"
    results = analyze_bert_similarity_by_models(bert_file)
