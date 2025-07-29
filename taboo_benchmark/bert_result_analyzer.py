# BERT Semantic Similarity Result Analyzer - Direct analysis of saved result files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set font for better display
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_bert_similarity_results(filepath):
    """Load BERT similarity analysis results"""
    try:
        similarity_df = pd.read_csv(filepath, encoding='utf-8')
        print(f"✅ Successfully loaded BERT similarity results: {len(similarity_df)} records")
        print(f"Data columns: {list(similarity_df.columns)}")
        
        # Display basic statistics
        print(f"\n📊 Basic Statistics:")
        print(f"  • Total guesses: {len(similarity_df):,}")
        print(f"  • Average similarity: {similarity_df['similarity'].mean():.4f}")
        print(f"  • Similarity range: {similarity_df['similarity'].min():.4f} - {similarity_df['similarity'].max():.4f}")
        
        if 'model_pair' in similarity_df.columns:
            print(f"  • Number of model pairs: {similarity_df['model_pair'].nunique()}")
            print(f"  • Model pairs: {list(similarity_df['model_pair'].unique())}")
        
        return similarity_df
    except Exception as e:
        print(f"❌ Failed to load result file: {e}")
        return None

def analyze_model_performance(similarity_df):
    """Analyze model performance"""
    if similarity_df is None or len(similarity_df) == 0:
        return None
    
    print("\n🔍 Model Performance Analysis")
    
    # 1. Analysis by model pairs
    if 'model_pair' in similarity_df.columns:
        model_pair_stats = similarity_df.groupby('model_pair').agg({
            'similarity': ['mean', 'std', 'count'],
            'success': 'mean' if 'success' in similarity_df.columns else lambda x: None
        }).round(4)
        print("\nModel Pair Performance:")
        print(model_pair_stats)
    
    # 2. Analysis by Hinter model
    if 'hinter_model' in similarity_df.columns:
        hinter_stats = similarity_df.groupby('hinter_model').agg({
            'similarity': ['mean', 'std', 'count'],
            'success': 'mean' if 'success' in similarity_df.columns else lambda x: None
        }).round(4)
        print("\nHinter Model Performance:")
        print(hinter_stats)
    
    # 3. Analysis by Guesser model
    if 'guesser_model' in similarity_df.columns:
        guesser_stats = similarity_df.groupby('guesser_model').agg({
            'similarity': ['mean', 'std', 'count'],
            'success': 'mean' if 'success' in similarity_df.columns else lambda x: None
        }).round(4)
        print("\nGuesser Model Performance:")
        print(guesser_stats)
    
    return {
        'model_pair': model_pair_stats if 'model_pair' in similarity_df.columns else None,
        'hinter': hinter_stats if 'hinter_model' in similarity_df.columns else None,
        'guesser': guesser_stats if 'guesser_model' in similarity_df.columns else None
    }

def analyze_turn_patterns(similarity_df):
    """Analyze turn patterns"""
    if 'turn_number' not in similarity_df.columns:
        print("❌ Missing turn information")
        return None
    
    print("\n📈 Turn Pattern Analysis")
    
    # 1. Overall performance by turn
    turn_stats = similarity_df.groupby('turn_number')['similarity'].agg(['mean', 'std', 'count']).round(4)
    print("\nOverall Performance by Turn:")
    print(turn_stats)
    
    # 2. Model pair performance by turn
    if 'model_pair' in similarity_df.columns:
        model_turn_stats = similarity_df.groupby(['model_pair', 'turn_number'])['similarity'].mean().round(4)
        print("\nModel Pair Performance by Turn:")
        print(model_turn_stats.unstack(level=0))
    
    return {
        'turn_overall': turn_stats,
        'model_turn': model_turn_stats if 'model_pair' in similarity_df.columns else None
    }

def create_comprehensive_visualization(similarity_df):
    """Create comprehensive visualization"""
    if similarity_df is None or len(similarity_df) == 0:
        print("❌ No data available for visualization")
        return
    
    sns.set_style("white")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('BERT Semantic Similarity Analysis - Comprehensive Results', fontsize=16, fontweight='bold')
    
    # 1. Model pair overall performance comparison
    if 'model_pair' in similarity_df.columns:
        model_means = similarity_df.groupby('model_pair')['similarity'].mean().sort_values(ascending=False)
        bars = axes[0, 0].bar(range(len(model_means)), model_means.values, 
                              color=plt.cm.Set2(np.linspace(0, 1, len(model_means))))
        axes[0, 0].set_title('Model Pair Average Similarity Comparison', fontweight='bold')
        axes[0, 0].set_ylabel('Average Similarity')
        axes[0, 0].set_xticks(range(len(model_means)))
        axes[0, 0].set_xticklabels(model_means.index, rotation=45, ha='right')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                            f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Similarity trend by turn
    if 'turn_number' in similarity_df.columns:
        turn_means = similarity_df.groupby('turn_number')['similarity'].mean()
        turn_stds = similarity_df.groupby('turn_number')['similarity'].std()
        
        axes[0, 1].plot(turn_means.index, turn_means.values, 'o-', linewidth=2, markersize=8)
        axes[0, 1].fill_between(turn_means.index, 
                                turn_means.values - turn_stds.values,
                                turn_means.values + turn_stds.values, 
                                alpha=0.3)
        axes[0, 1].set_title('Similarity Trend by Turn Number', fontweight='bold')
        axes[0, 1].set_xlabel('Turn Number')
        axes[0, 1].set_ylabel('Average Similarity')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Success vs failure game similarity distribution
    if 'success' in similarity_df.columns:
        success_data = similarity_df[similarity_df['success'] == True]['similarity']
        failure_data = similarity_df[similarity_df['success'] == False]['similarity']
        
        if len(success_data) > 0:
            axes[0, 2].hist(success_data, bins=30, alpha=0.7, label='Successful Games', color='green')
        if len(failure_data) > 0:
            axes[0, 2].hist(failure_data, bins=30, alpha=0.7, label='Failed Games', color='red')
        axes[0, 2].set_title('Success vs Failure Game Similarity Distribution', fontweight='bold')
        axes[0, 2].set_xlabel('Similarity')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Model pair similarity heatmap
    if 'hinter_model' in similarity_df.columns and 'guesser_model' in similarity_df.columns:
        heatmap_data = similarity_df.groupby(['hinter_model', 'guesser_model'])['similarity'].mean().unstack()
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis', ax=axes[1, 0])
        axes[1, 0].set_title('Model Pair Similarity Heatmap', fontweight='bold')
        axes[1, 0].set_xlabel('Guesser Model')
        axes[1, 0].set_ylabel('Hinter Model')
    
    # 5. Similarity distribution by turn (boxplot)
    if 'turn_number' in similarity_df.columns:
        turn_data = [similarity_df[similarity_df['turn_number'] == turn]['similarity'].values 
                     for turn in sorted(similarity_df['turn_number'].unique())]
        turn_labels = [f'Turn {turn}' for turn in sorted(similarity_df['turn_number'].unique())]
        
        box_plot = axes[1, 1].boxplot(turn_data, labels=turn_labels, patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(turn_data)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        axes[1, 1].set_title('Similarity Distribution by Turn', fontweight='bold')
        axes[1, 1].set_xlabel('Turn')
        axes[1, 1].set_ylabel('Similarity')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Model pair success rate vs average similarity scatter plot
    if 'model_pair' in similarity_df.columns and 'success' in similarity_df.columns:
        model_stats = similarity_df.groupby('model_pair').agg({
            'similarity': 'mean',
            'success': 'mean'
        }).reset_index()
        
        scatter = axes[1, 2].scatter(model_stats['similarity'], model_stats['success'], 
                                   s=100, alpha=0.7, c=range(len(model_stats)), cmap='tab10')
        
        for i, row in model_stats.iterrows():
            axes[1, 2].annotate(row['model_pair'], 
                              (row['similarity'], row['success']),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, ha='left')
        
        axes[1, 2].set_title('Model Pair Success Rate vs Average Similarity', fontweight='bold')
        axes[1, 2].set_xlabel('Average Similarity')
        axes[1, 2].set_ylabel('Success Rate')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def generate_comprehensive_report(similarity_df):
    """Generate comprehensive analysis report"""
    if similarity_df is None or len(similarity_df) == 0:
        print("❌ No data available for analysis")
        return
    
    print("\n" + "="*80)
    print("                    BERT Semantic Similarity Comprehensive Analysis Report")
    print("="*80)
    
    # Basic statistics
    total_guesses = len(similarity_df)
    avg_similarity = similarity_df['similarity'].mean()
    std_similarity = similarity_df['similarity'].std()
    
    print(f"\n📊 Basic Statistics:")
    print(f"  • Total guesses: {total_guesses:,}")
    print(f"  • Average similarity: {avg_similarity:.4f}")
    print(f"  • Similarity standard deviation: {std_similarity:.4f}")
    print(f"  • Similarity range: {similarity_df['similarity'].min():.4f} - {similarity_df['similarity'].max():.4f}")
    
    # Model performance ranking
    if 'model_pair' in similarity_df.columns:
        print(f"\n🏆 Model Pair Performance Ranking:")
        model_ranking = similarity_df.groupby('model_pair').agg({
            'similarity': 'mean',
            'success': 'mean' if 'success' in similarity_df.columns else lambda x: None
        }).round(4).sort_values('similarity', ascending=False)
        
        for i, (model_pair, stats) in enumerate(model_ranking.iterrows(), 1):
            success_info = f", Success rate: {stats['success']:.4f}" if 'success' in stats and not pd.isna(stats['success']) else ""
            print(f"  {i}. {model_pair}: Similarity {stats['similarity']:.4f}{success_info}")
    
    # Turn analysis
    if 'turn_number' in similarity_df.columns:
        print(f"\n📈 Turn Performance Analysis:")
        turn_analysis = similarity_df.groupby('turn_number')['similarity'].agg(['mean', 'count']).round(4)
        for turn, stats in turn_analysis.iterrows():
            print(f"  • Turn {turn}: Average similarity {stats['mean']:.4f} ({int(stats['count'])} guesses)")
    
    print("\n" + "="*80)

def save_analysis_results(similarity_df, model_stats, turn_stats, output_prefix='bert_analysis'):
    """Save analysis results"""
    try:
        # Save summary statistics
        with pd.ExcelWriter(f'{output_prefix}_summary.xlsx') as writer:
            if model_stats and model_stats['model_pair'] is not None:
                model_stats['model_pair'].to_excel(writer, sheet_name='Model_Pair_Stats')
            if model_stats and model_stats['hinter'] is not None:
                model_stats['hinter'].to_excel(writer, sheet_name='Hinter_Stats')
            if model_stats and model_stats['guesser'] is not None:
                model_stats['guesser'].to_excel(writer, sheet_name='Guesser_Stats')
            if turn_stats and turn_stats['turn_overall'] is not None:
                turn_stats['turn_overall'].to_excel(writer, sheet_name='Turn_Stats')
        
        print(f"✅ Analysis results saved to: {output_prefix}_summary.xlsx")
        
    except Exception as e:
        print(f"❌ Failed to save results: {e}")

def main(result_filepath):
    """Main analysis function"""
    print("🔍 Starting BERT similarity result analysis...")
    
    # 1. Load result data
    similarity_df = load_bert_similarity_results(result_filepath)
    if similarity_df is None:
        return
    
    # 2. Model performance analysis
    model_stats = analyze_model_performance(similarity_df)
    
    # 3. Turn pattern analysis
    turn_stats = analyze_turn_patterns(similarity_df)
    
    # 4. Create visualization
    create_comprehensive_visualization(similarity_df)
    
    # 5. Generate comprehensive report
    generate_comprehensive_report(similarity_df)
    
    # 6. Save analysis results
    save_analysis_results(similarity_df, model_stats, turn_stats)
    
    print("🎉 BERT similarity result analysis completed!")

if __name__ == "__main__":
    # Usage example
    result_file = "bert_multi_model_similarity_analysis.csv"  # Replace with your result file path
    main(result_file)
