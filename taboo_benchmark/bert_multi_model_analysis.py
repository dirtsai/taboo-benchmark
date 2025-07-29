# BERT语义相似度分析 - 多模型多轮次版本
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import json
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class BERTSimilarityAnalyzer:
    """BERT语义相似度分析器"""
    
    def __init__(self, model_name='bert-base-chinese'):
        """初始化BERT模型"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ BERT模型已加载到 {self.device}")
    
    def get_embedding(self, text, max_length=10):
        """获取文本的BERT嵌入向量"""
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors='pt', 
                truncation=True, 
                max_length=max_length,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embedding
        except Exception as e:
            print(f"获取嵌入向量失败: {text}, 错误: {e}")
            return np.zeros((1, 768))
    
    def calculate_similarity(self, text1, text2):
        """计算两个文本的余弦相似度"""
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return similarity
    
    def analyze_game_similarities(self, game_data):
        """分析单场游戏的相似度变化"""
        target_word = game_data['target_word']
        all_guesses = game_data['all_guesses']
        
        similarities = []
        for i, guess in enumerate(all_guesses):
            similarity = self.calculate_similarity(guess, target_word)
            similarities.append({
                'turn_number': i + 1,
                'guess': guess,
                'target': target_word,
                'similarity': similarity
            })
        
        return similarities

def load_experiment_data():
    """加载实验数据"""
    try:
        with open('data/dataset.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        results_df = pd.read_csv('results/taboo_experiment_20250712_004918/complete_experiment_results.csv')
        
        dataset_df = pd.DataFrame(dataset)
        dataset_info = dataset_df[['target', 'part_of_speech']].copy()
        dataset_info = dataset_info.rename(columns={'target': 'target_word'})
        
        merged_df = results_df.merge(dataset_info, on='target_word', how='left')
        
        print(f"✅ 数据加载完成: {len(merged_df)} 条记录")
        return merged_df
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None

import re

def extract_guesses_from_all_guesses(all_guesses_str):
    """从all_guesses字段提取每轮猜测词"""
    if pd.isna(all_guesses_str):
        return []
    items = all_guesses_str.split('|')
    guesses = []
    for item in items:
        match = re.search(r'\[GUESS\]\s*(.+)', item)
        if match:
            guess = match.group(1).strip()
            if guess and guess not in ['INVALID_FORMAT', 'FORMAT_ERROR']:
                guesses.append(guess)
    return guesses

def run_bert_similarity_analysis():
    """运行BERT相似度分析（支持多模型多轮次分析）"""
    print("🔍 开始BERT语义相似度分析...")
    
    analyzer = BERTSimilarityAnalyzer()
    merged_df = load_experiment_data()
    if merged_df is None:
        return None
    
    print("📊 提取猜测数据...")
    game_analyses = []
    
    for idx, row in merged_df.iterrows():
        if idx % 100 == 0:
            print(f"处理进度: {idx}/{len(merged_df)}")
        
        guesses = extract_guesses_from_all_guesses(row['all_guesses'])
        
        if guesses:
            similarities = analyzer.analyze_game_similarities({
                'target_word': row['target_word'],
                'all_guesses': guesses
            })
            for sim in similarities:
                # 修正字段名：使用实际的CSV字段名
                hinter_model = row.get('hinter_model', 'Unknown')
                guesser_model = row.get('guesser_model', 'Unknown')
                
                # 清理模型名称（去掉openai/前缀等）
                if isinstance(hinter_model, str) and '/' in hinter_model:
                    hinter_model = hinter_model.split('/')[-1]
                if isinstance(guesser_model, str) and '/' in guesser_model:
                    guesser_model = guesser_model.split('/')[-1]
                
                sim.update({
                    'game_id': row.get('game_id', f'game_{idx}'),
                    'hinter_model': hinter_model,
                    'guesser_model': guesser_model,
                    'model_pair': f"{hinter_model}-{guesser_model}",
                    'success': row.get('success', False),
                    'turns_used': row.get('turns_used', 0),
                    'part_of_speech': row.get('part_of_speech', 'unknown'),
                    'category': row.get('category', 'unknown')
                })
            game_analyses.extend(similarities)
    
    if game_analyses:
        similarity_df = pd.DataFrame(game_analyses)
        print(f"✅ 相似度分析完成: {len(similarity_df)} 条记录")
        return similarity_df
    else:
        print("❌ 没有找到有效的猜测数据")
        return None

def analyze_multi_model_similarity_trends(similarity_df):
    """分析多模型多轮次相似度趋势"""
    if similarity_df is None or len(similarity_df) == 0:
        print("❌ 没有数据可供分析")
        return None
    
    print("\n🔍 多模型多轮次相似度趋势分析")
    
    # 按模型对和轮数分析
    model_turn_similarity = similarity_df.groupby(['model_pair', 'turn_number'])['similarity'].agg(['mean', 'std', 'count']).round(4)
    print("\n各模型对各轮平均相似度:")
    print(model_turn_similarity)
    
    # 按Hinter模型和轮数分析
    hinter_turn_similarity = similarity_df.groupby(['hinter_model', 'turn_number'])['similarity'].agg(['mean', 'std', 'count']).round(4)
    print("\n各Hinter模型各轮平均相似度:")
    print(hinter_turn_similarity)
    
    # 按Guesser模型和轮数分析
    guesser_turn_similarity = similarity_df.groupby(['guesser_model', 'turn_number'])['similarity'].agg(['mean', 'std', 'count']).round(4)
    print("\n各Guesser模型各轮平均相似度:")
    print(guesser_turn_similarity)
    
    return {
        'model_turn': model_turn_similarity,
        'hinter_turn': hinter_turn_similarity,
        'guesser_turn': guesser_turn_similarity
    }

def visualize_multi_model_similarity_analysis(similarity_df):
    """可视化多模型多轮次相似度分析"""
    if similarity_df is None or len(similarity_df) == 0:
        print("❌ 没有数据可供可视化")
        return
    
    sns.set_style("white")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('BERT语义相似度分析 - 多模型多轮次对比', fontsize=16, fontweight='bold')
    
    # 1. 各模型对按轮数的相似度变化
    model_turn_data = similarity_df.groupby(['model_pair', 'turn_number'])['similarity'].mean().unstack(level=0)
    model_turn_data.plot(kind='line', ax=axes[0, 0], marker='o', linewidth=2)
    axes[0, 0].set_title('各模型对按轮数的相似度变化', fontweight='bold')
    axes[0, 0].set_xlabel('轮数')
    axes[0, 0].set_ylabel('平均相似度')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Hinter模型按轮数的相似度变化
    hinter_turn_data = similarity_df.groupby(['hinter_model', 'turn_number'])['similarity'].mean().unstack(level=0)
    hinter_turn_data.plot(kind='line', ax=axes[0, 1], marker='s', linewidth=2)
    axes[0, 1].set_title('各Hinter模型按轮数的相似度变化', fontweight='bold')
    axes[0, 1].set_xlabel('轮数')
    axes[0, 1].set_ylabel('平均相似度')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 模型对相似度热力图
    heatmap_data = similarity_df.groupby(['hinter_model', 'guesser_model'])['similarity'].mean().unstack()
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis', ax=axes[0, 2])
    axes[0, 2].set_title('模型对相似度热力图', fontweight='bold')
    
    # 4. Hinter模型整体表现对比
    hinter_means = similarity_df.groupby('hinter_model')['similarity'].mean().sort_values(ascending=False)
    bars = axes[1, 0].bar(range(len(hinter_means)), hinter_means.values)
    axes[1, 0].set_title('Hinter模型平均相似度对比', fontweight='bold')
    axes[1, 0].set_xticks(range(len(hinter_means)))
    axes[1, 0].set_xticklabels(hinter_means.index, rotation=45, ha='right')
    
    # 5. Guesser模型整体表现对比
    guesser_means = similarity_df.groupby('guesser_model')['similarity'].mean().sort_values(ascending=False)
    bars = axes[1, 1].bar(range(len(guesser_means)), guesser_means.values)
    axes[1, 1].set_title('Guesser模型平均相似度对比', fontweight='bold')
    axes[1, 1].set_xticks(range(len(guesser_means)))
    axes[1, 1].set_xticklabels(guesser_means.index, rotation=45, ha='right')
    
    # 6. 各轮次整体相似度分布
    turn_data = [similarity_df[similarity_df['turn_number'] == turn]['similarity'].values 
                 for turn in sorted(similarity_df['turn_number'].unique())]
    turn_labels = [f'第{turn}轮' for turn in sorted(similarity_df['turn_number'].unique())]
    
    axes[1, 2].boxplot(turn_data, labels=turn_labels)
    axes[1, 2].set_title('各轮次相似度分布', fontweight='bold')
    axes[1, 2].set_xlabel('轮次')
    axes[1, 2].set_ylabel('相似度')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def generate_multi_model_similarity_report(similarity_df):
    """生成多模型多轮次相似度分析报告"""
    if similarity_df is None or len(similarity_df) == 0:
        print("❌ 没有数据可供分析")
        return
    
    print("\n" + "="*80)
    print("          BERT语义相似度分析报告 - 多模型多轮次版本")
    print("="*80)
    
    # 基础统计
    total_guesses = len(similarity_df)
    avg_similarity = similarity_df['similarity'].mean()
    
    print(f"\n📊 基础统计:")
    print(f"  • 总猜测数: {total_guesses:,}")
    print(f"  • 平均相似度: {avg_similarity:.4f}")
    
    # 模型对分析
    print(f"\n🤖 模型对分析:")
    for model_pair in similarity_df['model_pair'].unique():
        data = similarity_df[similarity_df['model_pair'] == model_pair]
        avg_sim = data['similarity'].mean()
        count = len(data)
        success_rate = data['success'].mean()
        print(f"  • {model_pair}: 相似度 {avg_sim:.4f}, 成功率 {success_rate:.4f} ({count} 次)")
    
    # 轮数分析
    print(f"\n📈 轮数分析:")
    for turn in sorted(similarity_df['turn_number'].unique()):
        turn_data = similarity_df[similarity_df['turn_number'] == turn]
        avg_sim = turn_data['similarity'].mean()
        count = len(turn_data)
        print(f"  • 第{turn}轮: 平均相似度 {avg_sim:.4f} ({count} 次猜测)")
    
    print("\n" + "="*80)

def save_multi_model_similarity_results(similarity_df, analysis_results):
    """保存多模型相似度分析结果"""
    if similarity_df is not None:
        similarity_df.to_csv('bert_multi_model_similarity_analysis.csv', index=False, encoding='utf-8')
        print(f"✅ 相似度分析结果已保存")
        
        if analysis_results:
            with pd.ExcelWriter('bert_multi_model_stats.xlsx') as writer:
                analysis_results['model_turn'].to_excel(writer, sheet_name='模型对轮数统计')
                analysis_results['hinter_turn'].to_excel(writer, sheet_name='Hinter轮数统计')
                analysis_results['guesser_turn'].to_excel(writer, sheet_name='Guesser轮数统计')
            print(f"✅ 统计结果已保存到Excel文件")

# 主函数
if __name__ == "__main__":
    # 运行BERT相似度分析
    similarity_df = run_bert_similarity_analysis()
    
    if similarity_df is not None:
        # 分析趋势
        analysis_results = analyze_multi_model_similarity_trends(similarity_df)
        
        # 可视化分析
        visualize_multi_model_similarity_analysis(similarity_df)
        
        # 生成报告
        generate_multi_model_similarity_report(similarity_df)
        
        # 保存结果
        save_multi_model_similarity_results(similarity_df, analysis_results)
        
        print("🎉 BERT多模型多轮次语义相似度分析完成！")
    else:
        print("❌ 分析失败，请检查数据文件")
