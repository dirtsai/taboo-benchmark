#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kimi实验结果分析脚本
分析现有的8组模型对实验结果
"""

import pandas as pd
import os
from datetime import datetime

def analyze_kimi_experiment():
    """分析Kimi实验结果"""
    print("📊 Kimi实验结果分析")
    print("=" * 50)
    
    results_dir = "results/kimi_experiment_20250717_125711"
    
    if not os.path.exists(results_dir):
        print("❌ 实验结果目录不存在")
        return
    
    # 收集所有CSV文件
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    print(f"📁 找到 {len(csv_files)} 个结果文件")
    
    # 合并所有结果
    all_results = []
    
    for csv_file in sorted(csv_files):
        file_path = os.path.join(results_dir, csv_file)
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            all_results.extend(df.to_dict('records'))
            print(f"   ✅ {csv_file}: {len(df)} 条记录")
        except Exception as e:
            print(f"   ❌ {csv_file}: 读取失败 - {e}")
    
    if not all_results:
        print("❌ 无法读取任何结果数据")
        return
    
    # 基本统计
    print(f"\n📊 基本统计:")
    total_games = len(all_results)
    total_success = sum(1 for r in all_results if r['success'])
    overall_success_rate = total_success / total_games * 100
    
    print(f"   🎮 总游戏数: {total_games:,} 场")
    print(f"   ✅ 总成功数: {total_success:,} 场")
    print(f"   📈 总成功率: {overall_success_rate:.1f}%")
    
    # 创建DataFrame用于分析
    summary_df = pd.DataFrame(all_results)
    
    # 检查缺失的实验组
    print(f"\n🔍 实验组检查:")
    expected_pairs = [
        "kimi-k2→kimi-k2", "kimi-k2→gpt-4o", "kimi-k2→gemini-2.5-flash", 
        "kimi-k2→deepseek-chat-v3-0324", "kimi-k2→claude-sonnet-4",
        "gpt-4o→kimi-k2", "gemini-2.5-flash→kimi-k2", 
        "deepseek-chat-v3-0324→kimi-k2", "claude-sonnet-4→kimi-k2"
    ]
    
    existing_pairs = summary_df['pair_name'].unique()
    missing_pairs = [pair for pair in expected_pairs if pair not in existing_pairs]
    
    print(f"   ✅ 已完成: {len(existing_pairs)}/9 组")
    print(f"   ❌ 缺失: {len(missing_pairs)} 组")
    
    if missing_pairs:
        print(f"   🚧 缺失组合: {', '.join(missing_pairs)}")
    
    # 按角色分析
    print(f"\n🎭 按角色分析:")
    kimi_as_hinter = summary_df[summary_df['hinter_name'] == 'kimi-k2']
    kimi_as_guesser = summary_df[summary_df['guesser_name'] == 'kimi-k2']
    
    if len(kimi_as_hinter) > 0:
        hinter_success = sum(kimi_as_hinter['success'])
        hinter_rate = hinter_success / len(kimi_as_hinter) * 100
        print(f"   🌙 Kimi作Hinter: {hinter_success}/{len(kimi_as_hinter)} ({hinter_rate:.1f}%)")
    
    if len(kimi_as_guesser) > 0:
        guesser_success = sum(kimi_as_guesser['success'])
        guesser_rate = guesser_success / len(kimi_as_guesser) * 100
        print(f"   🌙 Kimi作Guesser: {guesser_success}/{len(kimi_as_guesser)} ({guesser_rate:.1f}%)")
    
    # 按模型对分析
    print(f"\n👥 各模型对成功率:")
    pair_stats = summary_df.groupby('pair_name').agg({
        'success': ['count', 'sum'],
        'turns_used': 'mean'
    }).round(1)
    
    for pair_name in sorted(pair_stats.index):
        count = int(pair_stats.loc[pair_name, ('success', 'count')])
        success = int(pair_stats.loc[pair_name, ('success', 'sum')])
        rate = success / count * 100
        avg_turns = pair_stats.loc[pair_name, ('turns_used', 'mean')]
        
        # 标记Kimi的角色
        if 'kimi-k2→' in pair_name:
            role = "(🌙H)"
        elif '→kimi-k2' in pair_name:
            role = "(🌙G)"
        else:
            role = ""
        
        print(f"   {pair_name:<35} {role:<5}: {success:2d}/{count} ({rate:5.1f}%) 平均{avg_turns:.1f}轮")
    
    # 失败原因分析
    failed_df = summary_df[summary_df['success'] == False]
    if len(failed_df) > 0:
        print(f"\n📉 失败原因分析 ({len(failed_df)}场失败):")
        failure_counts = failed_df['failure_reason'].value_counts()
        
        for reason, count in failure_counts.items():
            percentage = count / len(failed_df) * 100
            reason_map = {
                'TABOO_VIOLATION': '🚫 违反禁用词',
                'FORMAT_FAILURE': '🔤 格式错误',
                'API_FAILURE': '🌐 API失败',
                'MAX_TURNS_EXCEEDED': '⏱️ 轮数耗尽',
                'EXECUTION_ERROR': '💥 执行错误'
            }
            reason_name = reason_map.get(reason, reason)
            print(f"   {reason_name}: {count} 场 ({percentage:.1f}%)")
    
    # 按词汇类别分析
    if 'category' in summary_df.columns:
        print(f"\n🏷️ 按词汇类别成功率:")
        category_stats = summary_df.groupby('category').agg({
            'success': ['count', 'sum']
        }).round(1)
        
        for category in sorted(category_stats.index):
            count = int(category_stats.loc[category, ('success', 'count')])
            success = int(category_stats.loc[category, ('success', 'sum')])
            rate = success / count * 100
            print(f"   {category:<12}: {success:3d}/{count:3d} ({rate:5.1f}%)")
    
    # 保存汇总文件
    summary_file = f"{results_dir}/kimi_experiment_summary_8groups_20250717_125711.csv"
    summary_df.to_csv(summary_file, index=False, encoding='utf-8')
    print(f"\n💾 汇总文件已保存: {os.path.basename(summary_file)}")
    
    print(f"\n📁 结果目录: {results_dir}")
    print(f"📄 当前已完成 {len(existing_pairs)}/9 组实验")
    
    if missing_pairs:
        print(f"\n⚠️  注意: 还缺少 {missing_pairs[0]} 这一组实验")
        print(f"💡 建议: 运行notebook中的恢复代码来完成最后一组实验")
    else:
        print(f"\n🎉 所有实验组合已完成！")

if __name__ == "__main__":
    analyze_kimi_experiment() 