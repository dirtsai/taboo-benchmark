# Kimi模型实验 - 使用base_test框架
# 这个代码可以直接添加到base_test copy.ipynb作为新的代码单元

# 1. 加载Quick80数据集
print("📚 加载Quick80数据集...")
with open("quick80_dataset.json", 'r', encoding='utf-8') as f:
    quick80_dataset = json.load(f)
print(f"✅ Quick80数据集加载完成，共{len(quick80_dataset)}条记录")

# 2. 配置Kimi实验模型 - 包含所有5个模型的完整对战
print("\n🤖 配置Kimi实验模型...")

# 所有模型列表（包括Kimi）
ALL_MODELS = [
    "moonshotai/kimi-k2",  # Kimi模型
    "openai/gpt-4o",
    "google/gemini-2.5-pro", 
    "deepseek/deepseek-chat-v3-0324",
    "anthropic/claude-sonnet-4"
]

KIMI_MODEL = "moonshotai/kimi-k2"

print(f"🌙 Kimi模型: {KIMI_MODEL}")
print(f"🤖 所有参与模型: {len(ALL_MODELS)} 个")
for i, model in enumerate(ALL_MODELS, 1):
    model_name = model.split('/')[-1]
    if model == KIMI_MODEL:
        print(f"   {i}. {model_name} ⭐ (Kimi)")
    else:
        print(f"   {i}. {model_name}")

# 其他4个模型（除了Kimi）
OTHER_MODELS = [model for model in ALL_MODELS if model != KIMI_MODEL]

print(f"\n📊 实验设计:")
print(f"   • Kimi作为Hinter vs 其他4个模型作为Guesser: 4种组合")
print(f"   • Kimi作为Guesser vs 其他4个模型作为Hinter: 4种组合")
print(f"   • Kimi vs Kimi 自对战: 1种组合")
print(f"   • 总计: 4 + 4 + 1 = 9种独特组合")
print(f"   • 总游戏数: {len(quick80_dataset)} × 9 = {len(quick80_dataset) * 9}")

# 3. 分阶段运行Kimi实验
print(f"\n🚀 开始运行Kimi实验...")
print(f"💡 使用base_test框架，分3个阶段运行")

all_kimi_results = []

# 阶段1: Kimi作为Hinter vs 其他4个模型作为Guesser
print(f"\n🎯 阶段1: Kimi作为Hinter vs 其他模型作为Guesser")
for i, guesser_model in enumerate(OTHER_MODELS, 1):
    print(f"\n🔄 运行 {i}/4: Kimi(Hinter) vs {guesser_model.split('/')[-1]}(Guesser)")
    
    # 使用简单模式，手动运行每个词汇的游戏
    phase1_results = []
    
    for j, word_data in enumerate(quick80_dataset, 1):
        if j % 20 == 0 or j == 1:
            print(f"   📝 处理词汇 {j}/{len(quick80_dataset)}: {word_data['target']}")
        
        try:
            game_result = enhanced_play_taboo_game(
                client=client,
                hinter_model=KIMI_MODEL,
                guesser_model=guesser_model,
                target_word=word_data['target'],
                taboo_words=word_data['taboo'],
                max_turns=5
            )
            
            # 添加额外信息
            game_result.update({
                'word_index': j-1,
                'category': word_data.get('category', 'unknown'),
                'experiment_phase': 'kimi_as_hinter'
            })
            
            phase1_results.append(game_result)
            
        except Exception as e:
            print(f"   ❌ 词汇 {word_data['target']} 游戏失败: {str(e)}")
            # 添加失败记录
            phase1_results.append({
                'target_word': word_data['target'],
                'hinter_model': KIMI_MODEL,
                'guesser_model': guesser_model,
                'success': False,
                'turns': 0,
                'error': str(e),
                'word_index': j-1,
                'category': word_data.get('category', 'unknown'),
                'experiment_phase': 'kimi_as_hinter'
            })
    
    # 转换为DataFrame并保存
    if phase1_results:
        phase1_df = pd.DataFrame(phase1_results)
        all_kimi_results.append(phase1_df)
        
        success_rate = phase1_df['success'].mean() * 100
        success_count = phase1_df['success'].sum()
        total_count = len(phase1_df)
        
        print(f"   ✅ 完成 {total_count} 场游戏，成功率: {success_rate:.1f}% ({success_count}/{total_count})")
        
        # 保存中间结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = f"results/kimi_hinter_vs_{guesser_model.split('/')[-1]}_{timestamp}.csv"
        os.makedirs('results', exist_ok=True)
        phase1_df.to_csv(temp_path, index=False, encoding='utf-8')
        print(f"   💾 中间结果已保存: {temp_path}")
    else:
        print(f"   ❌ 阶段失败，无结果")

# 阶段2: Kimi作为Guesser vs 其他4个模型作为Hinter
print(f"\n🎯 阶段2: 其他模型作为Hinter vs Kimi作为Guesser")
for i, hinter_model in enumerate(OTHER_MODELS, 1):
    print(f"\n🔄 运行 {i}/4: {hinter_model.split('/')[-1]}(Hinter) vs Kimi(Guesser)")
    
    # 使用简单模式，手动运行每个词汇的游戏
    phase2_results = []
    
    for j, word_data in enumerate(quick80_dataset, 1):
        if j % 20 == 0 or j == 1:
            print(f"   📝 处理词汇 {j}/{len(quick80_dataset)}: {word_data['target']}")
        
        try:
            game_result = enhanced_play_taboo_game(
                client=client,
                hinter_model=hinter_model,
                guesser_model=KIMI_MODEL,
                target_word=word_data['target'],
                taboo_words=word_data['taboo'],
                max_turns=5
            )
            
            # 添加额外信息
            game_result.update({
                'word_index': j-1,
                'category': word_data.get('category', 'unknown'),
                'experiment_phase': 'kimi_as_guesser'
            })
            
            phase2_results.append(game_result)
            
        except Exception as e:
            print(f"   ❌ 词汇 {word_data['target']} 游戏失败: {str(e)}")
            # 添加失败记录
            phase2_results.append({
                'target_word': word_data['target'],
                'hinter_model': hinter_model,
                'guesser_model': KIMI_MODEL,
                'success': False,
                'turns': 0,
                'error': str(e),
                'word_index': j-1,
                'category': word_data.get('category', 'unknown'),
                'experiment_phase': 'kimi_as_guesser'
            })
    
    # 转换为DataFrame并保存
    if phase2_results:
        phase2_df = pd.DataFrame(phase2_results)
        all_kimi_results.append(phase2_df)
        
        success_rate = phase2_df['success'].mean() * 100
        success_count = phase2_df['success'].sum()
        total_count = len(phase2_df)
        
        print(f"   ✅ 完成 {total_count} 场游戏，成功率: {success_rate:.1f}% ({success_count}/{total_count})")
        
        # 保存中间结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = f"results/{hinter_model.split('/')[-1]}_hinter_vs_kimi_{timestamp}.csv"
        os.makedirs('results', exist_ok=True)
        phase2_df.to_csv(temp_path, index=False, encoding='utf-8')
        print(f"   💾 中间结果已保存: {temp_path}")
    else:
        print(f"   ❌ 阶段失败，无结果")

# 阶段3: Kimi vs Kimi 自对战
print(f"\n🎯 阶段3: Kimi vs Kimi 自对战")
print(f"\n🔄 运行: Kimi(Hinter) vs Kimi(Guesser)")

# 使用简单模式，手动运行每个词汇的游戏
phase3_results = []

for j, word_data in enumerate(quick80_dataset, 1):
    if j % 20 == 0 or j == 1:
        print(f"   📝 处理词汇 {j}/{len(quick80_dataset)}: {word_data['target']}")
    
    try:
        game_result = enhanced_play_taboo_game(
            client=client,
            hinter_model=KIMI_MODEL,
            guesser_model=KIMI_MODEL,
            target_word=word_data['target'],
            taboo_words=word_data['taboo'],
            max_turns=5
        )
        
        # 添加额外信息
        game_result.update({
            'word_index': j-1,
            'category': word_data.get('category', 'unknown'),
            'experiment_phase': 'kimi_vs_kimi'
        })
        
        phase3_results.append(game_result)
        
    except Exception as e:
        print(f"   ❌ 词汇 {word_data['target']} 游戏失败: {str(e)}")
        # 添加失败记录
        phase3_results.append({
            'target_word': word_data['target'],
            'hinter_model': KIMI_MODEL,
            'guesser_model': KIMI_MODEL,
            'success': False,
            'turns': 0,
            'error': str(e),
            'word_index': j-1,
            'category': word_data.get('category', 'unknown'),
            'experiment_phase': 'kimi_vs_kimi'
        })

# 转换为DataFrame并保存
if phase3_results:
    phase3_df = pd.DataFrame(phase3_results)
    all_kimi_results.append(phase3_df)
    
    success_rate = phase3_df['success'].mean() * 100
    success_count = phase3_df['success'].sum()
    total_count = len(phase3_df)
    
    print(f"   ✅ 完成 {total_count} 场游戏，成功率: {success_rate:.1f}% ({success_count}/{total_count})")
    
    # 保存中间结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_path = f"results/kimi_vs_kimi_self_play_{timestamp}.csv"
    os.makedirs('results', exist_ok=True)
    phase3_df.to_csv(temp_path, index=False, encoding='utf-8')
    print(f"   💾 中间结果已保存: {temp_path}")
else:
    print(f"   ❌ 阶段失败，无结果")

# 合并所有结果
if all_kimi_results:
    all_results = pd.concat(all_kimi_results, ignore_index=True)

if len(all_kimi_results) > 0:
    print(f"\n🎉 Kimi实验成功完成!")
    print(f"📊 总游戏数: {len(all_results)}")
    print(f"✅ 整体成功率: {all_results['success'].mean() * 100:.1f}%")
    
    # 分析Kimi的表现
    print(f"\n🌙 Kimi模型详细分析:")
    print(f"=" * 50)
    
    # Kimi作为Hinter的表现
    kimi_as_hinter = all_results[all_results['hinter_model'] == KIMI_MODEL]
    print(f"\n🎯 Kimi作为Hinter:")
    print(f"   总游戏数: {len(kimi_as_hinter)}")
    print(f"   成功率: {kimi_as_hinter['success'].mean() * 100:.1f}%")
    print(f"   成功/总数: {kimi_as_hinter['success'].sum()}/{len(kimi_as_hinter)}")
    
    print(f"   按Guesser模型分析:")
    for model in ALL_MODELS:
        model_results = kimi_as_hinter[kimi_as_hinter['guesser_model'] == model]
        if len(model_results) > 0:
            success_rate = model_results['success'].mean() * 100
            model_name = model.split('/')[-1]
            if model == KIMI_MODEL:
                print(f"     vs {model_name} (自己): {success_rate:.1f}% ({model_results['success'].sum()}/{len(model_results)})")
            else:
                print(f"     vs {model_name}: {success_rate:.1f}% ({model_results['success'].sum()}/{len(model_results)})")
    
    # Kimi作为Guesser的表现
    kimi_as_guesser = all_results[all_results['guesser_model'] == KIMI_MODEL]
    print(f"\n🎯 Kimi作为Guesser:")
    print(f"   总游戏数: {len(kimi_as_guesser)}")
    print(f"   成功率: {kimi_as_guesser['success'].mean() * 100:.1f}%")
    print(f"   成功/总数: {kimi_as_guesser['success'].sum()}/{len(kimi_as_guesser)}")
    
    print(f"   按Hinter模型分析:")
    for model in ALL_MODELS:
        model_results = kimi_as_guesser[kimi_as_guesser['hinter_model'] == model]
        if len(model_results) > 0:
            success_rate = model_results['success'].mean() * 100
            model_name = model.split('/')[-1]
            if model == KIMI_MODEL:
                print(f"     vs {model_name} (自己): {success_rate:.1f}% ({model_results['success'].sum()}/{len(model_results)})")
            else:
                print(f"     vs {model_name}: {success_rate:.1f}% ({model_results['success'].sum()}/{len(model_results)})")
    
    kimi_related_results = all_results[
        (all_results['hinter_model'] == KIMI_MODEL) | 
        (all_results['guesser_model'] == KIMI_MODEL)
    ]
    
    # 保存Kimi相关结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    kimi_results_path = f"results/kimi_experiment_results_{timestamp}.csv"
    kimi_related_results.to_csv(kimi_results_path, index=False, encoding='utf-8')
    
    # 保存完整实验结果
    full_results_path = f"results/full_5x5_experiment_results_{timestamp}.csv"
    all_results.to_csv(full_results_path, index=False, encoding='utf-8')
    
    print(f"\n💾 结果保存:")
    print(f"   Kimi相关结果: {kimi_results_path}")
    print(f"   完整实验结果: {full_results_path}")
    
    # 生成Kimi实验的详细分析
    print(f"\n📊 Kimi实验详细分析:")
    print(f"=" * 60)
    
    # 分析各个阶段的结果
    print(f"\n📈 分阶段结果:")
    
    # 阶段1: Kimi作为Hinter
    kimi_hinter_results = all_results[all_results['hinter_model'] == KIMI_MODEL]
    kimi_hinter_vs_others = kimi_hinter_results[kimi_hinter_results['guesser_model'] != KIMI_MODEL]
    
    if len(kimi_hinter_vs_others) > 0:
        print(f"\n🎯 阶段1 - Kimi作为Hinter vs 其他模型:")
        print(f"   总游戏数: {len(kimi_hinter_vs_others)}")
        print(f"   成功率: {kimi_hinter_vs_others['success'].mean() * 100:.1f}%")
        print(f"   成功/总数: {kimi_hinter_vs_others['success'].sum()}/{len(kimi_hinter_vs_others)}")
        
        print(f"   按对手分析:")
        for model in OTHER_MODELS:
            model_results = kimi_hinter_vs_others[kimi_hinter_vs_others['guesser_model'] == model]
            if len(model_results) > 0:
                success_rate = model_results['success'].mean() * 100
                model_name = model.split('/')[-1]
                print(f"     vs {model_name}: {success_rate:.1f}% ({model_results['success'].sum()}/{len(model_results)})")
    
    # 阶段2: Kimi作为Guesser
    kimi_guesser_results = all_results[all_results['guesser_model'] == KIMI_MODEL]
    kimi_guesser_vs_others = kimi_guesser_results[kimi_guesser_results['hinter_model'] != KIMI_MODEL]
    
    if len(kimi_guesser_vs_others) > 0:
        print(f"\n🎯 阶段2 - Kimi作为Guesser vs 其他模型:")
        print(f"   总游戏数: {len(kimi_guesser_vs_others)}")
        print(f"   成功率: {kimi_guesser_vs_others['success'].mean() * 100:.1f}%")
        print(f"   成功/总数: {kimi_guesser_vs_others['success'].sum()}/{len(kimi_guesser_vs_others)}")
        
        print(f"   按对手分析:")
        for model in OTHER_MODELS:
            model_results = kimi_guesser_vs_others[kimi_guesser_vs_others['hinter_model'] == model]
            if len(model_results) > 0:
                success_rate = model_results['success'].mean() * 100
                model_name = model.split('/')[-1]
                print(f"     vs {model_name}: {success_rate:.1f}% ({model_results['success'].sum()}/{len(model_results)})")
    
    # 特别关注Kimi vs Kimi的结果
    kimi_vs_kimi = all_results[
        (all_results['hinter_model'] == KIMI_MODEL) & 
        (all_results['guesser_model'] == KIMI_MODEL)
    ]
    
    if len(kimi_vs_kimi) > 0:
        print(f"\n🌙 特别关注 - Kimi vs Kimi 自对战:")
        print(f"   游戏数: {len(kimi_vs_kimi)}")
        print(f"   成功率: {kimi_vs_kimi['success'].mean() * 100:.1f}%")
        print(f"   成功/总数: {kimi_vs_kimi['success'].sum()}/{len(kimi_vs_kimi)}")
        if kimi_vs_kimi['success'].any():
            avg_turns = kimi_vs_kimi[kimi_vs_kimi['success']]['turns'].mean()
            print(f"   平均成功轮次: {avg_turns:.1f}")
    
    print(f"\n🎉 Kimi实验完成!")
    print(f"📊 实验包含了 9 种独特的模型组合")
    print(f"🌙 Kimi模型参与了 {len(all_results)} 场游戏")
    print(f"💡 这是一个针对Kimi的专门实验，包括自对战")

else:
    print("❌ 实验失败，请检查配置和网络连接")
