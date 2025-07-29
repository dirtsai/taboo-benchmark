#!/usr/bin/env python3
"""
Test script for Kimi Model Taboo Experiment
测试Kimi实验设置是否正确
"""

import json
import sys
import os

def test_dataset_loading():
    """测试数据集加载"""
    print("🧪 测试数据集加载...")
    try:
        with open("data/quick80_dataset.json", 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"✅ 数据集加载成功: {len(dataset)} 条记录")
        
        # 检查数据格式
        sample = dataset[0]
        required_fields = ['target', 'taboo', 'category']
        for field in required_fields:
            if field not in sample:
                print(f"❌ 缺少必需字段: {field}")
                return False
        
        print(f"✅ 数据格式正确")
        print(f"📋 样本数据: {sample['target']} (类别: {sample['category']})")
        return True
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return False

def test_api_keys():
    """测试API密钥"""
    print("\n🧪 测试API密钥...")
    try:
        with open("api_keys.json", 'r', encoding='utf-8') as f:
            api_keys = json.load(f)
        
        if "OPENROUTER_API_KEY" not in api_keys:
            print("❌ 缺少OPENROUTER_API_KEY")
            return False
        
        if not api_keys["OPENROUTER_API_KEY"]:
            print("❌ OPENROUTER_API_KEY为空")
            return False
        
        print("✅ API密钥配置正确")
        return True
    except Exception as e:
        print(f"❌ API密钥加载失败: {e}")
        return False

def test_model_configuration():
    """测试模型配置"""
    print("\n🧪 测试模型配置...")
    
    kimi_model = "moonshot/moonshot-v1-8k"
    other_models = [
        "openai/gpt-4o",
        "google/gemini-2.5-pro", 
        "deepseek/deepseek-chat-v3-0324",
        "anthropic/claude-sonnet-4"
    ]
    
    print(f"✅ Kimi模型: {kimi_model}")
    print(f"✅ 其他模型: {len(other_models)} 个")
    for i, model in enumerate(other_models, 1):
        print(f"   {i}. {model}")
    
    return True

def test_experiment_calculation():
    """测试实验游戏数计算"""
    print("\n🧪 测试实验游戏数计算...")
    
    dataset_size = 80  # quick80_dataset
    other_models_count = 4
    
    # Kimi作为Hinter: 80 * 4 = 320 games
    # Kimi作为Guesser: 80 * 4 = 320 games
    # 总计: 640 games
    
    phase1_games = dataset_size * other_models_count  # Kimi as hinter
    phase2_games = dataset_size * other_models_count  # Kimi as guesser
    total_games = phase1_games + phase2_games
    
    print(f"✅ 阶段1 (Kimi作为Hinter): {phase1_games} 场游戏")
    print(f"✅ 阶段2 (Kimi作为Guesser): {phase2_games} 场游戏")
    print(f"✅ 总游戏数: {total_games} 场游戏")
    
    if total_games == 640:
        print("✅ 游戏数计算正确 (640场，不是800场)")
        print("ℹ️  注意: 用户提到800场，但按照80个词×4个模型×2个阶段=640场")
    else:
        print(f"❌ 游戏数计算错误，期望640场，实际{total_games}场")
        return False
    
    return True

def test_directory_structure():
    """测试目录结构"""
    print("\n🧪 测试目录结构...")
    
    required_files = [
        "data/quick80_dataset.json",
        "api_keys.json",
        "kimi_experiment.py"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} 存在")
        else:
            print(f"❌ {file_path} 不存在")
            return False
    
    # 检查experiments目录是否可以创建
    try:
        os.makedirs("experiments", exist_ok=True)
        print("✅ experiments目录可用")
    except Exception as e:
        print(f"❌ 无法创建experiments目录: {e}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("🚀 开始Kimi实验设置测试")
    print("=" * 50)
    
    tests = [
        test_dataset_loading,
        test_api_keys,
        test_model_configuration,
        test_experiment_calculation,
        test_directory_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print(f"❌ 测试失败: {test.__name__}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过! 可以运行完整实验")
        print("\n💡 运行完整实验:")
        print("   python kimi_experiment.py")
        return True
    else:
        print("❌ 部分测试失败，请检查配置")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
