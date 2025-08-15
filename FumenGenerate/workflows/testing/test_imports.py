#!/usr/bin/env python3
"""
import sys
import os
# 修复工作区重组后的导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)


测试models目录导入
"""

import sys
import os

# 确保路径正确
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试各个模块的导入"""
    print("🔍 测试模块导入...")
    
    # 测试核心模块
    try:
        from core.mcz_parser import MCZParser
        print("✅ core.mcz_parser 导入成功")
    except Exception as e:
        print(f"❌ core.mcz_parser 导入失败: {e}")
    
    try:
        from core.four_k_extractor import FourKBeatmapExtractor
        print("✅ core.four_k_extractor 导入成功")
    except Exception as e:
        print(f"❌ core.four_k_extractor 导入失败: {e}")
    
    try:
        from core.audio_beatmap_analyzer import AudioBeatmapAnalyzer
        print("✅ core.audio_beatmap_analyzer 导入成功")
    except Exception as e:
        print(f"❌ core.audio_beatmap_analyzer 导入失败: {e}")
    
    # 测试models模块
    try:
        from models.beatmap_learning_system import BeatmapLearningSystem
        print("✅ models.beatmap_learning_system 导入成功")
    except Exception as e:
        print(f"❌ models.beatmap_learning_system 导入失败: {e}")
    
    try:
        from models.hybrid_beatmap_system import HybridBeatmapLearningSystem
        print("✅ models.hybrid_beatmap_system 导入成功")
    except Exception as e:
        print(f"❌ models.hybrid_beatmap_system 导入失败: {e}")
    
    try:
        from models.deep_learning_beatmap_system import DeepBeatmapLearningSystem
        print("✅ models.deep_learning_beatmap_system 导入成功")
    except Exception as e:
        print(f"❌ models.deep_learning_beatmap_system 导入失败: {e}")

if __name__ == "__main__":
    test_imports()
    print("\n🎯 导入测试完成！")
