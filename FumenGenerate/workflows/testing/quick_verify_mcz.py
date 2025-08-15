#!/usr/bin/env python3
"""
import sys
import os
# 修复工作区重组后的导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)


快速验证脚本 - 测试MCZ处理流程是否正常
"""

import os
import sys
import numpy as np
import tempfile
import zipfile

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from core.mcz_parser import MCZParser
    from core.four_k_extractor import FourKBeatmapExtractor
    from core.audio_beatmap_analyzer import AudioBeatmapAnalyzer
    print("✅ 核心模块加载成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)

def quick_test_mcz_pipeline():
    """快速测试MCZ处理管道"""
    
    print("🔍 快速MCZ处理管道测试")
    print("=" * 50)
    
    traindata_dir = "trainData"
    mcz_files = [f for f in os.listdir(traindata_dir) if f.endswith('.mcz')][:3]
    
    parser = MCZParser()
    extractor = FourKBeatmapExtractor()
    analyzer = AudioBeatmapAnalyzer()
    
    success_count = 0
    
    for i, mcz_file in enumerate(mcz_files):
        print(f"\n[{i+1}/3] 测试: {mcz_file}")
        mcz_path = os.path.join(traindata_dir, mcz_file)
        
        try:
            # 创建临时目录
            with tempfile.TemporaryDirectory() as temp_dir:
                # 解压MCZ文件
                with zipfile.ZipFile(mcz_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                print("  ✅ MCZ解压成功")
                
                # 解析MCZ
                song_data = parser.parse_mcz_file(mcz_path)
                if not song_data:
                    print("  ❌ MCZ解析失败")
                    continue
                print(f"  ✅ MCZ解析成功: {len(song_data.mc_beatmaps)} 个MC谱面")
                
                # 提取4K谱面
                beatmaps_4k = extractor.extract_4k_beatmap(song_data)
                if not beatmaps_4k:
                    print("  ❌ 4K谱面提取失败")
                    continue
                print(f"  ✅ 4K谱面提取成功: {len(beatmaps_4k)} 个谱面")
                
                # 查找音频文件
                audio_files = []
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.lower().endswith(('.ogg', '.mp3', '.wav')):
                            audio_files.append(os.path.join(root, file))
                
                if not audio_files:
                    print("  ❌ 未找到音频文件")
                    continue
                print(f"  ✅ 找到音频文件: {len(audio_files)} 个")
                
                # 测试音频特征提取（只测试能否开始）
                try:
                    print("  🎵 开始音频特征提取...")
                    # 创建一个小的测试 - 只加载几秒钟的音频
                    import librosa
                    y, sr = librosa.load(audio_files[0], duration=5.0)  # 只加载5秒
                    print(f"  ✅ 音频加载成功: {len(y)} 样本, {sr} Hz")
                    
                    # 测试基本特征提取
                    rms = librosa.feature.rms(y=y)[0]
                    print(f"  ✅ RMS特征提取成功: {len(rms)} 帧")
                    
                    success_count += 1
                    print("  🎉 整体测试成功!")
                    
                except Exception as e:
                    print(f"  ❌ 音频处理失败: {e}")
                    continue
                    
        except Exception as e:
            print(f"  ❌ 整体测试失败: {e}")
            continue
    
    print(f"\n📊 测试结果:")
    print(f"   ✅ 成功: {success_count}/3 文件")
    print(f"   📈 成功率: {success_count/3*100:.1f}%")
    
    if success_count > 0:
        print("🎉 MCZ处理管道基本正常!")
        print("💡 建议：可以继续大规模训练，但考虑:")
        print("   - 限制音频处理时长")
        print("   - 使用更快的特征提取方法")
        print("   - 增加异常处理和超时机制")
    else:
        print("❌ MCZ处理管道存在问题，需要进一步调试")

if __name__ == "__main__":
    quick_test_mcz_pipeline()
