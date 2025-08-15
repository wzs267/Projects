#!/usr/bin/env python3
"""
import sys
import os
# 修复工作区重组后的导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)


调试MCZ处理问题
"""

import os
import sys
import traceback

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from core.mcz_parser import MCZParser
    from core.four_k_extractor import FourKBeatmapExtractor  
    from core.audio_beatmap_analyzer import AudioBeatmapAnalyzer
    import numpy as np
    
    def debug_mcz_file(mcz_file: str):
        """调试单个MCZ文件处理"""
        print(f"\n🔍 调试文件: {mcz_file}")
        
        traindata_dir = "trainData"
        mcz_path = os.path.join(traindata_dir, mcz_file)
        
        parser = MCZParser()
        extractor = FourKBeatmapExtractor()
        analyzer = AudioBeatmapAnalyzer()
        
        try:
            # 步骤1: 解析MCZ
            print("  📋 步骤1: 解析MCZ文件...")
            song_data = parser.parse_mcz_file(mcz_path)
            if not song_data:
                print("  ❌ MCZ解析失败：无数据返回")
                return False
            print(f"  ✅ MCZ解析成功，包含 {len(song_data.mc_beatmaps)} 个MC谱面，{len(song_data.tja_beatmaps)} 个TJA谱面")
            
            # 步骤2: 提取4K谱面
            print("  🎯 步骤2: 提取4K谱面...")
            beatmaps_4k = extractor.extract_4k_beatmap(song_data)
            if not beatmaps_4k:
                print("  ❌ 4K谱面提取失败：无4K谱面")
                return False
            print(f"  ✅ 找到 {len(beatmaps_4k)} 个4K谱面")
            
            # 步骤3: 获取音频文件
            print("  🎵 步骤3: 获取音频文件...")
            audio_files = song_data.audio_files
            if not audio_files:
                print("  ❌ 音频获取失败：无音频文件")
                return False
            print(f"  ✅ 找到 {len(audio_files)} 个音频文件")
            
            # 步骤4: 音频谱面对齐
            print("  🔗 步骤4: 音频谱面对齐...")
            audio_file = audio_files[0]
            beatmap = beatmaps_4k[0]
            
            # 提取音频特征
            audio_features = analyzer.extract_audio_features(audio_file)
            
            # 转换谱面为事件格式
            beatmap_dict = {
                'notes': beatmap.notes,
                'timing_points': beatmap.timing_points,
                'metadata': beatmap.metadata
            }
            beatmap_events = analyzer.extract_beatmap_events(beatmap_dict)
            
            # 对齐音频和谱面
            aligned_data = analyzer.align_audio_beatmap(
                audio_features, beatmap_events, {}
            )
            
            if not aligned_data:
                print("  ❌ 音频谱面对齐失败")
                return False
                
            print(f"  ✅ 对齐成功，生成 {len(aligned_data.audio_features)} 个特征帧")
            print(f"      音频特征维度: {aligned_data.audio_features.shape}")
            print(f"      谱面事件维度: {aligned_data.beatmap_events.shape}")
            return True
                    
        except Exception as e:
            print(f"  ❌ 处理异常: {e}")
            traceback.print_exc()
            return False
    
    def main():
        print("🔍 MCZ处理调试工具")
        print("=" * 50)
        
        # 测试前5个文件
        traindata_dir = "trainData"
        mcz_files = [f for f in os.listdir(traindata_dir) if f.endswith('.mcz')][:5]
        
        success_count = 0
        
        for mcz_file in mcz_files:
            if debug_mcz_file(mcz_file):
                success_count += 1
        
        print(f"\n📊 调试结果: {success_count}/{len(mcz_files)} 文件处理成功")
        
        if success_count == 0:
            print("❌ 所有文件都处理失败，需要进一步调查")
        else:
            print(f"✅ {success_count} 个文件处理成功")

    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    traceback.print_exc()
