#!/usr/bin/env python3
"""
import sys
import os
# 修复工作区重组后的导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)


调试MCZ对象结构
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.mcz_parser import MCZParser
from core.four_k_extractor import FourKBeatmapExtractor

def debug_mcz_structure():
    """调试MCZ对象结构"""
    parser = MCZParser()
    extractor = FourKBeatmapExtractor()
    
    mcz_file = "_song_10088.mcz"
    mcz_path = os.path.join("trainData", mcz_file)
    
    print(f"🔍 调试MCZ结构: {mcz_file}")
    
    try:
        # 解析MCZ
        song_data = parser.parse_mcz_file(mcz_path)
        print(f"✅ song_data类型: {type(song_data)}")
        print(f"✅ song_data属性: {dir(song_data)}")
        
        if hasattr(song_data, 'title'):
            print(f"   标题: {song_data.title}")
        if hasattr(song_data, 'artist'):
            print(f"   艺术家: {song_data.artist}")
        if hasattr(song_data, 'mc_beatmaps'):
            print(f"   MC谱面数: {len(song_data.mc_beatmaps)}")
        if hasattr(song_data, 'tja_beatmaps'):
            print(f"   TJA谱面数: {len(song_data.tja_beatmaps)}")
        if hasattr(song_data, 'audio_files'):
            print(f"   音频文件数: {len(song_data.audio_files)}")
        
        # 提取4K谱面
        beatmaps_4k = extractor.extract_4k_beatmap(song_data)
        print(f"✅ 4K谱面数: {len(beatmaps_4k)}")
        
        if beatmaps_4k:
            beatmap = beatmaps_4k[0]
            print(f"✅ 第一个谱面类型: {type(beatmap)}")
            print(f"✅ 谱面属性: {dir(beatmap)}")
            
            if hasattr(beatmap, 'difficulty_name'):
                print(f"   难度名: {beatmap.difficulty_name}")
            if hasattr(beatmap, 'difficulty_level'):
                print(f"   难度级别: {beatmap.difficulty_level}")
            if hasattr(beatmap, 'notes'):
                print(f"   音符数: {len(beatmap.notes)}")
                if beatmap.notes:
                    note = beatmap.notes[0]
                    print(f"   第一个音符类型: {type(note)}")
                    print(f"   音符属性: {dir(note)}")
            if hasattr(beatmap, 'timing_points'):
                print(f"   时间点数: {len(beatmap.timing_points)}")
                if beatmap.timing_points:
                    tp = beatmap.timing_points[0]
                    print(f"   第一个时间点类型: {type(tp)}")
                    print(f"   时间点属性: {dir(tp)}")
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_mcz_structure()
