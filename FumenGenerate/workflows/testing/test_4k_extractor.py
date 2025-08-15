#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试4K谱面提取器 - 只分析前几个文件
"""

import os
from four_k_extractor import FourKBeatmapExtractor

def test_4k_extractor():
    """测试4K谱面提取器"""
    data_dir = r"d:\Projects\FumenGenerate\trainData"
    
    # 获取前5个MCZ文件进行测试
    mcz_files = [f for f in os.listdir(data_dir) if f.endswith('.mcz')][:5]
    
    print(f"测试文件: {mcz_files}")
    
    # 创建提取器
    extractor = FourKBeatmapExtractor()
    
    # 逐个处理文件
    for mcz_file in mcz_files:
        mcz_path = os.path.join(data_dir, mcz_file)
        print(f"\n正在分析: {mcz_file}")
        
        try:
            mcz_data = extractor.parser.parse_mcz_file(mcz_path)
            four_k_beatmaps = extractor.extract_4k_beatmap(mcz_data)
            
            print(f"  总MC谱面数: {len(mcz_data.mc_beatmaps)}")
            print(f"  4K谱面数: {len(four_k_beatmaps)}")
            
            for i, beatmap in enumerate(mcz_data.mc_beatmaps):
                is_4k = extractor.is_4k_beatmap(beatmap)
                used_columns = set(note.column for note in beatmap.notes) if beatmap.notes else set()
                print(f"    谱面{i+1}: {beatmap.metadata.version}, 模式={beatmap.metadata.mode}, 列={used_columns}, 4K={is_4k}")
            
            extractor.four_k_beatmaps.extend(four_k_beatmaps)
            
        except Exception as e:
            print(f"  处理失败: {e}")
    
    # 分析结果
    if extractor.four_k_beatmaps:
        print(f"\n=== 测试结果总结 ===")
        extractor.analyze_4k_beatmaps()
        
        # 保存测试结果
        extractor.save_4k_beatmaps("test_4k_beatmaps.json")
        extractor.create_training_dataset("test_4k_training_data.csv")
    else:
        print("\n没有找到4K谱面")

if __name__ == "__main__":
    test_4k_extractor()
