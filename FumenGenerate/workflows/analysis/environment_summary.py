#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4K谱面学习环境总结

展示已构建的完整4K谱面分析和数据预处理环境。
"""

import os
import json
import pandas as pd
import numpy as np
import pickle
from data_processor import FourKDataProcessor

def show_environment_summary():
    """展示环境总结"""
    print("=" * 60)
    print("4K谱面智能生成系统 - 数据分析环境")
    print("=" * 60)
    
    print("\n📁 项目结构:")
    files = [
        "mcz_parser.py - MCZ文件解析器（基础解析功能）",
        "four_k_extractor.py - 4K谱面提取器（专门提取4K谱面）", 
        "data_processor.py - 数据预处理器（特征工程和序列化）",
        "batch_analyzer.py - 批量分析器（统计分析）",
        "test_4k_extractor.py - 测试脚本",
        "trainData/ - 训练数据目录（包含MCZ文件）"
    ]
    
    for file_desc in files:
        print(f"  ✓ {file_desc}")
    
    print("\n📊 数据格式理解:")
    print("  ✓ MCZ文件 = ZIP压缩包，包含:")
    print("    - 音频文件 (.ogg)")
    print("    - 图片文件 (.jpg)")
    print("    - MC谱面文件 (.mc - JSON格式)")
    print("    - TJA谱面文件 (.tja - 太鼓达人格式)")
    
    print("\n🎮 4K谱面特征:")
    print("  ✓ 游戏模式: 0（四轨道下落式）")
    print("  ✓ 使用列: 0, 1, 2, 3（四个轨道）")
    print("  ✓ 难度标识: 版本名包含'4K'")
    print("  ✓ 音符类型: 普通音符(1.0) + 长按音符(2.0)")
    
    # 检查生成的文件
    print("\n📄 已生成的数据文件:")
    generated_files = [
        ("test_4k_beatmaps.json", "4K谱面原始数据（JSON格式）"),
        ("test_4k_training_data.csv", "4K训练数据集（CSV格式）"),
        ("processed_4k_sequences.pkl", "预处理后的序列数据"),
        ("processed_4k_features.csv", "提取的特征数据")
    ]
    
    for filename, description in generated_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"  ✓ {filename} - {description} ({size:,} bytes)")
        else:
            print(f"  ✗ {filename} - {description} (未生成)")

def analyze_processed_data():
    """分析处理后的数据"""
    print("\n" + "=" * 40)
    print("数据分析结果")
    print("=" * 40)
    
    # 分析CSV数据
    if os.path.exists("test_4k_training_data.csv"):
        df = pd.read_csv("test_4k_training_data.csv")
        print(f"\n📈 训练数据集统计:")
        print(f"  • 总谱面数: {len(df)}")
        print(f"  • 独特歌曲数: {df['song_title'].nunique()}")
        print(f"  • 独特艺术家数: {df['artist'].nunique()}")
        
        print(f"\n🎵 音符统计:")
        print(f"  • 平均音符数: {df['note_count'].mean():.1f}")
        print(f"  • 音符数范围: {df['note_count'].min()} - {df['note_count'].max()}")
        print(f"  • 平均音符密度: {df['note_density'].mean():.2f} 音符/节拍")
        
        print(f"\n🎼 BPM统计:")
        print(f"  • 平均BPM: {df['initial_bpm'].mean():.1f}")
        print(f"  • BPM范围: {df['initial_bpm'].min():.1f} - {df['initial_bpm'].max():.1f}")
        
        print(f"\n🎯 难度分布:")
        diff_counts = df['difficulty_version'].value_counts()
        for diff, count in diff_counts.head(5).items():
            print(f"  • {diff}: {count}")
    
    # 分析序列数据
    if os.path.exists("processed_4k_sequences.pkl"):
        print(f"\n🔧 预处理数据:")
        try:
            processor = FourKDataProcessor()
            sequences = processor.load_processed_data("processed_4k_sequences.pkl")
        except Exception as e:
            print(f"  ✗ 无法加载序列数据: {e}")
            sequences = []
        
        print(f"  • 序列数量: {len(sequences)}")
        if sequences:
            sample = sequences[0]
            print(f"  • 序列形状: {sample.note_grid.shape}")
            print(f"  • 时间分辨率: 每节拍16步")
            print(f"  • 最大序列长度: 2000步")
            
            # 计算一些统计信息
            total_notes = sum(np.sum(seq.note_grid > 0) for seq in sequences)
            total_steps = sum(seq.note_grid.shape[0] for seq in sequences)
            print(f"  • 总音符数: {total_notes:,}")
            print(f"  • 总时间步数: {total_steps:,}")
            print(f"  • 平均密度: {total_notes/total_steps:.4f}")

def show_next_steps():
    """显示下一步建议"""
    print("\n" + "=" * 40)
    print("下一步开发建议")
    print("=" * 40)
    
    steps = [
        "🚀 机器学习模型开发:",
        "   • 使用LSTM/GRU处理序列数据",
        "   • 使用Transformer处理音符序列",
        "   • 考虑VAE生成变分自编码器",
        "",
        "🎵 音频特征提取:",
        "   • 提取音频的梅尔频谱图",
        "   • 节拍检测和节奏分析",
        "   • 音调和和弦分析",
        "",
        "📊 数据增强:",
        "   • 处理更多MCZ文件（当前只有15个样本）",
        "   • 数据平衡（不同难度的样本数量）",
        "   • 交叉验证数据分割",
        "",
        "🔄 模型训练:",
        "   • 音乐特征 → 谱面生成",
        "   • 条件生成（指定难度、风格）",
        "   • 序列到序列学习"
    ]
    
    for step in steps:
        print(step)

def create_sample_usage():
    """创建使用示例"""
    print("\n" + "=" * 40) 
    print("使用示例")
    print("=" * 40)
    
    print("\n💡 如何使用这个环境:")
    
    usage_examples = [
        "# 1. 提取4K谱面",
        "from four_k_extractor import FourKBeatmapExtractor",
        "extractor = FourKBeatmapExtractor()",
        "beatmaps = extractor.extract_from_directory('trainData')",
        "",
        "# 2. 数据预处理", 
        "from data_processor import FourKDataProcessor",
        "processor = FourKDataProcessor(time_resolution=16)",
        "sequences = processor.process_dataset('four_k_beatmaps.json')",
        "",
        "# 3. 获取训练数据",
        "note_grids, timing_grids, features = processor.create_training_arrays(sequences)",
        "# note_grids.shape: (n_samples, max_length, 4)",
        "# timing_grids.shape: (n_samples, max_length, 1)", 
        "# features.shape: (n_samples, n_features)",
        "",
        "# 4. 现在可以用于机器学习模型训练！"
    ]
    
    for line in usage_examples:
        print(line)

def main():
    """主函数"""
    show_environment_summary()
    analyze_processed_data()
    show_next_steps()
    create_sample_usage()
    
    print("\n" + "=" * 60)
    print("🎉 4K谱面学习环境构建完成！")
    print("现在可以开始训练音乐到谱面的生成模型了。")
    print("=" * 60)

if __name__ == "__main__":
    main()
