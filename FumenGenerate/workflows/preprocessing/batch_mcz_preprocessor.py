#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量MCZ预处理脚本
将所有MCZ文件预处理为小批量训练能使用的标准格式
"""

import os
import sys
import json
import shutil
import tempfile
import zipfile
from typing import List, Dict, Any
from tqdm import tqdm

# 修复工作区重组后的导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from core.mcz_parser import MCZParser
from core.four_k_extractor import FourKBeatmapExtractor

class MCZBatchPreprocessor:
    """MCZ批量预处理器"""
    
    def __init__(self, output_dir: str = "preprocessed_data"):
        self.traindata_dir = "trainData"
        self.output_dir = output_dir
        self.audio_dir = os.path.join(output_dir, "audio")
        self.beatmaps_file = os.path.join(output_dir, "all_4k_beatmaps.json")
        
        # 初始化组件
        self.parser = MCZParser()
        self.extractor = FourKBeatmapExtractor()
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        
    def process_single_mcz(self, mcz_file: str) -> List[Dict[str, Any]]:
        """处理单个MCZ文件，返回标准格式的谱面列表"""
        mcz_path = os.path.join(self.traindata_dir, mcz_file)
        results = []
        
        print(f"🔍 开始处理: {mcz_file}")
        
        temp_dir = None
        try:
            # 创建临时目录
            temp_dir = tempfile.mkdtemp(prefix="mcz_preprocess_")
            print(f"   📁 临时目录: {temp_dir}")
            
            # 解压MCZ
            with zipfile.ZipFile(mcz_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            print(f"   ✅ 解压完成")
            
            # 解析MCZ
            song_data = self.parser.parse_mcz_file(mcz_path)
            if not song_data:
                print(f"   ❌ MCZ解析失败")
                return results
            print(f"   ✅ MCZ解析成功")
            
            # 提取4K谱面
            beatmaps_4k = self.extractor.extract_4k_beatmap(song_data)
            if not beatmaps_4k:
                print(f"   ❌ 4K谱面提取失败")
                return results
            print(f"   ✅ 找到 {len(beatmaps_4k)} 个4K谱面")
            
            # 查找并复制音频文件
            audio_files = []
            print(f"   🎵 查找音频文件...")
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith(('.ogg', '.mp3', '.wav')):
                        src_path = os.path.join(root, file)
                        # 生成唯一的音频文件名
                        song_id = os.path.splitext(mcz_file)[0]
                        audio_name = f"{song_id}_{file}"
                        dst_path = os.path.join(self.audio_dir, audio_name)
                        shutil.copy2(src_path, dst_path)
                        audio_files.append(audio_name)
                        print(f"     ✅ 复制音频: {file} -> {audio_name}")
            
            if not audio_files:
                print(f"   ❌ 未找到音频文件")
                return results
            print(f"   ✅ 找到 {len(audio_files)} 个音频文件")
            
            # 转换每个谱面为标准格式
            print(f"   🎯 转换谱面格式...")
            for i, beatmap in enumerate(beatmaps_4k):
                print(f"     处理谱面 {i+1}/{len(beatmaps_4k)}: {beatmap.difficulty_version}")
                try:
                    # 使用beatmap对象的属性（已经包含了计算好的统计信息）
                    song_title = beatmap.song_title
                    artist = beatmap.artist
                    creator = beatmap.creator
                    difficulty_version = beatmap.difficulty_version
                    
                    # 转换notes为字典格式
                    notes = []
                    for note in beatmap.notes:
                        endbeat = getattr(note, 'endbeat', None)
                        if endbeat is None:
                            endbeat = note.beat
                        
                        notes.append({
                            'beat': note.beat,
                            'column': note.column,
                            'endbeat': endbeat,
                            'is_long': endbeat > note.beat if endbeat is not None else False
                        })
                    
                    # 转换timing_points为字典格式
                    timing_points = []
                    for tp in beatmap.timing_points:
                        timing_points.append({
                            'beat': tp.beat,
                            'bpm': tp.bpm
                        })
                    
                    # 构建标准格式（使用beatmap对象中已计算的统计信息）
                    beatmap_dict = {
                        'title': song_title,
                        'artist': artist,
                        'creator': creator,
                        'difficulty_name': difficulty_version,
                        'difficulty_level': 0,  # FourKBeatmap中没有level字段
                        'audio_file': audio_files[0],  # 使用第一个音频文件
                        'notes': notes,
                        'timing_points': timing_points,
                        'note_count': beatmap.note_count,
                        'note_density': beatmap.note_density,
                        'long_notes_ratio': beatmap.long_notes_ratio,
                        'avg_bpm': beatmap.avg_bpm,
                        'duration': beatmap.duration,
                        'initial_bpm': beatmap.initial_bpm,
                        'source_mcz': mcz_file
                    }
                    
                    results.append(beatmap_dict)
                    print(f"     ✅ 谱面转换成功: {song_title} - {difficulty_version}")
                    
                except Exception as e:
                    # 跳过有问题的谱面
                    print(f"     ❌ 谱面转换失败: {e}")
                    continue
            
        except Exception as e:
            # 记录错误信息用于调试
            print(f"❌ 处理 {mcz_file} 失败: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 清理临时目录
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
        
        return results
    
    def process_all_mcz_files(self, max_files: int = None) -> List[Dict[str, Any]]:
        """处理所有MCZ文件"""
        if not os.path.exists(self.traindata_dir):
            raise FileNotFoundError(f"训练数据目录不存在: {self.traindata_dir}")
        
        mcz_files = [f for f in os.listdir(self.traindata_dir) if f.endswith('.mcz')]
        if max_files:
            mcz_files = mcz_files[:max_files]
        
        print(f"🚀 开始处理 {len(mcz_files)} 个MCZ文件...")
        
        all_beatmaps = []
        success_count = 0
        
        for mcz_file in tqdm(mcz_files, desc="处理MCZ文件"):
            beatmaps = self.process_single_mcz(mcz_file)
            if beatmaps:
                all_beatmaps.extend(beatmaps)
                success_count += 1
        
        print(f"✅ 预处理完成:")
        print(f"   📁 处理文件: {len(mcz_files)}")
        print(f"   ✅ 成功文件: {success_count}")
        print(f"   🎯 总谱面数: {len(all_beatmaps)}")
        print(f"   📊 成功率: {success_count/len(mcz_files)*100:.1f}%")
        
        return all_beatmaps
    
    def save_beatmaps(self, beatmaps: List[Dict[str, Any]]):
        """保存谱面数据"""
        with open(self.beatmaps_file, 'w', encoding='utf-8') as f:
            json.dump(beatmaps, f, ensure_ascii=False, indent=2)
        
        print(f"💾 谱面数据已保存: {self.beatmaps_file}")
        print(f"🎵 音频文件目录: {self.audio_dir}")

def main():
    """主函数"""
    print("🎮 MCZ批量预处理工具")
    print("=" * 50)
    print("📝 将MCZ文件预处理为小批量训练格式")
    print()
    
    # 创建预处理器
    preprocessor = MCZBatchPreprocessor()
    
    # 处理所有MCZ文件（限制前100个进行测试）
    beatmaps = preprocessor.process_all_mcz_files(max_files=None)

    if beatmaps:
        # 保存结果
        preprocessor.save_beatmaps(beatmaps)
        
        print(f"\n🎉 预处理完成！")
        print(f"📊 数据统计:")
        print(f"   🎯 总谱面数: {len(beatmaps)}")
        print(f"   🎵 音频文件数: {len(os.listdir(preprocessor.audio_dir))}")
        print(f"   📁 数据文件: {preprocessor.beatmaps_file}")
        print()
        print(f"🚀 现在可以使用以下命令进行大规模训练:")
        print(f"   python large_scale_train_with_preprocessed.py")
    else:
        print("❌ 没有成功预处理任何谱面数据")

if __name__ == "__main__":
    main()
