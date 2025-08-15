#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4K谱面专用解析器

专门用于提取和分析MCZ文件中的4K谱面（四轨道竖直下落式）数据。
这些谱面将用于训练音乐到谱面的生成模型。
"""

import os
import json
import zipfile
import tempfile
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from .mcz_parser import MCZParser, MCZFile, MCBeatmap, Note, TimingPoint


@dataclass
class FourKBeatmap:
    """4K谱面数据结构"""
    # 基本信息
    song_title: str
    artist: str
    song_id: int
    creator: str
    difficulty_version: str
    
    # 音频和图片文件
    audio_files: List[str]
    image_files: List[str]
    
    # 谱面数据
    notes: List[Note]
    timing_points: List[TimingPoint]
    
    # 谱面特征
    note_count: int
    duration: float  # 谱面时长（节拍）
    note_density: float  # 音符密度
    column_distribution: Dict[int, int]  # 每列音符数量分布
    long_notes_count: int
    long_notes_ratio: float
    
    # BPM信息
    initial_bpm: float
    bpm_changes: List[Tuple[float, float]]  # (时间点, BPM值)
    avg_bpm: float
    
    # 原始文件信息
    source_mcz_file: str


class FourKBeatmapExtractor:
    """4K谱面提取器"""
    
    def __init__(self):
        self.parser = MCZParser()
        self.four_k_beatmaps = []
    
    def is_4k_beatmap(self, beatmap: MCBeatmap) -> bool:
        """判断是否为4K谱面"""
        # 检查难度版本名称是否包含"4K"
        if "4K" in beatmap.metadata.version:
            return True
        
        # 检查游戏模式是否为0（通常4K模式）
        if beatmap.metadata.mode == 0:
            return True
        
        # 检查使用的列数是否为4列且列号为0-3
        if beatmap.notes:
            used_columns = set(note.column for note in beatmap.notes)
            if used_columns.issubset({0, 1, 2, 3}) and len(used_columns) >= 3:
                return True
        
        return False
    
    def extract_4k_beatmap(self, mcz_data: MCZFile) -> List[FourKBeatmap]:
        """从MCZ数据中提取4K谱面"""
        four_k_beatmaps = []
        
        for beatmap in mcz_data.mc_beatmaps:
            if self.is_4k_beatmap(beatmap):
                four_k_data = self._convert_to_4k_beatmap(beatmap, mcz_data)
                four_k_beatmaps.append(four_k_data)
        
        return four_k_beatmaps
    
    def _convert_to_4k_beatmap(self, beatmap: MCBeatmap, mcz_data: MCZFile) -> FourKBeatmap:
        """将MC谱面转换为4K谱面数据结构"""
        # 计算谱面特征
        notes = beatmap.notes
        column_distribution = {}
        for note in notes:
            col = note.column
            column_distribution[col] = column_distribution.get(col, 0) + 1
        
        # 计算时长和密度
        if notes:
            beat_positions = [note.beat[0] + note.beat[1]/note.beat[2] for note in notes]
            duration = max(beat_positions) - min(beat_positions) if len(beat_positions) >= 2 else 0
            note_density = len(notes) / duration if duration > 0 else 0
        else:
            duration = 0
            note_density = 0
        
        # 计算长按音符
        long_notes_count = sum(1 for note in notes if note.endbeat is not None)
        long_notes_ratio = long_notes_count / len(notes) if notes else 0
        
        # 计算BPM信息
        bpm_changes = []
        if beatmap.timing_points:
            for tp in beatmap.timing_points:
                beat_time = tp.beat[0] + tp.beat[1]/tp.beat[2]
                bpm_changes.append((beat_time, tp.bpm))
            
            initial_bpm = beatmap.timing_points[0].bpm
            avg_bpm = sum(tp.bpm for tp in beatmap.timing_points) / len(beatmap.timing_points)
        else:
            initial_bpm = 0
            avg_bpm = 0
        
        return FourKBeatmap(
            song_title=beatmap.metadata.song_info.title,
            artist=beatmap.metadata.song_info.artist,
            song_id=beatmap.metadata.song_info.song_id,
            creator=beatmap.metadata.creator,
            difficulty_version=beatmap.metadata.version,
            audio_files=mcz_data.audio_files,
            image_files=mcz_data.image_files,
            notes=notes,
            timing_points=beatmap.timing_points,
            note_count=len(notes),
            duration=duration,
            note_density=note_density,
            column_distribution=column_distribution,
            long_notes_count=long_notes_count,
            long_notes_ratio=long_notes_ratio,
            initial_bpm=initial_bpm,
            bpm_changes=bpm_changes,
            avg_bpm=avg_bpm,
            source_mcz_file=mcz_data.file_path
        )
    
    def extract_from_directory(self, data_dir: str) -> List[FourKBeatmap]:
        """从目录中批量提取4K谱面"""
        mcz_files = [f for f in os.listdir(data_dir) if f.endswith('.mcz')]
        
        print(f"发现 {len(mcz_files)} 个MCZ文件")
        four_k_count = 0
        
        for mcz_file in tqdm(mcz_files, desc="提取4K谱面"):
            mcz_path = os.path.join(data_dir, mcz_file)
            try:
                mcz_data = self.parser.parse_mcz_file(mcz_path)
                four_k_beatmaps = self.extract_4k_beatmap(mcz_data)
                self.four_k_beatmaps.extend(four_k_beatmaps)
                four_k_count += len(four_k_beatmaps)
                
                if len(four_k_beatmaps) > 0:
                    print(f"  {mcz_file}: 找到 {len(four_k_beatmaps)} 个4K谱面")
                    
            except Exception as e:
                print(f"处理失败 {mcz_file}: {e}")
        
        print(f"\n总共提取到 {four_k_count} 个4K谱面")
        return self.four_k_beatmaps
    
    def save_4k_beatmaps(self, output_file: str = "four_k_beatmaps.json"):
        """保存4K谱面数据到JSON文件"""
        if not self.four_k_beatmaps:
            print("没有4K谱面数据可保存")
            return
        
        # 转换为可JSON序列化的格式
        serializable_data = []
        for beatmap in self.four_k_beatmaps:
            data = {
                'song_title': beatmap.song_title,
                'artist': beatmap.artist,
                'song_id': beatmap.song_id,
                'creator': beatmap.creator,
                'difficulty_version': beatmap.difficulty_version,
                'audio_files': [os.path.basename(f) for f in beatmap.audio_files],
                'image_files': [os.path.basename(f) for f in beatmap.image_files],
                'note_count': beatmap.note_count,
                'duration': beatmap.duration,
                'note_density': beatmap.note_density,
                'column_distribution': beatmap.column_distribution,
                'long_notes_count': beatmap.long_notes_count,
                'long_notes_ratio': beatmap.long_notes_ratio,
                'initial_bpm': beatmap.initial_bpm,
                'avg_bpm': beatmap.avg_bpm,
                'bpm_changes_count': len(beatmap.bpm_changes),
                'source_mcz_file': os.path.basename(beatmap.source_mcz_file),
                # 保存音符数据
                'notes': [
                    {
                        'beat': note.beat,
                        'column': note.column,
                        'endbeat': note.endbeat
                    } for note in beatmap.notes
                ],
                # 保存时间轴数据
                'timing_points': [
                    {
                        'beat': tp.beat,
                        'bpm': tp.bpm
                    } for tp in beatmap.timing_points
                ]
            }
            serializable_data.append(data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        
        print(f"4K谱面数据已保存到: {output_file}")
    
    def create_training_dataset(self, output_csv: str = "four_k_training_data.csv"):
        """创建训练数据集CSV文件"""
        if not self.four_k_beatmaps:
            print("没有4K谱面数据可创建训练集")
            return
        
        training_data = []
        for beatmap in self.four_k_beatmaps:
            row = {
                'song_title': beatmap.song_title,
                'artist': beatmap.artist,
                'song_id': beatmap.song_id,
                'creator': beatmap.creator,
                'difficulty_version': beatmap.difficulty_version,
                'note_count': beatmap.note_count,
                'duration': beatmap.duration,
                'note_density': beatmap.note_density,
                'long_notes_ratio': beatmap.long_notes_ratio,
                'initial_bpm': beatmap.initial_bpm,
                'avg_bpm': beatmap.avg_bpm,
                'bpm_changes_count': len(beatmap.bpm_changes),
                'column_0_notes': beatmap.column_distribution.get(0, 0),
                'column_1_notes': beatmap.column_distribution.get(1, 0),
                'column_2_notes': beatmap.column_distribution.get(2, 0),
                'column_3_notes': beatmap.column_distribution.get(3, 0),
                'source_mcz_file': os.path.basename(beatmap.source_mcz_file),
                'has_audio': len(beatmap.audio_files) > 0,
                'has_image': len(beatmap.image_files) > 0
            }
            training_data.append(row)
        
        df = pd.DataFrame(training_data)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"训练数据集已保存到: {output_csv}")
        return df
    
    def analyze_4k_beatmaps(self):
        """分析4K谱面的统计信息"""
        if not self.four_k_beatmaps:
            print("没有4K谱面数据可分析")
            return
        
        print("=== 4K谱面分析报告 ===")
        print(f"总4K谱面数: {len(self.four_k_beatmaps)}")
        
        # 基本统计
        note_counts = [b.note_count for b in self.four_k_beatmaps]
        durations = [b.duration for b in self.four_k_beatmaps if b.duration > 0]
        densities = [b.note_density for b in self.four_k_beatmaps if b.note_density > 0]
        
        print(f"\n音符数统计:")
        print(f"  平均: {sum(note_counts)/len(note_counts):.1f}")
        print(f"  范围: {min(note_counts)} - {max(note_counts)}")
        
        if durations:
            print(f"\n谱面时长统计:")
            print(f"  平均: {sum(durations)/len(durations):.1f} 节拍")
            print(f"  范围: {min(durations):.1f} - {max(durations):.1f} 节拍")
        
        if densities:
            print(f"\n音符密度统计:")
            print(f"  平均: {sum(densities)/len(densities):.2f} 音符/节拍")
            print(f"  范围: {min(densities):.2f} - {max(densities):.2f} 音符/节拍")
        
        # BPM统计
        bpms = [b.initial_bpm for b in self.four_k_beatmaps if b.initial_bpm > 0]
        if bpms:
            print(f"\nBPM统计:")
            print(f"  平均: {sum(bpms)/len(bpms):.1f}")
            print(f"  范围: {min(bpms):.1f} - {max(bpms):.1f}")
        
        # 难度分布
        difficulties = {}
        for beatmap in self.four_k_beatmaps:
            diff = beatmap.difficulty_version
            difficulties[diff] = difficulties.get(diff, 0) + 1
        
        print(f"\n难度分布:")
        for diff, count in sorted(difficulties.items(), key=lambda x: x[1], reverse=True):
            print(f"  {diff}: {count}")
        
        # 艺术家分布
        artists = {}
        for beatmap in self.four_k_beatmaps:
            artist = beatmap.artist
            artists[artist] = artists.get(artist, 0) + 1
        
        print(f"\n艺术家TOP10:")
        for artist, count in sorted(artists.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {artist}: {count}")


def main():
    """主函数 - 提取4K谱面"""
    data_dir = r"d:\Projects\FumenGenerate\trainData"
    
    # 创建4K谱面提取器
    extractor = FourKBeatmapExtractor()
    
    # 从目录中提取4K谱面
    print("开始提取4K谱面...")
    four_k_beatmaps = extractor.extract_from_directory(data_dir)
    
    # 分析统计信息
    extractor.analyze_4k_beatmaps()
    
    # 保存数据
    extractor.save_4k_beatmaps()
    extractor.create_training_dataset()
    
    print("\n4K谱面提取完成！")


if __name__ == "__main__":
    main()
