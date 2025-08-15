#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4K谱面数据预处理和特征工程

为机器学习模型准备4K谱面训练数据，包括：
1. 音符序列编码
2. 节拍网格量化
3. 特征向量生成
4. 数据标准化
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


@dataclass
class BeatmapSequence:
    """谱面序列数据"""
    note_grid: np.ndarray    # 量化后的音符网格 [时间步, 4列]
    timing_info: np.ndarray  # 时间信息 [时间步, BPM]
    metadata: Dict[str, Any]  # 元数据


class FourKDataProcessor:
    """4K谱面数据处理器"""
    
    def __init__(self, time_resolution: int = 32):
        """
        初始化数据处理器
        
        Args:
            time_resolution: 时间量化分辨率（每个四分音符分成多少份）
        """
        self.time_resolution = time_resolution
        self.scaler = StandardScaler()
        self.processed_sequences = []
        
    def load_4k_data(self, json_file: str) -> List[Dict[str, Any]]:
        """加载4K谱面JSON数据"""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def quantize_beat_to_grid(self, beat: List[int], resolution: int) -> int:
        """
        将节拍位置量化到网格
        
        Args:
            beat: [小节, 分子, 分母] 形式的节拍
            resolution: 时间分辨率
            
        Returns:
            量化后的时间步索引
        """
        measure, numerator, denominator = beat
        # 计算在当前小节内的位置（0-1之间）
        position_in_measure = numerator / denominator
        # 转换为绝对时间步（正确公式：beat = measure + position_in_measure）
        absolute_time_step = (measure + position_in_measure) * resolution
        return int(absolute_time_step)
    
    def create_note_grid(self, notes: List[Dict], timing_points: List[Dict], 
                        max_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建音符网格和时间信息
        
        Args:
            notes: 音符列表
            timing_points: 时间轴变化点列表
            max_length: 最大长度（用于padding）
            
        Returns:
            (note_grid, timing_grid) 音符网格和时间网格
        """
        if not notes:
            return np.array([]), np.array([])
        
        # 计算所需的时间步长度
        max_time_step = 0
        for note in notes:
            time_step = self.quantize_beat_to_grid(note['beat'], self.time_resolution)
            max_time_step = max(max_time_step, time_step)
            
            # 考虑长按音符的结束时间
            if note.get('endbeat'):
                end_time_step = self.quantize_beat_to_grid(note['endbeat'], self.time_resolution)
                max_time_step = max(max_time_step, end_time_step)
        
        # 如果指定了最大长度，使用它
        if max_length is not None:
            grid_length = max_length
        else:
            grid_length = max_time_step + 1
        
        # 创建音符网格 [时间步, 4列]
        note_grid = np.zeros((grid_length, 4), dtype=np.float32)
        
        # 填充音符数据
        for note in notes:
            start_step = self.quantize_beat_to_grid(note['beat'], self.time_resolution)
            column = note['column']
            
            if 0 <= column <= 3 and start_step < grid_length:
                if note.get('endbeat'):
                    # 长按音符
                    end_step = self.quantize_beat_to_grid(note['endbeat'], self.time_resolution)
                    # 标记为长按音符（值为2.0）
                    for step in range(start_step, min(end_step + 1, grid_length)):
                        note_grid[step, column] = 2.0
                else:
                    # 普通音符（值为1.0）
                    note_grid[start_step, column] = 1.0
        
        # 创建时间网格（BPM信息）
        timing_grid = np.zeros((grid_length, 1), dtype=np.float32)
        
        # 填充BPM信息
        current_bpm = 120.0  # 默认BPM
        for i in range(grid_length):
            # 查找当前时间步对应的BPM
            current_beat = i / self.time_resolution  # 转换为节拍数（修复）
            
            for tp in timing_points:
                tp_beat = tp['beat'][0] + tp['beat'][1] / tp['beat'][2]
                if tp_beat <= current_beat:
                    current_bpm = tp['bpm']
            
            timing_grid[i, 0] = current_bpm
        
        return note_grid, timing_grid
    
    def process_beatmap_data(self, beatmap_data: Dict[str, Any], 
                           max_length: Optional[int] = None) -> BeatmapSequence:
        """
        处理单个谱面数据
        
        Args:
            beatmap_data: 谱面数据字典
            max_length: 最大序列长度
            
        Returns:
            BeatmapSequence: 处理后的序列数据
        """
        notes = beatmap_data.get('notes', [])
        timing_points = beatmap_data.get('timing_points', [])
        
        # 创建音符和时间网格
        note_grid, timing_grid = self.create_note_grid(notes, timing_points, max_length)
        
        # 提取元数据特征
        metadata = {
            'song_title': beatmap_data.get('song_title', ''),
            'artist': beatmap_data.get('artist', ''),
            'difficulty_version': beatmap_data.get('difficulty_version', ''),
            'note_count': beatmap_data.get('note_count', 0),
            'duration': beatmap_data.get('duration', 0),
            'note_density': beatmap_data.get('note_density', 0),
            'long_notes_ratio': beatmap_data.get('long_notes_ratio', 0),
            'initial_bpm': beatmap_data.get('initial_bpm', 120),
            'avg_bpm': beatmap_data.get('avg_bpm', 120),
            'bpm_changes_count': beatmap_data.get('bpm_changes_count', 0),
            'column_distribution': [
                beatmap_data.get('column_distribution', {}).get(str(i), 0) 
                for i in range(4)
            ]
        }
        
        return BeatmapSequence(
            note_grid=note_grid,
            timing_info=timing_grid,
            metadata=metadata
        )
    
    def process_dataset(self, json_file: str, max_length: Optional[int] = None) -> List[BeatmapSequence]:
        """
        处理整个数据集
        
        Args:
            json_file: 4K谱面JSON文件路径
            max_length: 最大序列长度
            
        Returns:
            处理后的序列列表
        """
        data = self.load_4k_data(json_file)
        
        print(f"处理 {len(data)} 个4K谱面...")
        
        # 如果没有指定最大长度，计算数据集中的最大长度
        if max_length is None:
            max_steps = 0
            for beatmap_data in data:
                for note in beatmap_data.get('notes', []):
                    step = self.quantize_beat_to_grid(note['beat'], self.time_resolution)
                    max_steps = max(max_steps, step)
                    if note.get('endbeat'):
                        end_step = self.quantize_beat_to_grid(note['endbeat'], self.time_resolution)
                        max_steps = max(max_steps, end_step)
            
            max_length = max_steps + 1
            print(f"计算得出最大序列长度: {max_length}")
        
        sequences = []
        for i, beatmap_data in enumerate(data):
            try:
                sequence = self.process_beatmap_data(beatmap_data, max_length)
                sequences.append(sequence)
                
                if (i + 1) % 10 == 0:
                    print(f"已处理 {i + 1}/{len(data)} 个谱面")
                    
            except Exception as e:
                print(f"处理谱面 {i} 时出错: {e}")
        
        self.processed_sequences = sequences
        print(f"成功处理 {len(sequences)} 个谱面")
        return sequences
    
    def extract_features(self, sequences: List[BeatmapSequence]) -> pd.DataFrame:
        """
        提取特征向量用于机器学习
        
        Args:
            sequences: 处理后的序列列表
            
        Returns:
            特征数据框
        """
        features = []
        
        for sequence in sequences:
            # 基本元数据特征
            feature_dict = {
                'note_count': sequence.metadata['note_count'],
                'duration': sequence.metadata['duration'],
                'note_density': sequence.metadata['note_density'],
                'long_notes_ratio': sequence.metadata['long_notes_ratio'],
                'initial_bpm': sequence.metadata['initial_bpm'],
                'avg_bpm': sequence.metadata['avg_bpm'],
                'bpm_changes_count': sequence.metadata['bpm_changes_count']
            }
            
            # 列分布特征
            col_dist = sequence.metadata['column_distribution']
            for i in range(4):
                feature_dict[f'column_{i}_notes'] = col_dist[i]
                feature_dict[f'column_{i}_ratio'] = col_dist[i] / max(sum(col_dist), 1)
            
            # 从音符网格中提取统计特征
            if sequence.note_grid.size > 0:
                note_grid = sequence.note_grid
                
                # 每列的活跃性统计
                for i in range(4):
                    column_data = note_grid[:, i]
                    feature_dict[f'column_{i}_activity'] = np.mean(column_data > 0)
                    feature_dict[f'column_{i}_intensity'] = np.mean(column_data)
                
                # 总体统计
                feature_dict['total_activity'] = np.mean(note_grid > 0)
                feature_dict['avg_notes_per_step'] = np.mean(np.sum(note_grid > 0, axis=1))
                feature_dict['max_simultaneous_notes'] = np.max(np.sum(note_grid > 0, axis=1))
                
                # 长按音符统计
                long_notes = note_grid == 2.0
                feature_dict['long_note_activity'] = np.mean(long_notes)
                
                # 节奏复杂度（相邻时间步的变化）
                if len(note_grid) > 1:
                    changes = np.sum(np.abs(np.diff(note_grid, axis=0)), axis=1)
                    feature_dict['rhythm_complexity'] = np.mean(changes)
                else:
                    feature_dict['rhythm_complexity'] = 0
            else:
                # 空谱面的默认值
                for i in range(4):
                    feature_dict[f'column_{i}_activity'] = 0
                    feature_dict[f'column_{i}_intensity'] = 0
                
                feature_dict['total_activity'] = 0
                feature_dict['avg_notes_per_step'] = 0
                feature_dict['max_simultaneous_notes'] = 0
                feature_dict['long_note_activity'] = 0
                feature_dict['rhythm_complexity'] = 0
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def save_processed_data(self, sequences: List[BeatmapSequence], 
                          output_file: str = "processed_4k_sequences.pkl"):
        """保存处理后的序列数据"""
        with open(output_file, 'wb') as f:
            pickle.dump(sequences, f)
        print(f"处理后的序列数据已保存到: {output_file}")
    
    def load_processed_data(self, input_file: str) -> List[BeatmapSequence]:
        """加载处理后的序列数据"""
        with open(input_file, 'rb') as f:
            sequences = pickle.load(f)
        print(f"从 {input_file} 加载了 {len(sequences)} 个序列")
        return sequences
    
    def create_training_arrays(self, sequences: List[BeatmapSequence]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        创建训练用的numpy数组
        
        Returns:
            (note_grids, timing_grids, metadata_features)
        """
        if not sequences:
            return np.array([]), np.array([]), np.array([])
        
        # 提取所有网格数据
        note_grids = []
        timing_grids = []
        
        for sequence in sequences:
            note_grids.append(sequence.note_grid)
            timing_grids.append(sequence.timing_info)
        
        # 转换为numpy数组
        note_grids = np.array(note_grids)
        timing_grids = np.array(timing_grids)
        
        # 提取元数据特征
        features_df = self.extract_features(sequences)
        metadata_features = features_df.values
        
        return note_grids, timing_grids, metadata_features


def main():
    """主函数 - 数据预处理示例"""
    
    # 检查是否有测试数据
    json_file = "test_4k_beatmaps.json"
    if not os.path.exists(json_file):
        print(f"请先运行 test_4k_extractor.py 生成 {json_file}")
        return
    
    # 创建数据处理器
    processor = FourKDataProcessor(time_resolution=16)  # 16分音符精度
    
    # 处理数据集
    sequences = processor.process_dataset(json_file, max_length=2000)  # 限制最大长度
    
    # 保存处理后的数据
    processor.save_processed_data(sequences)
    
    # 提取特征
    features_df = processor.extract_features(sequences)
    features_df.to_csv("processed_4k_features.csv", index=False)
    print("特征数据已保存到: processed_4k_features.csv")
    
    # 创建训练数组
    note_grids, timing_grids, metadata_features = processor.create_training_arrays(sequences)
    
    print(f"\n=== 数据预处理完成 ===")
    print(f"序列数量: {len(sequences)}")
    print(f"音符网格形状: {note_grids.shape}")
    print(f"时间网格形状: {timing_grids.shape}")
    print(f"元数据特征形状: {metadata_features.shape}")
    
    # 显示一些统计信息
    if len(sequences) > 0:
        print(f"\n样本谱面信息:")
        sample = sequences[0]
        print(f"  歌曲: {sample.metadata['song_title']}")
        print(f"  艺术家: {sample.metadata['artist']}")
        print(f"  难度: {sample.metadata['difficulty_version']}")
        print(f"  音符网格形状: {sample.note_grid.shape}")
        print(f"  非零音符步数: {np.sum(sample.note_grid > 0)}")


if __name__ == "__main__":
    main()
