#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音游谱面生成：基于音频特征的学习方案设计

核心思路：
1. 音频特征提取：分贝变化、节拍检测、频谱分析
2. 谱面映射学习：音频事件 → 击打事件的对应关系
3. 难度参数控制：通过难度级别调节音符密度和复杂度
4. 时间对齐：确保音频特征与谱面时间轴精确对应

学习目标：
- 学习何时应该放置音符（基于音频突变点）
- 学习长条音符的放置（基于音频持续特征）
- 学习不同轨道的分配策略
- 学习难度与音符密度的关系
"""

import os
import numpy as np
import librosa
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import json
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class AudioFeatures:
    """音频特征数据结构"""
    # 基础音频信息
    sr: int                         # 采样率
    duration: float                 # 时长(秒)
    
    # 幅度和能量特征
    rms_energy: np.ndarray         # RMS能量(分贝变化的主要指标)
    spectral_centroid: np.ndarray  # 频谱质心
    zero_crossing_rate: np.ndarray # 过零率
    
    # 节拍和节奏特征
    tempo: float                   # BPM
    beat_times: np.ndarray        # 节拍时间点
    onset_times: np.ndarray       # 音符起始时间点
    onset_strength: np.ndarray    # 音符起始强度
    
    # 频谱特征
    mfcc: np.ndarray              # MFCC特征
    chroma: np.ndarray            # 色度特征
    
    # 时间轴（用于对齐谱面）
    time_frames: np.ndarray       # 时间帧


@dataclass
class BeatmapEvent:
    """谱面事件数据结构"""
    time: float                   # 事件时间(秒)
    event_type: str              # 事件类型: 'note', 'long_start', 'long_end'
    column: int                  # 轨道(0-3)
    intensity: float             # 强度(0-1)
    duration: Optional[float]    # 持续时间(长条音符)


@dataclass
class AlignedData:
    """对齐后的音频-谱面数据"""
    time_grid: np.ndarray        # 统一时间网格
    audio_features: np.ndarray   # 对应的音频特征矩阵
    beatmap_events: np.ndarray   # 对应的谱面事件矩阵
    difficulty_params: Dict[str, float]  # 难度参数


class AudioBeatmapAnalyzer:
    """音频-谱面分析器"""
    
    def __init__(self, time_resolution: float = 0.01):
        """
        初始化分析器
        
        Args:
            time_resolution: 时间分辨率(秒)，即每个时间步的长度
        """
        self.time_resolution = time_resolution
        self.scaler = MinMaxScaler()
    
    def extract_audio_features(self, audio_path: str) -> AudioFeatures:
        """
        提取音频特征
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            AudioFeatures: 提取的音频特征
        """
        print(f"正在提取音频特征: {os.path.basename(audio_path)}")
        
        # 加载音频
        y, sr = librosa.load(audio_path, sr=None)
        duration = len(y) / sr
        
        # 基础特征提取
        hop_length = int(sr * self.time_resolution)  # 每个时间步对应的样本数
        
        # 1. RMS能量（分贝变化的关键指标）
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # 2. 频谱特征
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
        
        # 3. 节拍检测
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        # 4. 音符起始检测
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        
        # 5. 高级特征
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        
        # 时间轴
        time_frames = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        return AudioFeatures(
            sr=sr,
            duration=duration,
            rms_energy=rms_db,
            spectral_centroid=spectral_centroid,
            zero_crossing_rate=zero_crossing_rate,
            tempo=tempo,
            beat_times=beat_times,
            onset_times=onset_times,
            onset_strength=onset_strength,
            mfcc=mfcc,
            chroma=chroma,
            time_frames=time_frames
        )
    
    def extract_beatmap_events(self, beatmap_data: Dict[str, Any]) -> List[BeatmapEvent]:
        """
        从谱面数据中提取事件
        
        Args:
            beatmap_data: 谱面数据字典
            
        Returns:
            List[BeatmapEvent]: 谱面事件列表
        """
        events = []
        notes = beatmap_data.get('notes', [])
        timing_points = beatmap_data.get('timing_points', [])
        
        # 构建BPM时间轴
        bpm_timeline = {}
        for tp in timing_points:
            beat_time = tp['beat'][0] + tp['beat'][1] / tp['beat'][2]
            bpm_timeline[beat_time] = tp['bpm']
        
        # 转换音符为事件
        for note in notes:
            # 计算音符的绝对时间
            beat_pos = note['beat'][0] + note['beat'][1] / note['beat'][2]
            
            # 根据BPM计算实际时间（简化版本，假设4/4拍）
            # 实际应该根据BPM变化精确计算
            current_bpm = 120  # 默认BPM
            for beat_time in sorted(bpm_timeline.keys()):
                if beat_time <= beat_pos:
                    current_bpm = bpm_timeline[beat_time]
            
            # 转换为秒
            seconds_per_beat = 60.0 / current_bpm
            time_in_seconds = beat_pos * seconds_per_beat
            
            column = note['column']
            
            if note.get('endbeat'):
                # 长条音符
                end_beat_pos = note['endbeat'][0] + note['endbeat'][1] / note['endbeat'][2]
                end_time_in_seconds = end_beat_pos * seconds_per_beat
                duration = end_time_in_seconds - time_in_seconds
                
                # 长条开始事件
                events.append(BeatmapEvent(
                    time=time_in_seconds,
                    event_type='long_start',
                    column=column,
                    intensity=1.0,
                    duration=duration
                ))
                
                # 长条结束事件
                events.append(BeatmapEvent(
                    time=end_time_in_seconds,
                    event_type='long_end',
                    column=column,
                    intensity=0.5,
                    duration=None
                ))
            else:
                # 普通音符
                events.append(BeatmapEvent(
                    time=time_in_seconds,
                    event_type='note',
                    column=column,
                    intensity=1.0,
                    duration=None
                ))
        
        return sorted(events, key=lambda x: x.time)
    
    def align_audio_beatmap(self, audio_features: AudioFeatures, 
                          beatmap_events: List[BeatmapEvent],
                          difficulty_params: Dict[str, float]) -> AlignedData:
        """
        对齐音频特征和谱面事件到统一时间网格
        
        Args:
            audio_features: 音频特征
            beatmap_events: 谱面事件
            difficulty_params: 难度参数
            
        Returns:
            AlignedData: 对齐后的数据
        """
        # 创建统一时间网格
        max_time = min(audio_features.duration, 
                      max([event.time for event in beatmap_events]) + 5 if beatmap_events else audio_features.duration)
        time_grid = np.arange(0, max_time, self.time_resolution)
        
        # 构建音频特征矩阵
        audio_matrix = []
        
        for t in time_grid:
            # 找到最接近的音频帧
            frame_idx = np.argmin(np.abs(audio_features.time_frames - t))
            
            # 构建特征向量
            feature_vector = [
                audio_features.rms_energy[frame_idx],                    # RMS能量(分贝)
                audio_features.spectral_centroid[frame_idx],             # 频谱质心
                audio_features.zero_crossing_rate[frame_idx],            # 过零率
                np.interp(t, audio_features.time_frames, 
                         audio_features.onset_strength),                 # 音符起始强度
            ]
            
            # 添加MFCC特征（前5个系数）
            mfcc_features = [audio_features.mfcc[i, frame_idx] for i in range(min(5, audio_features.mfcc.shape[0]))]
            feature_vector.extend(mfcc_features)
            
            # 添加色度特征（平均值）
            chroma_mean = np.mean(audio_features.chroma[:, frame_idx])
            feature_vector.append(chroma_mean)
            
            # 检查是否靠近节拍点
            beat_proximity = min([abs(t - bt) for bt in audio_features.beat_times], default=float('inf'))
            is_near_beat = 1.0 if beat_proximity < self.time_resolution * 2 else 0.0
            feature_vector.append(is_near_beat)
            
            # 检查是否靠近音符起始点
            onset_proximity = min([abs(t - ot) for ot in audio_features.onset_times], default=float('inf'))
            is_near_onset = 1.0 if onset_proximity < self.time_resolution * 2 else 0.0
            feature_vector.append(is_near_onset)
            
            audio_matrix.append(feature_vector)
        
        audio_matrix = np.array(audio_matrix)
        
        # 构建谱面事件矩阵 [时间步, 4列 + 事件类型]
        # 列0-3: 各轨道的事件强度
        # 列4-6: 事件类型 (note, long_start, long_end)
        beatmap_matrix = np.zeros((len(time_grid), 7))
        
        for event in beatmap_events:
            # 找到最接近的时间步
            time_idx = np.argmin(np.abs(time_grid - event.time))
            
            if 0 <= event.column <= 3:
                # 设置轨道强度
                beatmap_matrix[time_idx, event.column] = event.intensity
                
                # 设置事件类型
                if event.event_type == 'note':
                    beatmap_matrix[time_idx, 4] = 1.0
                elif event.event_type == 'long_start':
                    beatmap_matrix[time_idx, 5] = 1.0
                    # 填充长条持续时间
                    if event.duration:
                        end_idx = min(len(time_grid) - 1, 
                                     time_idx + int(event.duration / self.time_resolution))
                        beatmap_matrix[time_idx:end_idx, event.column] = 0.5  # 长条持续标记
                elif event.event_type == 'long_end':
                    beatmap_matrix[time_idx, 6] = 1.0
        
        return AlignedData(
            time_grid=time_grid,
            audio_features=audio_matrix,
            beatmap_events=beatmap_matrix,
            difficulty_params=difficulty_params
        )
    
    def analyze_beatmap_patterns(self, aligned_data: AlignedData) -> Dict[str, Any]:
        """
        分析谱面模式，为学习提供洞察
        
        Args:
            aligned_data: 对齐后的数据
            
        Returns:
            分析结果字典
        """
        audio_features = aligned_data.audio_features
        beatmap_events = aligned_data.beatmap_events
        
        analysis = {}
        
        # 1. 音频特征与音符放置的相关性分析
        rms_energy = audio_features[:, 0]  # RMS能量
        onset_strength = audio_features[:, 3]  # 音符起始强度
        note_activity = np.sum(beatmap_events[:, :4], axis=1)  # 总音符活动
        
        # 计算相关性
        rms_note_corr = np.corrcoef(rms_energy, note_activity)[0, 1]
        onset_note_corr = np.corrcoef(onset_strength, note_activity)[0, 1]
        
        analysis['correlations'] = {
            'rms_energy_note': rms_note_corr,
            'onset_strength_note': onset_note_corr
        }
        
        # 2. 音符密度分析
        note_density = np.mean(note_activity > 0)
        long_note_ratio = np.sum(beatmap_events[:, 5]) / max(np.sum(beatmap_events[:, 4:7]), 1)
        
        analysis['note_statistics'] = {
            'note_density': note_density,
            'long_note_ratio': long_note_ratio,
            'avg_simultaneous_notes': np.mean(np.sum(beatmap_events[:, :4] > 0, axis=1))
        }
        
        # 3. 轨道使用模式
        column_usage = [np.sum(beatmap_events[:, i] > 0) for i in range(4)]
        analysis['column_patterns'] = {
            f'column_{i}_usage': usage for i, usage in enumerate(column_usage)
        }
        
        # 4. 难度特征
        difficulty = aligned_data.difficulty_params
        analysis['difficulty_features'] = difficulty
        
        return analysis
    
    def visualize_alignment(self, aligned_data: AlignedData, 
                          start_time: float = 0, duration: float = 30,
                          save_path: Optional[str] = None):
        """
        可视化音频-谱面对齐结果
        
        Args:
            aligned_data: 对齐后的数据
            start_time: 开始时间(秒)
            duration: 显示时长(秒)
            save_path: 保存路径
        """
        time_grid = aligned_data.time_grid
        audio_features = aligned_data.audio_features
        beatmap_events = aligned_data.beatmap_events
        
        # 选择时间范围
        start_idx = int(start_time / self.time_resolution)
        end_idx = int((start_time + duration) / self.time_resolution)
        
        t = time_grid[start_idx:end_idx]
        audio_slice = audio_features[start_idx:end_idx]
        beatmap_slice = beatmap_events[start_idx:end_idx]
        
        # 创建图表
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        fig.suptitle(f'音频-谱面对齐分析 ({start_time:.1f}s - {start_time+duration:.1f}s)', fontsize=14)
        
        # 1. RMS能量和音符起始强度
        ax1 = axes[0]
        ax1.plot(t, audio_slice[:, 0], label='RMS能量(dB)', alpha=0.7)
        ax1.plot(t, audio_slice[:, 3], label='音符起始强度', alpha=0.7)
        ax1.set_ylabel('强度')
        ax1.set_title('音频特征')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 四个轨道的音符事件
        ax2 = axes[1]
        colors = ['red', 'green', 'blue', 'orange']
        for i in range(4):
            note_positions = np.where(beatmap_slice[:, i] > 0)[0]
            if len(note_positions) > 0:
                ax2.scatter(t[note_positions], [i] * len(note_positions), 
                           c=colors[i], s=50, alpha=0.8, label=f'轨道{i}')
        ax2.set_ylabel('轨道')
        ax2.set_title('音符放置')
        ax2.set_ylim(-0.5, 3.5)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 事件类型
        ax3 = axes[2]
        note_events = np.where(beatmap_slice[:, 4] > 0)[0]
        long_start_events = np.where(beatmap_slice[:, 5] > 0)[0]
        long_end_events = np.where(beatmap_slice[:, 6] > 0)[0]
        
        if len(note_events) > 0:
            ax3.scatter(t[note_events], [0] * len(note_events), 
                       c='red', s=30, alpha=0.8, label='普通音符')
        if len(long_start_events) > 0:
            ax3.scatter(t[long_start_events], [1] * len(long_start_events), 
                       c='green', s=30, alpha=0.8, label='长条开始')
        if len(long_end_events) > 0:
            ax3.scatter(t[long_end_events], [2] * len(long_end_events), 
                       c='blue', s=30, alpha=0.8, label='长条结束')
        
        ax3.set_ylabel('事件类型')
        ax3.set_title('谱面事件类型')
        ax3.set_ylim(-0.5, 2.5)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 音频-谱面关联
        ax4 = axes[3]
        total_note_activity = np.sum(beatmap_slice[:, :4], axis=1)
        ax4_twin = ax4.twinx()
        
        ax4.plot(t, audio_slice[:, 0], 'b-', alpha=0.6, label='RMS能量')
        ax4_twin.plot(t, total_note_activity, 'r-', alpha=0.8, label='音符活动')
        
        ax4.set_ylabel('RMS能量(dB)', color='b')
        ax4_twin.set_ylabel('音符活动', color='r')
        ax4.set_xlabel('时间(秒)')
        ax4.set_title('音频特征与谱面活动对比')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可视化图表已保存到: {save_path}")
        
        plt.show()


def main():
    """主函数 - 演示音频-谱面分析"""
    print("=== 音游谱面生成：音频-谱面对齐分析 ===")
    
    # 检查测试数据
    if not os.path.exists("test_4k_beatmaps.json"):
        print("请先运行 test_4k_extractor.py 生成测试数据")
        return
    
    # 加载谱面数据
    with open("test_4k_beatmaps.json", 'r', encoding='utf-8') as f:
        beatmaps_data = json.load(f)
    
    # 创建分析器
    analyzer = AudioBeatmapAnalyzer(time_resolution=0.05)  # 50ms分辨率
    
    # 分析第一个谱面（如果有对应的音频文件）
    if beatmaps_data:
        beatmap = beatmaps_data[0]
        print(f"\n分析谱面: {beatmap['song_title']} - {beatmap['difficulty_version']}")
        
        # 查找对应的音频文件
        audio_files = beatmap.get('audio_files', [])
        if audio_files:
            # 这里需要实际的音频文件路径，目前仅作演示
            print(f"音频文件: {audio_files}")
            print("注意：需要实际的音频文件来进行完整分析")
            
            # 提取谱面事件
            beatmap_events = analyzer.extract_beatmap_events(beatmap)
            print(f"提取到 {len(beatmap_events)} 个谱面事件")
            
            # 显示前几个事件
            for i, event in enumerate(beatmap_events[:5]):
                print(f"  事件{i+1}: t={event.time:.2f}s, 类型={event.event_type}, 轨道={event.column}")
            
            # 分析难度参数
            difficulty_params = {
                'note_count': beatmap['note_count'],
                'note_density': beatmap['note_density'],
                'long_notes_ratio': beatmap['long_notes_ratio'],
                'avg_bpm': beatmap['avg_bpm']
            }
            
            print(f"\n难度参数: {difficulty_params}")
        else:
            print("未找到音频文件")
    
    print("\n=== 分析框架已准备就绪 ===")
    print("下一步：")
    print("1. 添加实际音频文件路径")
    print("2. 完善BPM时间转换算法")
    print("3. 实现机器学习模型")
    print("4. 设计难度控制机制")


if __name__ == "__main__":
    main()
