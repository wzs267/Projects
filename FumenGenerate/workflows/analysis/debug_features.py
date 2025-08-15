#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试特征向量长度问题
"""

import os
import numpy as np
import librosa
from audio_beatmap_analyzer import AudioBeatmapAnalyzer

def debug_feature_extraction():
    """调试特征提取过程"""
    analyzer = AudioBeatmapAnalyzer(time_resolution=0.05)
    
    # 测试音频文件
    audio_file = "extracted_audio/_song_10088_Kawaki wo Ameku.ogg"
    
    if not os.path.exists(audio_file):
        print(f"音频文件不存在: {audio_file}")
        return
    
    print("提取音频特征...")
    audio_features = analyzer.extract_audio_features(audio_file)
    
    print(f"时间帧数量: {len(audio_features.time_frames)}")
    print(f"RMS能量形状: {audio_features.rms_energy.shape}")
    print(f"频谱质心形状: {audio_features.spectral_centroid.shape}")
    print(f"过零率形状: {audio_features.zero_crossing_rate.shape}")
    print(f"MFCC形状: {audio_features.mfcc.shape}")
    print(f"色度形状: {audio_features.chroma.shape}")
    print(f"音符起始强度形状: {audio_features.onset_strength.shape}")
    
    # 检查特征向量长度一致性
    feature_lengths = []
    for i in range(min(10, len(audio_features.time_frames))):
        feature_vector = [
            audio_features.rms_energy[i] if i < len(audio_features.rms_energy) else 0.0,
            audio_features.spectral_centroid[i] if i < len(audio_features.spectral_centroid) else 0.0,
            audio_features.zero_crossing_rate[i] if i < len(audio_features.zero_crossing_rate) else 0.0,
            audio_features.onset_strength[i] if i < len(audio_features.onset_strength) else 0.0,
        ]
        
        # 添加MFCC特征
        for j in range(5):
            if j < audio_features.mfcc.shape[0] and i < audio_features.mfcc.shape[1]:
                feature_vector.append(audio_features.mfcc[j, i])
            else:
                feature_vector.append(0.0)
        
        # 添加色度特征
        if i < audio_features.chroma.shape[1]:
            chroma_mean = np.mean(audio_features.chroma[:, i])
            feature_vector.append(chroma_mean)
        else:
            feature_vector.append(0.0)
        
        # 节拍和起始邻近性
        t = audio_features.time_frames[i] if i < len(audio_features.time_frames) else 0.0
        beat_proximity = min([abs(t - bt) for bt in audio_features.beat_times], default=float('inf'))
        is_near_beat = 1.0 if beat_proximity < 0.1 else 0.0
        feature_vector.append(is_near_beat)
        
        onset_proximity = min([abs(t - ot) for ot in audio_features.onset_times], default=float('inf'))
        is_near_onset = 1.0 if onset_proximity < 0.1 else 0.0
        feature_vector.append(is_near_onset)
        
        # 难度参数
        feature_vector.extend([0.5, 0.1, 0.5])  # 测试值
        
        feature_lengths.append(len(feature_vector))
        print(f"时间步{i}: 特征向量长度={len(feature_vector)}, 内容={[type(x).__name__ for x in feature_vector]}")
        
        # 检查是否有嵌套列表
        for j, val in enumerate(feature_vector):
            if hasattr(val, '__len__') and not isinstance(val, (str, bytes)):
                print(f"  警告：特征{j}是序列类型: {type(val)}, 值={val}")
    
    print(f"\n特征向量长度: {set(feature_lengths)}")
    if len(set(feature_lengths)) == 1:
        print("✓ 所有特征向量长度一致")
    else:
        print("✗ 特征向量长度不一致")

if __name__ == "__main__":
    debug_feature_extraction()
