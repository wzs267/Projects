#!/usr/bin/env python3
"""
分析音游谱面生成中的音频特征提取方法
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import zipfile
import json
from pathlib import Path

def analyze_audio_features_for_beatmap():
    """分析音频特征与谱面生成的关系"""
    
    # 提取一个标准MCZ的音频文件
    mcz_path = "trainData/_song_4833.mcz"
    
    try:
        with zipfile.ZipFile(mcz_path, 'r') as mcz:
            audio_files = [f for f in mcz.namelist() if f.endswith('.ogg')]
            if not audio_files:
                print("❌ 未找到音频文件")
                return
            
            # 提取音频到临时位置
            import tempfile
            temp_dir = tempfile.mkdtemp()
            audio_file = audio_files[0]
            mcz.extract(audio_file, temp_dir)
            audio_path = os.path.join(temp_dir, audio_file)
            
            print(f"🎵 分析音频文件: {audio_file}")
            
            # 加载音频
            y, sr = librosa.load(audio_path, sr=22050)
            duration = len(y) / sr
            
            print(f"📊 音频基本信息:")
            print(f"   采样率: {sr} Hz")
            print(f"   时长: {duration:.2f} 秒")
            print(f"   样本数: {len(y)}")
            
            # 1. 音量/能量特征分析
            print(f"\n🔊 音量/能量特征分析:")
            
            # RMS能量 (Root Mean Square)
            hop_length = 512
            frame_length = 2048
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # 时间轴
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
            
            print(f"   RMS能量范围: {np.min(rms):.4f} - {np.max(rms):.4f}")
            print(f"   RMS能量均值: {np.mean(rms):.4f}")
            print(f"   RMS能量标准差: {np.std(rms):.4f}")
            
            # 2. 音调/频谱特征分析
            print(f"\n🎼 音调/频谱特征分析:")
            
            # 频谱质心 (Spectral Centroid) - 频谱重心
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            print(f"   频谱质心范围: {np.min(spectral_centroids):.1f} - {np.max(spectral_centroids):.1f} Hz")
            print(f"   频谱质心均值: {np.mean(spectral_centroids):.1f} Hz")
            
            # 色度特征 (Chroma) - 12个半音的强度
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            print(f"   色度特征形状: {chroma.shape}")
            print(f"   主要音调: {np.argmax(np.mean(chroma, axis=1))} (0=C, 1=C#, ...)")
            
            # MFCC (Mel-frequency cepstral coefficients) - 音色特征
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            print(f"   MFCC特征形状: {mfccs.shape}")
            
            # 3. 节拍/时间特征分析
            print(f"\n🥁 节拍/时间特征分析:")
            
            # 节拍跟踪
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beats, sr=sr)
            
            print(f"   检测到的BPM: {tempo:.2f}")
            print(f"   检测到的节拍数: {len(beats)}")
            print(f"   平均节拍间隔: {np.mean(np.diff(beat_times)):.3f} 秒")
            
            # 4. 音频变化检测
            print(f"\n🔄 音频变化检测:")
            
            # 频谱对比度 (Spectral Contrast)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            print(f"   频谱对比度形状: {contrast.shape}")
            
            # 过零率 (Zero Crossing Rate) - 音频信号的变化率
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            print(f"   过零率范围: {np.min(zcr):.4f} - {np.max(zcr):.4f}")
            print(f"   过零率均值: {np.mean(zcr):.4f}")
            
            # 5. 音频事件检测 (Onset Detection)
            print(f"\n🎯 音频事件检测:")
            
            # 音符起始检测
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            print(f"   检测到的音符起始数: {len(onset_times)}")
            print(f"   前10个起始时间: {onset_times[:10]}")
            
            # 音符起始强度
            onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
            print(f"   起始强度范围: {np.min(onset_strength):.4f} - {np.max(onset_strength):.4f}")
            
            return {
                'rms': rms,
                'times': times,
                'spectral_centroids': spectral_centroids,
                'chroma': chroma,
                'mfccs': mfccs,
                'tempo': tempo,
                'beat_times': beat_times,
                'onset_times': onset_times,
                'onset_strength': onset_strength,
                'zcr': zcr,
                'duration': duration,
                'sr': sr
            }
            
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_beatmap_timing():
    """分析谱面的时间分布与音频特征的关系"""
    mcz_path = "trainData/_song_4833.mcz"
    
    try:
        with zipfile.ZipFile(mcz_path, 'r') as mcz:
            # 读取谱面数据
            target_mc = "0/1511697495.mc"
            with mcz.open(target_mc, 'r') as f:
                mc_data = json.loads(f.read().decode('utf-8'))
            
            notes = mc_data.get('note', [])
            game_notes = [note for note in notes if 'column' in note]
            time_info = mc_data.get('time', [])
            
            if time_info:
                bpm = time_info[0].get('bpm', 156)
                
                # 计算每个音符的时间
                note_times = []
                for note in game_notes:
                    beat = note.get('beat', [])
                    if len(beat) >= 3:
                        x, y, z = beat[0], beat[1], beat[2]
                        beat_value = x + y / z
                        time_seconds = beat_value * 60 / bpm
                        note_times.append(time_seconds)
                
                note_times = np.array(sorted(note_times))
                
                print(f"\n🎼 谱面时间分析:")
                print(f"   谱面音符数: {len(note_times)}")
                print(f"   时间范围: {note_times[0]:.2f} - {note_times[-1]:.2f} 秒")
                print(f"   平均音符间隔: {np.mean(np.diff(note_times)):.3f} 秒")
                print(f"   音符密度: {len(note_times) / note_times[-1]:.2f} 个/秒")
                
                return note_times
                
    except Exception as e:
        print(f"❌ 谱面分析失败: {e}")
        return None

def compare_audio_and_beatmap():
    """对比音频特征和谱面时间分布"""
    print(f"🔍 音频特征与谱面生成的关系分析")
    print(f"=" * 60)
    
    # 分析音频特征
    audio_features = analyze_audio_features_for_beatmap()
    
    # 分析谱面时间分布
    note_times = analyze_beatmap_timing()
    
    if audio_features and note_times is not None:
        print(f"\n🎯 特征与谱面关联分析:")
        
        # 1. 音符起始与谱面音符的对应
        onset_times = audio_features['onset_times']
        beat_times = audio_features['beat_times']
        
        print(f"   音频起始事件数: {len(onset_times)}")
        print(f"   检测到的节拍数: {len(beat_times)}")
        print(f"   谱面音符数: {len(note_times)}")
        
        # 计算匹配度
        onset_vs_notes = len(onset_times) / len(note_times) if len(note_times) > 0 else 0
        beats_vs_notes = len(beat_times) / len(note_times) if len(note_times) > 0 else 0
        
        print(f"   起始事件/谱面音符比: {onset_vs_notes:.2f}")
        print(f"   节拍/谱面音符比: {beats_vs_notes:.2f}")
        
        # 2. 能量与音符密度的关系
        rms = audio_features['rms']
        times = audio_features['times']
        
        # 计算高能量区域
        rms_threshold = np.percentile(rms, 75)  # 75%分位数作为阈值
        high_energy_ratio = np.sum(rms > rms_threshold) / len(rms)
        
        print(f"   高能量区域占比: {high_energy_ratio:.2f}")
        print(f"   RMS能量阈值: {rms_threshold:.4f}")

def explain_feature_extraction_methods():
    """解释不同的特征提取方法"""
    print(f"\n" + "=" * 60)
    print(f"🧠 音游谱面生成中的特征提取方法详解")
    print(f"=" * 60)
    
    print(f"\n1. 📊 基于能量/音量的方法:")
    print(f"   • RMS (Root Mean Square): 计算音频片段的平均能量")
    print(f"   • 优点: 简单直观，高能量通常对应强拍")
    print(f"   • 缺点: 无法区分不同音色，可能错过轻柔但重要的音符")
    print(f"   • 应用: 初级谱面生成，强拍检测")
    
    print(f"\n2. 🎵 基于音符起始检测 (Onset Detection):")
    print(f"   • 方法: 检测音频中音符开始的时刻")
    print(f"   • 特征: 频谱变化、相位变化、复杂度变化")
    print(f"   • 优点: 直接对应'什么时候有新声音'")
    print(f"   • 缺点: 可能过于密集，需要筛选")
    print(f"   • 应用: 现代音游谱面生成的核心方法")
    
    print(f"\n3. 🥁 基于节拍跟踪 (Beat Tracking):")
    print(f"   • 方法: 分析音乐的节拍模式")
    print(f"   • 特征: 周期性能量变化、节拍强度")
    print(f"   • 优点: 符合音乐理论，节奏感强")
    print(f"   • 缺点: 可能错过细节，适合基础节拍")
    print(f"   • 应用: 节拍框架建立，难度分级")
    
    print(f"\n4. 🎨 基于频谱特征:")
    print(f"   • MFCC: 音色特征，区分不同乐器")
    print(f"   • Chroma: 音调特征，检测和声变化")
    print(f"   • Spectral Centroid: 音色亮度")
    print(f"   • 优点: 能区分音乐元素，生成更丰富的谱面")
    print(f"   • 缺点: 计算复杂，需要音乐理论知识")
    print(f"   • 应用: 高级谱面生成，按音色分配键位")
    
    print(f"\n5. 🤖 基于机器学习的方法:")
    print(f"   • 监督学习: 从现有谱面学习'什么样的音频对应什么样的谱面'")
    print(f"   • 特征组合: 结合多种音频特征")
    print(f"   • 时序模型: LSTM/Transformer捕捉时间依赖")
    print(f"   • 优点: 能学习复杂模式，接近人类谱师水平")
    print(f"   • 缺点: 需要大量训练数据")
    print(f"   • 应用: 我们当前的训练模型")

def analyze_our_training_approach():
    """分析我们当前训练方法中的特征提取"""
    print(f"\n" + "=" * 60)
    print(f"🔬 我们当前训练方法的特征提取分析")
    print(f"=" * 60)
    
    print(f"\n📋 回顾我们的训练过程:")
    print(f"   1. 预处理阶段:")
    print(f"      • 从354个MCZ文件中提取音频特征")
    print(f"      • 提取的特征包括: tempo, spectral_centroid, zcr, mfcc等")
    print(f"      • 同时提取谱面的beat位置和键位信息")
    
    print(f"\n   2. 特征工程:")
    print(f"      • 将音频片段与对应时间的谱面音符关联")
    print(f"      • 创建'音频特征 -> 是否应该有音符'的映射")
    print(f"      • 包含时间窗口内的多维特征向量")
    
    print(f"\n   3. 训练目标:")
    print(f"      • 输入: 某个时间点的音频特征")
    print(f"      • 输出: 该时间点是否应该放置音符，以及应该在哪个键位")
    print(f"      • 损失函数: 预测准确率 (最终达到74.6%)")
    
    print(f"\n🎯 当前方法的特点:")
    print(f"   ✅ 优点:")
    print(f"      • 学习了大量真实谱面的模式")
    print(f"      • 考虑了多维音频特征的组合")
    print(f"      • 能够预测键位分配")
    
    print(f"   ⚠️  局限:")
    print(f"      • 特征提取相对简单，主要是统计特征")
    print(f"      • 缺少音乐理论指导")
    print(f"      • 时间分辨率可能不够精细")
    
    print(f"\n🚀 改进建议:")
    print(f"   • 添加onset detection作为核心特征")
    print(f"   • 引入节拍跟踪建立时间框架")
    print(f"   • 使用更细粒度的时间窗口")
    print(f"   • 考虑音乐结构 (verse, chorus等)")
    print(f"   • 引入难度自适应生成")

if __name__ == "__main__":
    # 分析音频特征与谱面的关系
    compare_audio_and_beatmap()
    
    # 解释各种特征提取方法
    explain_feature_extraction_methods()
    
    # 分析我们当前的训练方法
    analyze_our_training_approach()
