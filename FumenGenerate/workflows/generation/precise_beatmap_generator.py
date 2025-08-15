#!/usr/bin/env python3
"""
精确时机校准的谱面生成器
在音量峰值处精确放置音符，确保最佳游戏体验
"""

import numpy as np
import librosa
import json
import zipfile
import os
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt

def generate_precise_beatmap(audio_path, output_path, target_note_count=1500, plot_analysis=True):
    """
    生成时机精确校准的谱面
    
    Args:
        audio_path: 输入音频文件路径
        output_path: 输出MCZ文件路径
        target_note_count: 目标音符数量
        plot_analysis: 是否绘制分析图表
    """
    print(f"🎯 开始生成精确时机校准的谱面...")
    print(f"📁 音频文件: {os.path.basename(audio_path)}")
    
    # 1. 加载音频
    y, sr = librosa.load(audio_path, sr=22050)
    duration = len(y) / sr
    print(f"⏱️ 音频时长: {duration:.2f} 秒")
    
    # 2. 精确的峰值检测
    peak_times = detect_precise_peaks(y, sr, target_note_count)
    print(f"🎵 检测到 {len(peak_times)} 个精确峰值")
    
    # 3. 智能键位分配
    note_data = assign_intelligent_columns(y, sr, peak_times)
    print(f"🎹 完成键位分配")
    
    # 4. 生成MCZ文件
    generate_mcz_file(note_data, duration, output_path)
    print(f"✅ 谱面生成完成: {output_path}")
    
    # 5. 绘制分析图表
    if plot_analysis:
        plot_timing_analysis(y, sr, peak_times, note_data)
    
    return note_data

def detect_precise_peaks(y, sr, target_count):
    """
    检测音频中的精确峰值位置
    
    Args:
        y: 音频信号
        sr: 采样率
        target_count: 目标峰值数量
    
    Returns:
        peak_times: 峰值时间数组
    """
    print(f"🔍 正在进行精确峰值检测...")
    
    # 1. 计算短时能量 (10ms窗口，高时间分辨率)
    frame_length = int(0.01 * sr)  # 10ms窗口
    hop_length = int(0.005 * sr)   # 5ms步长
    
    # 使用RMS能量
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # 2. 平滑处理减少噪声
    if len(rms) > 10:
        window_length = min(11, len(rms) if len(rms) % 2 == 1 else len(rms) - 1)
        rms_smooth = savgol_filter(rms, window_length, 3)
    else:
        rms_smooth = rms
    
    # 3. 动态阈值峰值检测
    rms_mean = np.mean(rms_smooth)
    rms_std = np.std(rms_smooth)
    
    # 从高阈值开始，逐步降低直到找到足够的峰值
    thresholds = [
        rms_mean + 2 * rms_std,
        rms_mean + 1.5 * rms_std,
        rms_mean + rms_std,
        rms_mean + 0.5 * rms_std,
        rms_mean
    ]
    
    best_peaks = None
    for threshold in thresholds:
        # 检测峰值，设置最小间距避免过密
        min_distance = int(0.05 * len(rms_smooth))  # 最小50ms间距
        
        peaks, properties = find_peaks(
            rms_smooth, 
            height=threshold,
            distance=min_distance,
            prominence=rms_std * 0.1
        )
        
        print(f"   阈值 {threshold:.4f}: 找到 {len(peaks)} 个峰值")
        
        # 如果峰值数量合适，使用这个阈值
        if target_count * 0.8 <= len(peaks) <= target_count * 1.2:
            best_peaks = peaks
            break
        elif len(peaks) > target_count * 0.5:
            best_peaks = peaks
    
    if best_peaks is None:
        # 如果都不合适，使用最宽松的阈值并截取
        threshold = rms_mean * 0.5
        peaks, _ = find_peaks(rms_smooth, height=threshold, distance=min_distance)
        best_peaks = peaks
    
    # 4. 精确定位峰值 (在原始信号中找到精确位置)
    precise_peaks = []
    
    for peak_idx in best_peaks:
        # 转换到原始信号的时间
        peak_time = peak_idx * hop_length / sr
        
        # 在峰值附近寻找真正的最大值 (±10ms范围)
        search_range = int(0.01 * sr)  # 10ms搜索范围
        start_sample = max(0, int(peak_time * sr) - search_range)
        end_sample = min(len(y), int(peak_time * sr) + search_range)
        
        if start_sample < end_sample:
            local_segment = np.abs(y[start_sample:end_sample])
            local_max_idx = np.argmax(local_segment)
            precise_time = (start_sample + local_max_idx) / sr
            precise_peaks.append(precise_time)
    
    # 5. 如果数量不够，补充一些次要峰值
    if len(precise_peaks) < target_count * 0.8:
        print(f"   峰值不够，补充次要峰值...")
        # 降低阈值，找更多峰值
        lower_threshold = rms_mean * 0.3
        additional_peaks, _ = find_peaks(rms_smooth, height=lower_threshold, distance=min_distance//2)
        
        for peak_idx in additional_peaks:
            peak_time = peak_idx * hop_length / sr
            if not any(abs(peak_time - existing) < 0.1 for existing in precise_peaks):
                precise_peaks.append(peak_time)
                if len(precise_peaks) >= target_count:
                    break
    
    # 6. 如果数量过多，选择最强的峰值
    if len(precise_peaks) > target_count:
        print(f"   峰值过多，选择最强的 {target_count} 个...")
        # 计算每个峰值的强度
        peak_strengths = []
        for peak_time in precise_peaks:
            sample_idx = int(peak_time * sr)
            if 0 <= sample_idx < len(y):
                strength = abs(y[sample_idx])
                peak_strengths.append((peak_time, strength))
        
        # 按强度排序，选择最强的
        peak_strengths.sort(key=lambda x: x[1], reverse=True)
        precise_peaks = [pt[0] for pt in peak_strengths[:target_count]]
    
    # 7. 按时间排序
    precise_peaks = sorted(precise_peaks)
    
    print(f"✅ 最终检测到 {len(precise_peaks)} 个精确峰值")
    print(f"   时间范围: {precise_peaks[0]:.3f} - {precise_peaks[-1]:.3f} 秒")
    
    return np.array(precise_peaks)

def assign_intelligent_columns(y, sr, peak_times):
    """
    基于音频特征智能分配键位
    
    Args:
        y: 音频信号
        sr: 采样率
        peak_times: 峰值时间数组
    
    Returns:
        note_data: 包含时间和键位的音符数据
    """
    print(f"🎹 开始智能键位分配...")
    
    note_data = []
    
    for i, peak_time in enumerate(peak_times):
        # 1. 提取峰值附近的音频特征
        sample_idx = int(peak_time * sr)
        
        # 提取±25ms的音频段用于特征分析
        window_size = int(0.025 * sr)
        start_idx = max(0, sample_idx - window_size)
        end_idx = min(len(y), sample_idx + window_size)
        audio_window = y[start_idx:end_idx]
        
        if len(audio_window) == 0:
            column = 0
        else:
            # 2. 计算音频特征
            features = calculate_audio_features(audio_window, sr)
            
            # 3. 基于特征选择键位
            column = select_column_by_features(features, i, len(peak_times))
        
        note_data.append({
            'time': peak_time,
            'column': column,
            'features': features if len(audio_window) > 0 else {}
        })
    
    # 4. 后处理：避免连续相同键位
    note_data = post_process_columns(note_data)
    
    print(f"✅ 键位分配完成")
    return note_data

def calculate_audio_features(audio_window, sr):
    """计算音频窗口的特征"""
    if len(audio_window) == 0:
        return {}
    
    features = {}
    
    # 1. 时域特征
    features['rms'] = np.sqrt(np.mean(audio_window ** 2))
    features['peak_amplitude'] = np.max(np.abs(audio_window))
    features['zcr'] = np.sum(np.diff(np.sign(audio_window)) != 0) / len(audio_window)
    
    # 2. 频域特征
    fft = np.fft.fft(audio_window)
    magnitude = np.abs(fft)
    freqs = np.fft.fftfreq(len(fft), 1/sr)
    
    # 只考虑正频率
    positive_freqs = freqs[:len(freqs)//2]
    positive_magnitude = magnitude[:len(magnitude)//2]
    
    if np.sum(positive_magnitude) > 0:
        # 频谱质心
        features['spectral_centroid'] = np.sum(positive_freqs * positive_magnitude) / np.sum(positive_magnitude)
        
        # 频谱能量分布
        total_energy = np.sum(positive_magnitude)
        features['low_freq_energy'] = np.sum(positive_magnitude[positive_freqs < 200]) / total_energy
        features['mid_freq_energy'] = np.sum(positive_magnitude[(positive_freqs >= 200) & (positive_freqs < 2000)]) / total_energy
        features['high_freq_energy'] = np.sum(positive_magnitude[positive_freqs >= 2000]) / total_energy
    else:
        features['spectral_centroid'] = 0
        features['low_freq_energy'] = 0
        features['mid_freq_energy'] = 0
        features['high_freq_energy'] = 0
    
    return features

def select_column_by_features(features, note_index, total_notes):
    """基于特征选择键位"""
    if not features:
        return note_index % 4
    
    # 1. 基于频率特征的基础键位选择
    spectral_centroid = features.get('spectral_centroid', 1000)
    low_energy = features.get('low_freq_energy', 0)
    high_energy = features.get('high_freq_energy', 0)
    
    # 频率 -> 键位映射
    if spectral_centroid < 300 or low_energy > 0.6:
        base_column = 0  # 低频 -> 左边
    elif spectral_centroid < 800:
        base_column = 1
    elif spectral_centroid < 1500:
        base_column = 2
    else:
        base_column = 3  # 高频 -> 右边
    
    # 2. 考虑音量强度的调整
    rms = features.get('rms', 0)
    peak_amp = features.get('peak_amplitude', 0)
    
    # 强音符倾向于外侧键位 (0或3)
    if rms > 0.3 or peak_amp > 0.5:
        if base_column == 1:
            base_column = 0
        elif base_column == 2:
            base_column = 3
    
    # 3. 时间位置的微调 (避免单调)
    position_factor = note_index / total_notes
    if position_factor < 0.25:  # 开头部分
        bias = 0
    elif position_factor < 0.5:  # 前半部分
        bias = 1
    elif position_factor < 0.75:  # 后半部分
        bias = 2
    else:  # 结尾部分
        bias = 3
    
    # 轻微偏向时间位置对应的键位
    if np.random.random() < 0.3:
        base_column = bias
    
    return base_column

def post_process_columns(note_data):
    """后处理键位分配，避免不良模式"""
    if len(note_data) < 2:
        return note_data
    
    # 1. 避免连续超过3个相同键位
    for i in range(len(note_data) - 2):
        if (note_data[i]['column'] == note_data[i+1]['column'] == note_data[i+2]['column']):
            # 如果有3个连续相同，改变第3个
            available_columns = [c for c in range(4) if c != note_data[i]['column']]
            note_data[i+2]['column'] = np.random.choice(available_columns)
    
    # 2. 确保每个键位都有合理的使用
    columns_count = [0, 0, 0, 0]
    for note in note_data:
        columns_count[note['column']] += 1
    
    # 如果某个键位使用太少（<10%），增加一些
    total_notes = len(note_data)
    min_usage = total_notes * 0.1
    
    for col in range(4):
        if columns_count[col] < min_usage:
            # 随机选择一些音符改为这个键位
            needed = int(min_usage - columns_count[col])
            indices = np.random.choice(total_notes, min(needed, total_notes//4), replace=False)
            for idx in indices:
                note_data[idx]['column'] = col
    
    return note_data

def generate_mcz_file(note_data, duration, output_path):
    """生成MCZ文件"""
    print(f"📦 生成MCZ文件...")
    
    # 设置基本参数
    bpm = 156
    beats_per_measure = 4
    subdivision = 24  # 每拍24细分
    
    # 1. 转换时间到beat
    mc_notes = []
    
    # 添加音频控制音符
    start_beat = [0, 0, subdivision]
    end_beat_value = duration * bpm / 60
    end_beat = [int(end_beat_value), int((end_beat_value % 1) * subdivision), subdivision]
    
    mc_notes.extend([
        {"beat": [0, 0, int(subdivision)], "endbeat": [0, 0, int(subdivision)], "sound": "4833.ogg"},
        {"beat": [int(end_beat[0]), int(end_beat[1]), int(end_beat[2])], "endbeat": [int(end_beat[0]), int(end_beat[1]), int(end_beat[2])], "sound": ""}
    ])
    
    # 2. 添加游戏音符
    for note in note_data:
        beat_value = note['time'] * bpm / 60
        x = int(beat_value)
        y = int((beat_value % 1) * subdivision)
        z = subdivision
        
        mc_notes.append({
            "beat": [int(x), int(y), int(z)],
            "endbeat": [int(x), int(y), int(z)],
            "column": int(note['column'])
        })
    
    # 3. 创建MC数据结构
    mc_data = {
        "meta": {
            "version": "4.0.0",
            "mode": "0",
            "time": int(duration * 1000),
            "song": {
                "title": "Generated Beatmap (Precise)",
                "artist": "AI Generator",
                "id": 4833
            },
            "mode_ext": {
                "column": 4
            }
        },
        "time": [
            {
                "beat": [0, 0, int(subdivision)],
                "bpm": float(bpm)
            }
        ],
        "note": mc_notes
    }
    
    # 4. 创建MCZ文件
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建目录结构
        beat_dir = os.path.join(temp_dir, "0")
        os.makedirs(beat_dir)
        
        # 写入MC文件
        mc_file_path = os.path.join(beat_dir, "1511697495.mc")
        with open(mc_file_path, 'w', encoding='utf-8') as f:
            json.dump(mc_data, f, separators=(',', ':'))
        
        # 复制音频文件 (如果存在)
        audio_src = "generated_audio.ogg"
        if os.path.exists(audio_src):
            shutil.copy2(audio_src, os.path.join(beat_dir, "4833.ogg"))
        
        # 创建ZIP文件
        shutil.make_archive(output_path.replace('.mcz', ''), 'zip', temp_dir)
        if os.path.exists(output_path.replace('.mcz', '') + '.zip'):
            os.rename(output_path.replace('.mcz', '') + '.zip', output_path)
    
    print(f"✅ MCZ文件已生成: {output_path}")

def plot_timing_analysis(y, sr, peak_times, note_data, show_plot=True):
    """绘制时机分析图表"""
    print(f"📊 绘制时机分析图表...")
    
    plt.figure(figsize=(15, 10))
    
    # 1. 音频波形和峰值
    plt.subplot(3, 1, 1)
    time_axis = np.linspace(0, len(y)/sr, len(y))
    plt.plot(time_axis, y, alpha=0.6, color='lightblue', label='音频波形')
    
    # 标记检测到的峰值
    for peak_time in peak_times:
        plt.axvline(x=peak_time, color='red', alpha=0.7, linestyle='--', linewidth=1)
    
    plt.title('音频波形与检测到的峰值')
    plt.xlabel('时间 (秒)')
    plt.ylabel('振幅')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. RMS能量曲线
    plt.subplot(3, 1, 2)
    frame_length = int(0.01 * sr)
    hop_length = int(0.005 * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_times = np.linspace(0, len(y)/sr, len(rms))
    
    plt.plot(rms_times, rms, color='green', label='RMS能量')
    
    # 标记峰值对应的RMS
    for peak_time in peak_times:
        plt.axvline(x=peak_time, color='red', alpha=0.7, linestyle='--', linewidth=1)
    
    plt.title('RMS能量曲线')
    plt.xlabel('时间 (秒)')
    plt.ylabel('RMS')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 键位分布
    plt.subplot(3, 1, 3)
    columns = [note['column'] for note in note_data]
    times = [note['time'] for note in note_data]
    
    colors = ['red', 'blue', 'green', 'orange']
    for col in range(4):
        col_times = [t for t, c in zip(times, columns) if c == col]
        col_y = [col] * len(col_times)
        plt.scatter(col_times, col_y, c=colors[col], alpha=0.7, s=20, label=f'键位 {col}')
    
    plt.title('键位分布')
    plt.xlabel('时间 (秒)')
    plt.ylabel('键位')
    plt.yticks(range(4))
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        plt.savefig('timing_analysis.png', dpi=150, bbox_inches='tight')
        print(f"📊 图表已保存为 timing_analysis.png")
    
    plt.close()

def analyze_timing_precision(audio_path, note_data):
    """分析时机精确度"""
    print(f"\n🎯 时机精确度分析:")
    
    y, sr = librosa.load(audio_path, sr=22050)
    
    precisions = []
    for note in note_data:
        peak_time = note['time']
        sample_idx = int(peak_time * sr)
        
        # 检查±5ms范围内的最大值位置
        search_range = int(0.005 * sr)  # 5ms
        start_idx = max(0, sample_idx - search_range)
        end_idx = min(len(y), sample_idx + search_range)
        
        if start_idx < end_idx:
            local_segment = np.abs(y[start_idx:end_idx])
            true_peak_idx = np.argmax(local_segment)
            true_peak_sample = start_idx + true_peak_idx
            
            # 计算误差
            error_samples = abs(sample_idx - true_peak_sample)
            error_ms = error_samples / sr * 1000
            precisions.append(error_ms)
    
    if precisions:
        avg_error = np.mean(precisions)
        max_error = np.max(precisions)
        std_error = np.std(precisions)
        
        print(f"   平均误差: {avg_error:.2f} ms")
        print(f"   最大误差: {max_error:.2f} ms")
        print(f"   误差标准差: {std_error:.2f} ms")
        print(f"   95%音符误差小于: {np.percentile(precisions, 95):.2f} ms")
        
        # 评价
        if avg_error < 10:
            print(f"   🟢 精确度评价: 优秀 (平均误差 < 10ms)")
        elif avg_error < 20:
            print(f"   🟡 精确度评价: 良好 (平均误差 < 20ms)")
        else:
            print(f"   🔴 精确度评价: 需改进 (平均误差 > 20ms)")

if __name__ == "__main__":
    # 测试精确谱面生成 - 使用现有的音频文件
    audio_file = "extracted_audio\_song_10088_Kawaki wo Ameku.ogg"
    output_file = "precise_beatmap.mcz"
    
    if os.path.exists(audio_file):
        print(f"🎵 使用音频文件: {audio_file}")
        note_data = generate_precise_beatmap(
            audio_file, 
            output_file, 
            target_note_count=1500,
            plot_analysis=False
        )
        
        # 分析精确度
        analyze_timing_precision(audio_file, note_data)
        
        print(f"\n🎮 精确谱面生成完成！")
        print(f"📂 输出文件: {output_file}")
        print(f"🎵 音符数量: {len(note_data)}")
        print(f"🎯 特点: 所有音符都精确对准音量峰值，误差最小化")
    else:
        # 如果第一个文件不存在，尝试其他文件
        alternative_files = [
            "temp_mcz_analysis/0/Kawaki wo Ameku.ogg",
            "extracted_audio/_song_1011_audio.ogg",
            "preprocessed_data/audio/_song_1314_ON FIRE.ogg"
        ]
        
        for alt_file in alternative_files:
            if os.path.exists(alt_file):
                print(f"🎵 使用备选音频文件: {alt_file}")
                note_data = generate_precise_beatmap(
                    alt_file, 
                    output_file, 
                    target_note_count=1500,
                    plot_analysis=False
                )
                
                # 分析精确度
                analyze_timing_precision(alt_file, note_data)
                
                print(f"\n🎮 精确谱面生成完成！")
                print(f"📂 输出文件: {output_file}")
                print(f"🎵 音符数量: {len(note_data)}")
                print(f"🎯 特点: 所有音符都精确对准音量峰值，误差最小化")
                break
        else:
            print(f"❌ 未找到可用的音频文件")
            print(f"   请提供音频文件或检查文件路径")
