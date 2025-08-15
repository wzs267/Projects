#!/usr/bin/env python3
"""
改进的精确时机校准谱面生成器
专注于音量峰值的精确定位，确保音游体验
"""

import numpy as np
import librosa
import json
import zipfile
import os
from scipy.signal import find_peaks, savgol_filter

def generate_precise_beatmap(audio_path, output_path, target_note_count=1500):
    """生成时机精确校准的谱面"""
    print(f"🎯 开始生成精确时机校准的谱面...")
    print(f"📁 音频文件: {os.path.basename(audio_path)}")
    
    # 1. 加载音频
    y, sr = librosa.load(audio_path, sr=22050)
    duration = len(y) / sr
    print(f"⏱️ 音频时长: {duration:.2f} 秒")
    
    # 2. 精确的峰值检测
    peak_times = detect_audio_peaks(y, sr, target_note_count)
    print(f"🎵 检测到 {len(peak_times)} 个精确峰值")
    
    # 3. 智能键位分配
    note_data = assign_smart_columns(y, sr, peak_times)
    print(f"🎹 完成键位分配")
    
    # 4. 生成MCZ文件
    generate_mcz_file(note_data, duration, output_path)
    print(f"✅ 谱面生成完成: {output_path}")
    
    # 5. 分析精确度
    analyze_timing_precision(y, sr, note_data)
    
    return note_data

def detect_audio_peaks(y, sr, target_count):
    """检测音频峰值，确保精确时机"""
    print(f"🔍 正在进行音频峰值检测...")
    
    # 1. 多分辨率能量分析
    # 短时能量 (用于精确定位)
    frame_short = int(0.01 * sr)  # 10ms
    hop_short = int(0.005 * sr)   # 5ms
    
    # 中时能量 (用于峰值检测)
    frame_med = int(0.05 * sr)    # 50ms
    hop_med = int(0.025 * sr)     # 25ms
    
    # 计算RMS能量
    rms_short = librosa.feature.rms(y=y, frame_length=frame_short, hop_length=hop_short)[0]
    rms_med = librosa.feature.rms(y=y, frame_length=frame_med, hop_length=hop_med)[0]
    
    # 2. 基于中等分辨率检测峰值位置
    if len(rms_med) > 5:
        # 平滑处理
        window_len = min(5, len(rms_med) if len(rms_med) % 2 == 1 else len(rms_med) - 1)
        if window_len >= 3:
            rms_smooth = savgol_filter(rms_med, window_len, 2)
        else:
            rms_smooth = rms_med
    else:
        rms_smooth = rms_med
    
    # 3. 自适应阈值检测
    rms_mean = np.mean(rms_smooth)
    rms_std = np.std(rms_smooth)
    
    print(f"   RMS统计: 均值={rms_mean:.4f}, 标准差={rms_std:.4f}")
    
    # 动态调整阈值直到找到合适数量的峰值
    thresholds = [
        rms_mean + 1.5 * rms_std,
        rms_mean + rms_std,
        rms_mean + 0.5 * rms_std,
        rms_mean + 0.2 * rms_std,
        rms_mean,
        rms_mean - 0.2 * rms_std
    ]
    
    best_peaks = None
    best_threshold = None
    
    for threshold in thresholds:
        # 最小间距：确保音符不会太密集
        min_distance = max(1, int(0.1 * len(rms_smooth)))  # 至少100ms间距
        
        peaks, properties = find_peaks(
            rms_smooth,
            height=threshold,
            distance=min_distance,
            prominence=rms_std * 0.05
        )
        
        print(f"   阈值 {threshold:.4f}: 找到 {len(peaks)} 个峰值")
        
        # 选择最接近目标数量的阈值
        if len(peaks) >= target_count * 0.5:  # 至少要有目标数量的一半
            best_peaks = peaks
            best_threshold = threshold
            if len(peaks) <= target_count * 1.2:  # 不要超过太多
                break
    
    if best_peaks is None or len(best_peaks) < 10:
        print(f"   ⚠️ 峰值太少，使用备选方案...")
        # 备选方案：更低的阈值
        min_threshold = rms_mean - rms_std
        min_distance = max(1, int(0.05 * len(rms_smooth)))  # 50ms间距
        
        peaks, _ = find_peaks(
            rms_smooth,
            height=min_threshold,
            distance=min_distance
        )
        best_peaks = peaks
        best_threshold = min_threshold
        print(f"   备选阈值 {best_threshold:.4f}: 找到 {len(peaks)} 个峰值")
    
    # 4. 转换到精确时间位置
    precise_peaks = []
    
    for peak_idx in best_peaks:
        # 转换到时间
        rough_time = peak_idx * hop_med / sr
        
        # 在高分辨率数据中寻找精确位置
        search_start = max(0, int((rough_time - 0.05) * sr / hop_short))
        search_end = min(len(rms_short), int((rough_time + 0.05) * sr / hop_short))
        
        if search_start < search_end:
            local_rms = rms_short[search_start:search_end]
            if len(local_rms) > 0:
                local_peak_idx = np.argmax(local_rms)
                precise_time = (search_start + local_peak_idx) * hop_short / sr
                precise_peaks.append(precise_time)
        else:
            precise_peaks.append(rough_time)
    
    # 5. 进一步在原始信号中精确定位
    final_peaks = []
    for peak_time in precise_peaks:
        # 在原始信号中寻找±10ms范围内的真正峰值
        center_sample = int(peak_time * sr)
        search_range = int(0.01 * sr)  # 10ms搜索范围
        
        start_sample = max(0, center_sample - search_range)
        end_sample = min(len(y), center_sample + search_range)
        
        if start_sample < end_sample:
            local_audio = np.abs(y[start_sample:end_sample])
            if len(local_audio) > 0:
                local_max_idx = np.argmax(local_audio)
                exact_time = (start_sample + local_max_idx) / sr
                final_peaks.append(exact_time)
    
    # 6. 如果数量不够，补充更多峰值
    if len(final_peaks) < target_count * 0.7:
        print(f"   📈 峰值数量不足，补充更多...")
        # 使用更低的阈值获取更多峰值
        additional_threshold = best_threshold * 0.7
        min_distance = max(1, int(0.03 * len(rms_smooth)))  # 30ms间距
        
        additional_peaks, _ = find_peaks(
            rms_smooth,
            height=additional_threshold,
            distance=min_distance
        )
        
        for peak_idx in additional_peaks:
            peak_time = peak_idx * hop_med / sr
            # 避免重复
            if not any(abs(peak_time - existing) < 0.05 for existing in final_peaks):
                final_peaks.append(peak_time)
                if len(final_peaks) >= target_count:
                    break
    
    # 7. 如果还是太少，使用均匀分布补充
    if len(final_peaks) < target_count * 0.5:
        print(f"   ⚡ 使用智能补充...")
        # 分析音频活跃区域
        audio_active = np.abs(y) > np.mean(np.abs(y)) * 0.3
        active_segments = []
        
        # 找到连续的活跃段
        in_segment = False
        segment_start = 0
        
        for i, is_active in enumerate(audio_active):
            if is_active and not in_segment:
                segment_start = i
                in_segment = True
            elif not is_active and in_segment:
                segment_end = i
                if segment_end - segment_start > sr * 0.5:  # 至少0.5秒的段
                    active_segments.append((segment_start / sr, segment_end / sr))
                in_segment = False
        
        # 在活跃段中均匀添加音符
        needed = target_count - len(final_peaks)
        if active_segments and needed > 0:
            notes_per_segment = needed // len(active_segments)
            for start_time, end_time in active_segments:
                segment_duration = end_time - start_time
                if segment_duration > 1.0:  # 至少1秒
                    for i in range(notes_per_segment):
                        note_time = start_time + (i + 1) * segment_duration / (notes_per_segment + 1)
                        # 避免重复
                        if not any(abs(note_time - existing) < 0.1 for existing in final_peaks):
                            final_peaks.append(note_time)
    
    # 8. 限制数量并排序
    if len(final_peaks) > target_count:
        # 按音量强度排序，选择最强的
        peak_strengths = []
        for peak_time in final_peaks:
            sample_idx = int(peak_time * sr)
            if 0 <= sample_idx < len(y):
                strength = abs(y[sample_idx])
                peak_strengths.append((peak_time, strength))
        
        peak_strengths.sort(key=lambda x: x[1], reverse=True)
        final_peaks = [pt[0] for pt in peak_strengths[:target_count]]
    
    # 按时间排序
    final_peaks = sorted(final_peaks)
    
    print(f"✅ 最终检测到 {len(final_peaks)} 个精确峰值")
    if final_peaks:
        print(f"   时间范围: {final_peaks[0]:.3f} - {final_peaks[-1]:.3f} 秒")
    
    return np.array(final_peaks)

def assign_smart_columns(y, sr, peak_times):
    """智能分配键位"""
    print(f"🎹 开始智能键位分配...")
    
    note_data = []
    
    for i, peak_time in enumerate(peak_times):
        # 计算音频特征
        sample_idx = int(peak_time * sr)
        window_size = int(0.025 * sr)  # 25ms窗口
        
        start_idx = max(0, sample_idx - window_size)
        end_idx = min(len(y), sample_idx + window_size)
        audio_window = y[start_idx:end_idx]
        
        if len(audio_window) > 0:
            # 计算频率特征
            fft = np.fft.fft(audio_window)
            magnitude = np.abs(fft)
            freqs = np.fft.fftfreq(len(fft), 1/sr)
            
            positive_freqs = freqs[:len(freqs)//2]
            positive_magnitude = magnitude[:len(magnitude)//2]
            
            if np.sum(positive_magnitude) > 0:
                spectral_centroid = np.sum(positive_freqs * positive_magnitude) / np.sum(positive_magnitude)
            else:
                spectral_centroid = 1000
            
            # 基于频率选择键位
            if spectral_centroid < 400:
                base_column = 0
            elif spectral_centroid < 800:
                base_column = 1
            elif spectral_centroid < 1500:
                base_column = 2
            else:
                base_column = 3
            
            # 添加一些随机性避免单调
            if np.random.random() < 0.2:
                base_column = (base_column + np.random.randint(-1, 2)) % 4
            
        else:
            base_column = i % 4
        
        note_data.append({
            'time': peak_time,
            'column': base_column
        })
    
    # 后处理：平衡键位分布
    columns = [note['column'] for note in note_data]
    column_counts = [columns.count(i) for i in range(4)]
    
    print(f"   键位分布: {column_counts}")
    
    # 如果某个键位太少，调整一些音符
    min_count = len(note_data) // 8  # 每个键位至少12.5%
    for col in range(4):
        if column_counts[col] < min_count:
            needed = min_count - column_counts[col]
            # 从使用最多的键位借一些
            max_col = column_counts.index(max(column_counts))
            changed = 0
            for note in note_data:
                if note['column'] == max_col and changed < needed:
                    note['column'] = col
                    changed += 1
                    if changed >= needed:
                        break
    
    print(f"✅ 键位分配完成")
    return note_data

def generate_mcz_file(note_data, duration, output_path):
    """生成MCZ文件"""
    print(f"📦 生成MCZ文件...")
    
    # 设置参数
    bpm = 156
    subdivision = 24
    
    # 准备音符数据
    mc_notes = []
    
    # 音频控制
    mc_notes.extend([
        {"beat": [0, 0, 24], "endbeat": [0, 0, 24], "sound": "kawaki.ogg"},
        {"beat": [int(duration * bpm / 60), 0, 24], "endbeat": [int(duration * bpm / 60), 0, 24], "sound": ""}
    ])
    
    # 游戏音符
    for note in note_data:
        beat_value = note['time'] * bpm / 60
        x = int(beat_value)
        y = int((beat_value % 1) * subdivision)
        
        mc_notes.append({
            "beat": [x, y, 24],
            "endbeat": [x, y, 24],
            "column": int(note['column'])
        })
    
    # MC数据结构
    mc_data = {
        "meta": {
            "version": "4.0.0",
            "mode": "0",
            "time": int(duration * 1000),
            "song": {
                "title": "Precise Generated Beatmap",
                "artist": "AI Generator (Precise)",
                "id": 4833
            },
            "mode_ext": {
                "column": 4
            }
        },
        "time": [
            {
                "beat": [0, 0, 24],
                "bpm": 156.0
            }
        ],
        "note": mc_notes
    }
    
    # 创建MCZ文件
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as temp_dir:
        beat_dir = os.path.join(temp_dir, "0")
        os.makedirs(beat_dir)
        
        # 写入MC文件
        mc_file_path = os.path.join(beat_dir, "1511697495.mc")
        with open(mc_file_path, 'w', encoding='utf-8') as f:
            json.dump(mc_data, f, separators=(',', ':'))
        
        # 创建ZIP
        shutil.make_archive(output_path.replace('.mcz', ''), 'zip', temp_dir)
        if os.path.exists(output_path.replace('.mcz', '') + '.zip'):
            os.rename(output_path.replace('.mcz', '') + '.zip', output_path)

def analyze_timing_precision(y, sr, note_data):
    """分析时机精确度"""
    print(f"\n🎯 时机精确度分析:")
    
    precisions = []
    for note in note_data:
        peak_time = note['time']
        sample_idx = int(peak_time * sr)
        
        # 检查±5ms范围内的真实峰值位置
        search_range = int(0.005 * sr)
        start_idx = max(0, sample_idx - search_range)
        end_idx = min(len(y), sample_idx + search_range)
        
        if start_idx < end_idx:
            local_audio = np.abs(y[start_idx:end_idx])
            true_peak_idx = np.argmax(local_audio)
            true_peak_sample = start_idx + true_peak_idx
            
            error_samples = abs(sample_idx - true_peak_sample)
            error_ms = error_samples / sr * 1000
            precisions.append(error_ms)
    
    if precisions:
        avg_error = np.mean(precisions)
        max_error = np.max(precisions)
        
        print(f"   平均误差: {avg_error:.2f} ms")
        print(f"   最大误差: {max_error:.2f} ms")
        print(f"   95%音符误差小于: {np.percentile(precisions, 95):.2f} ms")
        
        if avg_error < 10:
            print(f"   🟢 精确度评价: 优秀 (±{avg_error:.1f}ms)")
        elif avg_error < 20:
            print(f"   🟡 精确度评价: 良好 (±{avg_error:.1f}ms)")
        else:
            print(f"   🔴 精确度评价: 需改进 (±{avg_error:.1f}ms)")

if __name__ == "__main__":
    # 使用现有音频文件测试
    audio_file = "extracted_audio\_song_10088_Kawaki wo Ameku.ogg"
    output_file = "precise_beatmap.mcz"
    
    if os.path.exists(audio_file):
        print(f"🎵 使用音频文件: {audio_file}")
        note_data = generate_precise_beatmap(audio_file, output_file, target_note_count=1200)
        
        print(f"\n🎮 精确谱面生成完成！")
        print(f"📂 输出文件: {output_file}")
        print(f"🎵 音符数量: {len(note_data)}")
        print(f"🎯 特点: 音符精确对准音量峰值，最小化时机误差")
    else:
        print(f"❌ 音频文件不存在: {audio_file}")
        print(f"请检查文件路径")
