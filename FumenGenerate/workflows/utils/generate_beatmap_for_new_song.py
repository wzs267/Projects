#!/usr/bin/env python3
"""
为新歌曲生成谱面并打包成MCZ文件
"""

import os
import sys
import tempfile
import zipfile
import json
import shutil
import numpy as np
import librosa
from pathlib import Path
import torch
import pickle
import random

class BeatmapGenerator:
    def __init__(self, model_path='models/large_scale_hybrid_model.pth'):
        """初始化谱面生成器"""
        self.model_path = model_path
        print(f"🤖 初始化谱面生成器")
        
    def extract_mcz_info(self, mcz_path):
        """简化的MCZ文件信息提取"""
        print(f"📂 解析MCZ文件: {mcz_path}")
        
        try:
            with zipfile.ZipFile(mcz_path, 'r') as mcz:
                # 查找音频文件
                all_files = mcz.namelist()
                audio_files = [f for f in all_files if f.endswith(('.ogg', '.mp3', '.wav'))]
                
                print(f"📁 MCZ文件包含 {len(all_files)} 个文件")
                print(f"🎵 找到 {len(audio_files)} 个音频文件: {audio_files}")
                
                if not audio_files:
                    raise ValueError("MCZ文件中没有找到音频文件")
                
                # 提取到临时目录
                temp_dir = tempfile.mkdtemp()
                audio_file = audio_files[0]  # 使用第一个音频文件
                
                print(f"📤 提取音频文件: {audio_file}")
                mcz.extract(audio_file, temp_dir)
                audio_path = os.path.join(temp_dir, audio_file)
                
                # 尝试解析歌曲信息
                song_title = os.path.splitext(os.path.basename(mcz_path))[0]
                if song_title.startswith('_song_'):
                    song_id = song_title.split('_')[-1]
                    song_title = f"Song {song_id}"
                
                return {
                    'title': song_title,
                    'artist': 'Unknown Artist',
                    'audio_path': audio_path,
                    'temp_dir': temp_dir
                }
                
        except Exception as e:
            print(f"❌ MCZ解析失败: {e}")
            return None
        
    def generate_beatmap_for_song(self, mcz_path, target_difficulty=20, target_keys=4):
        """为指定歌曲生成谱面"""
        print(f"🎵 开始为歌曲生成谱面: {mcz_path}")
        print(f"🎯 目标难度: {target_difficulty}, 按键数: {target_keys}K")
        
        # 提取歌曲信息
        try:
            song_data = self.extract_mcz_info(mcz_path)
            if not song_data:
                raise ValueError("无法解析MCZ文件")
                
            print(f"📖 歌曲信息:")
            print(f"   标题: {song_data.get('title', 'Unknown')}")
            print(f"   艺术家: {song_data.get('artist', 'Unknown')}")
            
            # 获取音频文件路径
            audio_path = song_data.get('audio_path')
            if not audio_path or not os.path.exists(audio_path):
                raise ValueError("音频文件不存在")
                
            print(f"🎵 音频文件: {audio_path}")
            
            # 提取音频特征
            print("🔍 提取音频特征...")
            audio_features = self.extract_audio_features(audio_path)
            
            # 生成谱面
            print("🎼 生成谱面...")
            beatmap = self.generate_beatmap(audio_features, target_difficulty, target_keys)
            
            # 创建完整的歌曲数据
            generated_song = {
                'title': song_data.get('title', 'Generated Song'),
                'artist': song_data.get('artist', 'AI'),
                'audio_path': audio_path,
                'difficulty': target_difficulty,
                'keys': target_keys,
                'beatmap': beatmap,
                'original_mcz': mcz_path
            }
            
            return generated_song
            
        except Exception as e:
            print(f"❌ 生成谱面失败: {e}")
            return None
            
    def extract_audio_features(self, audio_path):
        """提取音频特征"""
        try:
            # 加载音频
            y, sr = librosa.load(audio_path, sr=None)
            
            # 提取基本特征
            features = {}
            
            # 节拍相关特征
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = tempo
            features['beat_count'] = len(beats)
            
            # 频谱特征
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            # 过零率
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # MFCC特征
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i}_std'] = np.std(mfccs[i])
                
            # 色度特征
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = np.mean(chroma)
            features['chroma_std'] = np.std(chroma)
            
            # 音频长度
            features['duration'] = len(y) / sr
            
            return features
            
        except Exception as e:
            print(f"❌ 音频特征提取失败: {e}")
            return {}
            
    def generate_beatmap(self, audio_features, target_difficulty, target_keys):
        """基于音频特征生成谱面 - 简化版本"""
        print("🎼 使用基于规则的算法生成谱面...")
        
        # 获取基本参数
        duration = audio_features.get('duration', 120)  # 默认2分钟
        tempo = audio_features.get('tempo', 120)  # 默认120 BPM
        
        # 计算基本参数
        beats_per_second = float(tempo) / 60
        total_beats = int(float(duration) * beats_per_second)
        
        # 根据难度调整note密度
        difficulty_multiplier = target_difficulty / 30.0  # 标准化
        base_note_density = 0.2  # 基础note密度
        note_density = base_note_density * (0.5 + difficulty_multiplier * 1.5)
        
        # 限制密度范围
        note_density = max(0.1, min(0.8, note_density))
        
        print(f"🎼 生成参数:")
        print(f"   时长: {float(duration):.1f}秒")
        print(f"   BPM: {float(tempo):.1f}")
        print(f"   总拍数: {total_beats}")
        print(f"   Note密度: {note_density:.2f}")
        
        # 生成notes
        beatmap = []
        current_time = 0
        time_step = 60000 / float(tempo)  # 毫秒为单位的每拍时间
        
        # 使用音频特征影响生成
        spectral_mean = audio_features.get('spectral_centroid_mean', 1000)
        high_freq_factor = min(spectral_mean / 2000, 2.0)  # 高频内容影响
        
        for beat in range(total_beats):
            # 基于拍子位置调整密度
            beat_in_measure = beat % 4
            measure_multiplier = 1.0
            if beat_in_measure == 0:  # 强拍
                measure_multiplier = 1.3
            elif beat_in_measure == 2:  # 次强拍
                measure_multiplier = 1.1
                
            # 动态调整密度
            current_density = note_density * measure_multiplier * high_freq_factor
            
            # 决定是否在这一拍放置note
            if random.random() < current_density:
                # 避免连续同列
                available_columns = list(range(1, target_keys + 1))
                if beatmap and current_time - beatmap[-1]['time'] < time_step * 0.3:
                    last_column = beatmap[-1]['column']
                    if last_column in available_columns:
                        available_columns.remove(last_column)
                
                # 选择按键位置
                column = random.choice(available_columns)
                
                # 创建note
                note = {
                    'time': int(current_time),
                    'column': column,
                    'type': 'normal'
                }
                
                beatmap.append(note)
                
            current_time += time_step
            
        # 按时间排序
        beatmap.sort(key=lambda x: x['time'])
        
        print(f"✅ 生成了 {len(beatmap)} 个notes")
        return beatmap
        
    def create_mcz_package(self, song_data, output_path):
        """创建MCZ包"""
        try:
            print(f"📦 创建MCZ包: {output_path}")
            
            # 创建临时目录
            with tempfile.TemporaryDirectory() as temp_dir:
                # 准备文件
                info_file = os.path.join(temp_dir, 'info.json')
                beatmap_file = os.path.join(temp_dir, f"{song_data['title']}_4K_AI_Lv{song_data['difficulty']}.json")
                
                # 复制音频文件
                audio_filename = f"audio_{os.path.basename(song_data['audio_path'])}"
                temp_audio_path = os.path.join(temp_dir, audio_filename)
                shutil.copy2(song_data['audio_path'], temp_audio_path)
                
                # 创建info.json
                info_data = {
                    'title': song_data['title'],
                    'artist': song_data['artist'],
                    'audio_file': audio_filename,
                    'beatmaps': [{
                        'name': f"4K AI Lv.{song_data['difficulty']}",
                        'file': os.path.basename(beatmap_file),
                        'keys': song_data['keys'],
                        'difficulty': song_data['difficulty'],
                        'level': song_data['difficulty']
                    }],
                    'generated_by': 'AI Beatmap Generator',
                    'generation_time': str(np.datetime64('now'))
                }
                
                with open(info_file, 'w', encoding='utf-8') as f:
                    json.dump(info_data, f, ensure_ascii=False, indent=2)
                    
                # 创建谱面文件
                beatmap_data = {
                    'song_title': song_data['title'],
                    'artist': song_data['artist'],
                    'difficulty': song_data['difficulty'],
                    'keys': song_data['keys'],
                    'audio_file': audio_filename,
                    'notes': song_data['beatmap']
                }
                
                with open(beatmap_file, 'w', encoding='utf-8') as f:
                    json.dump(beatmap_data, f, ensure_ascii=False, indent=2)
                    
                # 创建ZIP包
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arc_name = os.path.relpath(file_path, temp_dir)
                            zipf.write(file_path, arc_name)
                            
            print(f"✅ MCZ包创建完成: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ MCZ包创建失败: {e}")
            return False

def main():
    """主函数"""
    # 配置
    input_mcz = "trainData/_song_4833.mcz"  # 选择一个有音频文件的MCZ
    output_mcz = "generated_beatmaps/generated_song_4833.mcz"
    target_difficulty = 20
    target_keys = 4
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_mcz), exist_ok=True)
    
    try:
        # 创建生成器
        generator = BeatmapGenerator()
        
        # 生成谱面
        song_data = generator.generate_beatmap_for_song(
            input_mcz, target_difficulty, target_keys
        )
        
        if song_data:
            # 创建MCZ包
            success = generator.create_mcz_package(song_data, output_mcz)
            
            if success:
                print(f"\n🎉 谱面生成成功！")
                print(f"📁 输出文件: {output_mcz}")
                print(f"🎵 歌曲: {song_data['title']}")
                print(f"🎯 难度: Lv.{song_data['difficulty']}")
                print(f"🎼 Notes数量: {len(song_data['beatmap'])}")
            else:
                print("❌ MCZ包创建失败")
        else:
            print("❌ 谱面生成失败")
            
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
