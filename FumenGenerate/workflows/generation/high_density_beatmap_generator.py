#!/usr/bin/env python3
"""
修复的谱面生成器 - 生成标准MC格式，使用正确的BPM和beat格式
现在集成训练好的ImprovedWeightedFusionTransformer模型
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
import random
import torch
from typing import Tuple, List, Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.improved_sequence_transformer import ImprovedWeightedFusionTransformer

class FixedBeatmapGenerator:
    def __init__(self, model_path="improved_weighted_fusion_model_3_7.pth"):
        """初始化谱面生成器"""
        print(f"🤖 初始化AI谱面生成器 (集成训练模型)")
        
        # 设备选择
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型配置
        self.sequence_length = 64
        self.feature_dim = 12  # 修正为训练模型的12维输入
        self.time_resolution = 0.05  # 50ms 时间精度
        
        # 加载训练好的模型
        self.model = self._load_trained_model(model_path)
        
        print(f"   📱 设备: {self.device}")
        print(f"   🎯 模型: {model_path}")
        print(f"   ⏱️ 时间精度: {self.time_resolution*1000:.0f}ms")
    
    def _load_trained_model(self, model_path: str) -> torch.nn.Module:
        """加载训练好的模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 创建模型结构
        model = ImprovedWeightedFusionTransformer(
            input_dim=self.feature_dim,
            d_model=256,
            num_heads=8,
            num_layers=6,
            dropout=0.1,
            rf_weight=0.3,
            nn_weight=0.7,
            learnable_weights=True
        ).to(self.device)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ 模型加载成功: {total_params:,} 参数")
        
        return model
    
    def extract_audio_features_for_model(self, audio_path: str) -> np.ndarray:
        """提取用于模型推理的12维音频特征"""
        print(f"🎵 提取模型特征: {os.path.basename(audio_path)}")
        
        # 加载音频
        y, sr = librosa.load(audio_path, sr=22050)
        duration = len(y) / sr
        
        # 计算时间帧
        hop_length = int(sr * self.time_resolution)
        n_frames = int(duration / self.time_resolution)
        
        features_list = []
        
        for i in range(n_frames):
            start_sample = i * hop_length
            end_sample = min(start_sample + hop_length, len(y))
            frame = y[start_sample:end_sample]
            
            if len(frame) < hop_length // 2:  # 跳过太短的帧
                break
            
            # 提取12维特征
            feature_vector = self._extract_frame_features(frame, sr)
            features_list.append(feature_vector)
        
        features = np.array(features_list)
        print(f"✅ 特征提取完成: {features.shape} ({duration:.1f}秒)")
        return features
    
    def _extract_frame_features(self, frame: np.ndarray, sr: int) -> np.ndarray:
        """提取单帧的12维特征"""
        features = np.zeros(12)
        
        try:
            # 确保frame不为空且有足够长度
            if len(frame) < 512:
                frame = np.pad(frame, (0, 512 - len(frame)), 'constant')
            
            # 1. RMS能量
            rms = np.sqrt(np.mean(frame**2))
            features[0] = np.clip(rms, 0, 1)
            
            # 2. 过零率
            zcr = np.mean(np.abs(np.diff(np.sign(frame)))) / 2
            features[1] = np.clip(zcr, 0, 1)
            
            # 3-12. MFCC前10个系数
            if len(frame) > 0:
                mfccs = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=10, n_fft=512, hop_length=256)
                mfcc_means = np.mean(mfccs, axis=1)
                # 归一化MFCC到0-1范围
                mfcc_normalized = (mfcc_means + 20) / 40  # MFCC通常在-20到20范围
                features[2:12] = np.clip(mfcc_normalized, 0, 1)
            
        except Exception as e:
            print(f"   ⚠️ 特征提取警告: {e}")
            # 返回随机特征，与训练时一致
            features = np.random.randn(12) * 0.5 + 0.3
            features = np.clip(features, 0, 1)
        
        return features
    
    def create_sequences(self, features: np.ndarray) -> np.ndarray:
        """创建序列数据用于模型推理"""
        if len(features) < self.sequence_length:
            # 如果音频太短，重复特征
            repeat_times = (self.sequence_length // len(features)) + 1
            features = np.tile(features, (repeat_times, 1))
        
        sequences = []
        for i in range(len(features) - self.sequence_length + 1):
            sequence = features[i:i + self.sequence_length]
            sequences.append(sequence)
        
        return np.array(sequences)
    
    def predict_beatmap(self, sequences: np.ndarray) -> np.ndarray:
        """使用模型预测谱面"""
        print(f"🧠 模型推理: {sequences.shape[0]} 个序列")
        
        all_note_predictions = []
        
        with torch.no_grad():
            for i in range(0, len(sequences), 32):  # 批处理
                batch = sequences[i:i+32]
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                
                note_pred, event_pred = self.model(batch_tensor)
                all_note_predictions.append(note_pred.cpu().numpy())
        
        note_predictions = np.vstack(all_note_predictions)
        print(f"✅ 推理完成: {note_predictions.shape}")
        return note_predictions
    
    def ai_generate_notes(self, audio_path: str, duration: float, tempo: float, target_keys: int = 4) -> List[Dict[str, Any]]:
        """使用AI模型生成音符"""
        print("🤖 使用AI模型生成音符...")
        
        # 1. 提取音频特征
        features = self.extract_audio_features_for_model(audio_path)
        
        # 2. 创建序列
        sequences = self.create_sequences(features)
        print(f"✅ 序列创建完成: {sequences.shape}")
        
        # 3. 模型推理
        note_predictions = self.predict_beatmap(sequences)
        
        # 4. 移除放大处理，使用原始预测值
        adjusted_predictions = note_predictions  # 使用原始预测值，不放大
        
        print(f"📊 预测统计: min={note_predictions.min():.6f}, max={note_predictions.max():.6f}")
        print(f"📊 调整后统计: min={adjusted_predictions.min():.6f}, max={adjusted_predictions.max():.6f}")
        print(f"📊 大于0.05的预测: {(adjusted_predictions > 0.05).sum()} / {adjusted_predictions.size}")
        
        # 🔍 调试：检查每个轨道的预测分布
        print(f"🔍 轨道预测分析:")
        for track in range(4):
            track_predictions = adjusted_predictions[:, track]
            track_above_threshold = (track_predictions > 0.0001).sum()  # 调整阈值到合理范围
            print(f"   轨道{track}: min={track_predictions.min():.6f}, max={track_predictions.max():.6f}, "
                  f"均值={track_predictions.mean():.6f}, 大于阈值0.0001: {track_above_threshold}")
        
        # 🔍 调试：显示前10个预测样本
        print(f"🔍 前10个预测样本:")
        for i in range(min(10, len(adjusted_predictions))):
            pred = adjusted_predictions[i]
            print(f"   样本{i}: [{pred[0]:.6f}, {pred[1]:.6f}, {pred[2]:.6f}, {pred[3]:.6f}]")
        
        # 5. 计算正确的beat范围（关键修复！）
        target_beats = (duration * tempo) / 60  # 正确的beat数计算
        print(f"📊 目标beat范围: 0 - {target_beats:.1f}拍")
        
        # 6. 生成音符 - 使用完整的beat范围而不是仅限于AI序列
        notes = []
        beat_duration = 60.0 / tempo  # 一拍的时长（秒）
        
        # 使用beat值循环，确保覆盖完整歌曲长度
        subdivisions = 6  # 进一步降低精度：12 → 6分音符（只在强拍和次强拍放置音符）
        current_beat = 0.0
        beat_step = 1.0 / subdivisions  # 每次增加1/6拍（大幅降低密度）
        
        ai_prediction_index = 0
        generated_per_track = [0, 0, 0, 0]  # 统计每个轨道生成的音符数
        
        while current_beat < target_beats:
            # 计算当前时间位置（秒）
            current_time = current_beat * 60 / tempo
            
            # 如果超过歌曲长度，停止生成
            if current_time >= duration:
                break
            
            # 计算beat数组格式 [x, y, z] 其中 current_beat = x + y/z
            x = int(current_beat)  # 整数部分
            y_fraction = current_beat - x  # 小数部分
            y = int(y_fraction * subdivisions)  # 转换为分子
            beat_array = [x, y, subdivisions]
            
            # 获取AI预测（如果还有的话，否则使用较低的基准概率）
            if ai_prediction_index < len(adjusted_predictions):
                ai_prediction = adjusted_predictions[ai_prediction_index]
                # 每个AI预测对应多个beat位置
                if current_beat >= (ai_prediction_index + 1) * self.time_resolution / beat_duration:
                    ai_prediction_index += 1
            else:
                # 超出AI预测范围，使用更低基准值
                ai_prediction = np.array([0.01, 0.01, 0.01, 0.01])  # 降低基准：0.02 → 0.01
            
            # 对每个轨道判断是否放置音符 (轨道索引: 0到target_keys-1)
            for track in range(target_keys):  # 4K模式: track = 0,1,2,3
                probability = ai_prediction[track] if ai_prediction_index < len(adjusted_predictions) else 0.01
                
                # 基于位置调整密度（进一步降低倍数）
                density_multiplier = 1.0
                if y % 6 == 0:  # 强拍（每4分音符）
                    density_multiplier = 0.8  # 进一步降低：1.2 → 0.8
                elif y % 3 == 0:  # 次强拍（每8分音符）
                    density_multiplier = 0.6  # 进一步降低：1.2 → 0.6
                elif y % 2 == 0:  # 弱拍（每16分音符）
                    density_multiplier = 0.4  # 进一步降低：1.0 → 0.4
                else:  # 最弱拍
                    density_multiplier = 0.2  # 进一步降低：0.8 → 0.2
                
                adjusted_probability = probability * density_multiplier
                
                # 使用合理阈值减少音符数量（调整到原始预测值范围）
                if adjusted_probability > 0.0003:  # 调整阈值到合理范围：0.5 → 0.0003
                    # 降低随机概率，减少音符密度
                    if np.random.random() < min(adjusted_probability * 20, 0.4):  # 调整系数：0.4→20, 0.15→0.4
                        note = {
                            'beat': beat_array,
                            'column': track  # 确保使用正确的轨道索引 (0,1,2,3)
                        }
                        notes.append(note)
                        generated_per_track[track] += 1  # 统计每轨道音符数
            
            # 增加beat值
            current_beat += beat_step
        
        # 后处理：移除过于接近的音符
        notes = self._post_process_notes(notes, target_keys)

        print(f"✅ AI生成音符: {len(notes)} 个")
        print(f"🔍 每轨道音符分布: 轨道0:{generated_per_track[0]}, 轨道1:{generated_per_track[1]}, 轨道2:{generated_per_track[2]}, 轨道3:{generated_per_track[3]}")
        return notes
    
    def _convert_to_beat_array(self, beat_offset: float) -> List[int]:
        """将beat转换为24分音符数组格式"""
        # 计算小节数和拍数
        measure = int(beat_offset // 4)
        beat_in_measure = beat_offset % 4
        
        # 转换为24分音符（每拍24个细分）
        subdivision = int(beat_in_measure * 24)
        
        return [measure, subdivision, 24]
    
    def _post_process_notes(self, notes: List[Dict[str, Any]], target_keys: int) -> List[Dict[str, Any]]:
        """后处理音符，移除冲突和过密的音符"""
        if not notes:
            return notes
        
        # 按beat排序
        notes.sort(key=lambda x: x['beat'][0] * 4 * 24 + x['beat'][1])
        
        processed_notes = []
        last_beat_per_track = {}
        min_interval = 6  # 增加最小间隔：2 → 4（24分音符单位）
        
        for note in notes:
            track = note['column']
            current_beat_pos = note['beat'][0] * 4 * 24 + note['beat'][1]
            
            # 检查同轨道音符间隔
            if track in last_beat_per_track:
                if current_beat_pos - last_beat_per_track[track] < min_interval:
                    continue  # 跳过过近的音符
            
            processed_notes.append(note)
            last_beat_per_track[track] = current_beat_pos
        
        return processed_notes
        
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
                    # 对于song_4833，我们知道是Hypernova
                    if song_id == '4833':
                        song_title = "Hypernova"
                        artist = "A4paper"
                    else:
                        song_title = f"Song {song_id}"
                        artist = "Unknown Artist"
                else:
                    artist = "Unknown Artist"
                
                return {
                    'title': song_title,
                    'artist': artist,
                    'audio_path': audio_path,  # 提取后的实际路径
                    'temp_dir': temp_dir,
                    'original_audio_file': audio_path,  # 使用实际路径，不是压缩包内路径
                    'audio_filename': os.path.basename(audio_file)  # 保存原始文件名
                }
                
        except Exception as e:
            print(f"❌ MCZ解析失败: {e}")
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
            
    def generate_beatmap_standard_format(self, audio_features, target_difficulty, target_keys, song_data):
        """生成标准MC格式的谱面 - 使用AI模型"""
        print("🎼 使用AI模型生成标准MC格式谱面...")
        
        # 获取基本参数
        duration = float(audio_features.get('duration', 120))
        # 使用标准文件的正确BPM
        tempo = 156.0  # 使用标准文件的BPM，不是音频分析的BPM
        
        print(f"🎼 生成参数:")
        print(f"   时长: {duration:.1f}秒")
        print(f"   BPM: {tempo:.1f}")
        print(f"   轨道数: {target_keys}")
        print(f"   难度: {target_difficulty}")
        
        # 使用AI模型生成notes
        ai_notes = self.ai_generate_notes(
            audio_path=song_data['original_audio_file'],
            duration=duration,
            tempo=tempo,
            target_keys=target_keys
        )
        
        # 创建完整的notes列表
        notes = []
        
        # 添加AI生成的游戏音符
        notes.extend(ai_notes)
        
        # 最后添加音频控制音符（关键！只需要1个，包含type和offset）
        audio_filename = song_data.get('audio_filename', os.path.basename(song_data['original_audio_file']))
        audio_control_note = {
            'beat': [0, 0, 4],  # 使用4作为分母，不是24
            'sound': audio_filename,
            'vol': 100,
            'offset': 0,  # 从头开始播放，解决时长问题
            'type': 1  # 关键参数：自动播放音乐
        }
        notes.append(audio_control_note)
        
        print(f"✅ 总音符数: {len(notes)} (包含1个音频控制音符)")
        print(f"🎮 游戏音符数: {len(ai_notes)}")
        print(f"📊 平均密度: {len(ai_notes)/duration:.2f} 个/秒")
        
        # 创建标准MC格式谱面
        difficulty_name = f"4K AI Lv.{target_difficulty}"
        
        mc_data = {
            "meta": {
                "creator": "AI Beatmap Generator",
                "version": difficulty_name,
                "id": random.randint(100000, 999999),
                "mode": 0,  # 0 = osu!mania style
                "song": {
                    "title": song_data['title'],
                    "artist": song_data.get('artist', 'Unknown'),
                    "id": song_data.get('song_id', 0)
                },
                "mode_ext": {
                    "column": target_keys,
                    "bar_begin": 0,
                    "divide": 24  # 24分音符精度
                }
            },
            "time": [
                {
                    "beat": [0, 0, 1],
                    "bpm": tempo
                }
            ],
            "note": notes
        }
        
        return mc_data
    
    def create_standard_mcz_package(self, song_data, mc_data, output_path):
        """创建标准格式的MCZ包"""
        try:
            print(f"📦 创建标准MCZ包: {output_path}")
            
            # 创建临时目录
            with tempfile.TemporaryDirectory() as temp_dir:
                # 准备文件路径 - 使用匹配的文件名
                audio_filename = song_data.get('audio_filename', os.path.basename(song_data['original_audio_file']))
                audio_basename = os.path.splitext(audio_filename)[0]
                mc_filename = f"{audio_basename}.mc"
                mc_file_path = os.path.join(temp_dir, mc_filename)
                
                # 复制音频文件
                temp_audio_path = os.path.join(temp_dir, audio_filename)
                shutil.copy2(song_data['audio_path'], temp_audio_path)
                
                # 创建MC文件
                with open(mc_file_path, 'w', encoding='utf-8') as f:
                    json.dump(mc_data, f, ensure_ascii=False, separators=(',', ':'))
                
                print(f"🎵 音频文件: {audio_filename}")
                print(f"📄 MC文件: {mc_filename}")
                print(f"💡 使用匹配的文件名约定")
                
                # 创建ZIP包
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # 添加文件到0/目录下（模拟标准结构）
                    zipf.write(mc_file_path, f"0/{mc_filename}")
                    zipf.write(temp_audio_path, f"0/{audio_filename}")
                    
            print(f"✅ 标准MCZ包创建完成: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ MCZ包创建失败: {e}")
            return False

def main():
    """主函数"""
    import sys
    
    # 如果提供了命令行参数，使用它，否则使用默认值
    if len(sys.argv) > 1:
        input_mcz = sys.argv[1]
    else:
        input_mcz = "trainData/_song_10088.mcz"
    
    # 输出文件名基于输入文件
    input_basename = os.path.splitext(os.path.basename(input_mcz))[0]
    output_mcz = f"generated_beatmaps/ai_{input_basename}.mcz"
    target_difficulty = 15  # 降低难度：20 → 15
    target_keys = 4  # 4K模式：轨道索引0,1,2,3
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_mcz), exist_ok=True)
    
    print(f"🎮 AI谱面生成器")
    print(f"📥 输入: {input_mcz}")
    print(f"📤 输出: {output_mcz}")
    print(f"🎯 难度: {target_difficulty}")
    print(f"🎹 轨道: {target_keys}K")
    
    try:
        # 创建修复的生成器
        generator = FixedBeatmapGenerator()
        
        # 提取歌曲信息
        song_data = generator.extract_mcz_info(input_mcz)
        if not song_data:
            raise ValueError("无法解析MCZ文件")
            
        print(f"📖 歌曲信息:")
        print(f"   标题: {song_data['title']}")
        print(f"   艺术家: {song_data['artist']}")
        
        # 提取音频特征
        print("🔍 提取音频特征...")
        audio_features = generator.extract_audio_features(song_data['audio_path'])
        
        # 生成高密度谱面
        print("🎼 生成高密度谱面...")
        mc_data = generator.generate_beatmap_standard_format(
            audio_features, target_difficulty, target_keys, song_data
        )
        
        if mc_data:
            # 创建标准MCZ包
            success = generator.create_standard_mcz_package(song_data, mc_data, output_mcz)
            
            if success:
                print(f"\n🎉 高密度谱面生成成功！")
                print(f"📁 输出文件: {output_mcz}")
                print(f"🎵 歌曲: {song_data['title']}")
                print(f"🎯 难度: {mc_data['meta']['version']}")
                print(f"🎼 总音符数量: {len(mc_data['note'])}")
                print(f"🎮 游戏音符数量: {len(mc_data['note']) - 1}")
                print(f"🔧 格式: 标准MC格式 (24分音符精度)")
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
