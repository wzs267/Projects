#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度学习谱面生成推理系统
基于训练好的深度学习模型生成高质量谱面
"""

import os
import numpy as np
import torch
import librosa
import json
from typing import List, Dict, Any, Tuple
from deep_learning_beatmap_system import DeepBeatmapLearningSystem, TransformerBeatmapGenerator
import matplotlib.pyplot as plt

class DeepBeatmapGenerator:
    """基于深度学习的谱面生成器"""
    
    def __init__(self, model_path: str = None):
        """
        Args:
            model_path: 训练好的模型路径
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.sequence_length = 64
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """加载训练好的模型"""
        print(f"📥 加载深度学习模型: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 创建模型实例
        self.model = TransformerBeatmapGenerator(
            input_dim=15,  # 标准音频特征维度
            d_model=256,
            num_heads=8,
            num_layers=6,
            dropout=0.1
        ).to(self.device)
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 加载标准化器
        self.scaler = checkpoint['scaler']
        
        print(f"✅ 模型加载成功！")
        print(f"   📊 模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   🏆 训练轮次: {checkpoint.get('epoch', 'unknown')}")
        print(f"   📈 验证损失: {checkpoint.get('val_loss', 'unknown'):.4f}")
    
    def extract_audio_features(self, audio_file: str) -> np.ndarray:
        """
        提取音频特征序列
        
        Args:
            audio_file: 音频文件路径
            
        Returns:
            audio_features: [时间步, 15维特征]
        """
        print(f"🎵 分析音频: {os.path.basename(audio_file)}")
        
        # 加载音频
        try:
            y, sr = librosa.load(audio_file, sr=22050)
        except Exception as e:
            print(f"❌ 音频加载失败: {e}")
            return None
        
        print(f"   ⏱️ 音频时长: {len(y)/sr:.1f}秒")
        print(f"   🎼 采样率: {sr} Hz")
        
        # 计算时间网格（50ms分辨率）
        time_resolution = 0.05
        frame_length = int(sr * time_resolution)
        num_frames = len(y) // frame_length
        
        audio_features = []
        
        for i in range(num_frames):
            start_sample = i * frame_length
            end_sample = start_sample + frame_length
            frame = y[start_sample:end_sample]
            
            if len(frame) < frame_length:
                frame = np.pad(frame, (0, frame_length - len(frame)))
            
            # 提取15维音频特征
            features = self._extract_frame_features(frame, sr)
            audio_features.append(features)
        
        audio_features = np.array(audio_features)
        print(f"   📊 提取特征: {audio_features.shape[0]} 个时间步 × {audio_features.shape[1]} 维特征")
        
        return audio_features
    
    def _extract_frame_features(self, frame: np.ndarray, sr: int) -> np.ndarray:
        """提取单帧音频特征"""
        features = []
        
        # 1. RMS能量(dB)
        rms = librosa.feature.rms(y=frame)[0, 0]
        rms_db = 20 * np.log10(max(rms, 1e-8))
        features.append(rms_db)
        
        # 2. 音符起始强度
        onset_strength = librosa.onset.onset_strength(y=frame, sr=sr)[0] if len(frame) > 512 else 0
        features.append(onset_strength)
        
        # 3-7. MFCC特征 (前5个)
        try:
            mfccs = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=5)
            features.extend(mfccs[:, 0])
        except:
            features.extend([0.0] * 5)
        
        # 8. 频谱质心
        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=frame, sr=sr)[0, 0]
            features.append(spectral_centroid)
        except:
            features.append(0.0)
        
        # 9. 色度特征均值
        try:
            chroma = librosa.feature.chroma_stft(y=frame, sr=sr)
            chroma_mean = np.mean(chroma)
            features.append(chroma_mean)
        except:
            features.append(0.0)
        
        # 10. 过零率
        zcr = librosa.feature.zero_crossing_rate(frame)[0, 0]
        features.append(zcr)
        
        # 11. 频谱对比度
        try:
            contrast = librosa.feature.spectral_contrast(y=frame, sr=sr)[0, 0]
            features.append(contrast)
        except:
            features.append(0.0)
        
        # 12. BPM（使用固定值，因为单帧无法准确估计）
        features.append(120.0)  # 默认BPM
        
        # 13-14. 难度参数（在生成时设置）
        features.extend([0.5, 0.5])  # 默认中等难度
        
        return np.array(features, dtype=np.float32)
    
    def generate_beatmap_deep(self, audio_file: str, difficulty: str = 'Normal', 
                            note_threshold: float = 0.5) -> Dict[str, Any]:
        """
        使用深度学习模型生成谱面
        
        Args:
            audio_file: 音频文件路径
            difficulty: 难度级别
            note_threshold: 音符放置阈值
            
        Returns:
            生成结果字典
        """
        if self.model is None:
            raise ValueError("模型未加载，请先调用 load_model()")
        
        print(f"🎮 使用深度学习生成 {difficulty} 难度谱面")
        
        # 提取音频特征
        audio_features = self.extract_audio_features(audio_file)
        if audio_features is None:
            return None
        
        # 设置难度参数
        difficulty_params = self._get_difficulty_params(difficulty)
        
        # 更新音频特征中的难度参数
        audio_features[:, -2] = difficulty_params['note_density']
        audio_features[:, -1] = difficulty_params['note_threshold']
        
        # 标准化特征
        if self.scaler:
            audio_features = self.scaler.transform(audio_features)
        
        # 生成谱面事件
        generated_events = self._predict_beatmap_sequence(audio_features, note_threshold)
        
        # 计算统计信息
        note_events = [e for e in generated_events if e['type'] == 'note']
        long_events = [e for e in generated_events if e['type'] == 'long_start']
        
        result = {
            'audio_file': audio_file,
            'difficulty': difficulty,
            'audio_duration': len(audio_features) * 0.05,
            'generated_events': generated_events,
            'statistics': {
                'total_events': len(generated_events),
                'note_count': len(note_events),
                'long_note_count': len(long_events),
                'note_density': len(generated_events) / (len(audio_features) * 0.05),
                'difficulty_params': difficulty_params
            }
        }
        
        print(f"✅ 深度学习生成完成!")
        print(f"   🎵 生成音符: {len(note_events)} 个普通音符 + {len(long_events)} 个长条")
        print(f"   📊 音符密度: {result['statistics']['note_density']:.2f} 音符/秒")
        
        return result
    
    def _get_difficulty_params(self, difficulty: str) -> Dict[str, float]:
        """获取难度参数"""
        params = {
            'Easy': {'note_density': 0.3, 'note_threshold': 0.7},
            'Normal': {'note_density': 0.5, 'note_threshold': 0.6},
            'Hard': {'note_density': 0.7, 'note_threshold': 0.5},
            'Expert': {'note_density': 0.9, 'note_threshold': 0.4}
        }
        return params.get(difficulty, params['Normal'])
    
    def _predict_beatmap_sequence(self, audio_features: np.ndarray, 
                                note_threshold: float) -> List[Dict[str, Any]]:
        """使用深度学习模型预测谱面序列"""
        generated_events = []
        
        # 滑动窗口预测
        for i in range(self.sequence_length, len(audio_features)):
            # 提取序列
            start_idx = i - self.sequence_length
            audio_seq = audio_features[start_idx:i]
            
            # 转换为PyTorch张量
            audio_tensor = torch.FloatTensor(audio_seq).unsqueeze(0).to(self.device)
            
            # 模型预测
            with torch.no_grad():
                note_probs, event_probs = self.model(audio_tensor)
                
                # 转换为numpy数组
                note_probs = note_probs.cpu().numpy()[0]  # [4]
                event_probs = event_probs.cpu().numpy()[0]  # [3]
            
            # 决定是否放置音符
            current_time = i * 0.05
            
            # 检查每个轨道
            for column in range(4):
                if note_probs[column] > note_threshold:
                    # 确定事件类型
                    event_type_idx = np.argmax(event_probs)
                    event_types = ['note', 'long_start', 'long_end']
                    event_type = event_types[event_type_idx]
                    
                    # 创建事件
                    event = {
                        'time': current_time,
                        'column': column,
                        'type': event_type,
                        'confidence': float(note_probs[column]),
                        'event_confidence': float(event_probs[event_type_idx])
                    }
                    
                    # 为长条音符添加持续时间
                    if event_type == 'long_start':
                        event['duration'] = self._estimate_long_note_duration(
                            audio_features, i, note_probs[column]
                        )
                    
                    generated_events.append(event)
        
        return generated_events
    
    def _estimate_long_note_duration(self, audio_features: np.ndarray, 
                                   start_idx: int, start_confidence: float) -> float:
        """估计长条音符持续时间"""
        min_duration = 0.2  # 最小200ms
        max_duration = 2.0  # 最大2秒
        
        # 基于起始置信度估计基础持续时间
        base_duration = 0.3 + (start_confidence - 0.5) * 0.4
        
        # 检查后续音频特征的持续性
        duration = base_duration
        for i in range(start_idx + 1, min(start_idx + 40, len(audio_features))):  # 检查后续2秒
            frame_energy = audio_features[i, 0]  # RMS能量
            if frame_energy < audio_features[start_idx, 0] - 10:  # 能量显著下降
                break
            duration += 0.05
        
        return max(min_duration, min(duration, max_duration))
    
    def visualize_generated_beatmap(self, result: Dict[str, Any], save_path: str = None):
        """可视化生成的谱面"""
        events = result['generated_events']
        duration = result['audio_duration']
        
        # 创建时间轴
        time_points = [e['time'] for e in events]
        columns = [e['column'] for e in events]
        colors = []
        
        for e in events:
            if e['type'] == 'note':
                colors.append('blue')
            elif e['type'] == 'long_start':
                colors.append('red')
            else:
                colors.append('orange')
        
        # 绘制谱面图
        plt.figure(figsize=(15, 8))
        
        # 主谱面图
        plt.subplot(2, 1, 1)
        plt.scatter(time_points, columns, c=colors, alpha=0.7, s=50)
        plt.ylim(-0.5, 3.5)
        plt.yticks([0, 1, 2, 3], ['轨道1', '轨道2', '轨道3', '轨道4'])
        plt.xlabel('时间 (秒)')
        plt.ylabel('轨道')
        plt.title(f'{result["difficulty"]} 难度谱面生成结果')
        plt.grid(True, alpha=0.3)
        
        # 添加图例
        import matplotlib.patches as mpatches
        blue_patch = mpatches.Patch(color='blue', label='普通音符')
        red_patch = mpatches.Patch(color='red', label='长条开始')
        orange_patch = mpatches.Patch(color='orange', label='长条结束')
        plt.legend(handles=[blue_patch, red_patch, orange_patch])
        
        # 音符密度图
        plt.subplot(2, 1, 2)
        time_bins = np.arange(0, duration, 1.0)  # 1秒为单位
        density, _ = np.histogram(time_points, bins=time_bins)
        plt.bar(time_bins[:-1], density, width=0.8, alpha=0.7, color='green')
        plt.xlabel('时间 (秒)')
        plt.ylabel('音符数量')
        plt.title('音符密度分布 (每秒)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 谱面可视化已保存: {save_path}")
        
        plt.show()


def demo_deep_generation():
    """演示深度学习谱面生成"""
    print("🎮 深度学习谱面生成演示")
    print("=" * 50)
    
    # 创建生成器
    generator = DeepBeatmapGenerator()
    
    # 检查是否有训练好的模型
    model_files = [
        'large_scale_beatmap_model.pth',
        'best_deep_beatmap_model.pth',
        'test_model.pth'
    ]
    
    model_path = None
    for model_file in model_files:
        if os.path.exists(model_file):
            model_path = model_file
            break
    
    if model_path:
        # 加载模型
        generator.load_model(model_path)
        
        # 查找测试音频
        audio_files = []
        if os.path.exists('extracted_audio'):
            audio_files = [f for f in os.listdir('extracted_audio') if f.endswith('.ogg')]
        
        if audio_files:
            test_audio = os.path.join('extracted_audio', audio_files[0])
            print(f"\n🎵 使用测试音频: {test_audio}")
            
            # 生成不同难度的谱面
            difficulties = ['Easy', 'Normal', 'Hard', 'Expert']
            
            for difficulty in difficulties:
                print(f"\n🎯 生成 {difficulty} 难度谱面...")
                result = generator.generate_beatmap_deep(test_audio, difficulty)
                
                if result:
                    stats = result['statistics']
                    print(f"   📊 生成结果:")
                    print(f"      • 音符数量: {stats['note_count']}")
                    print(f"      • 长条数量: {stats['long_note_count']}")
                    print(f"      • 音符密度: {stats['note_density']:.2f} 音符/秒")
                    
                    # 可视化第一个难度的结果
                    if difficulty == 'Normal':
                        generator.visualize_generated_beatmap(
                            result, f'deep_generated_{difficulty.lower()}_beatmap.png'
                        )
        else:
            print("❌ 未找到测试音频文件")
    else:
        print("❌ 未找到训练好的模型文件")
        print("💡 请先运行大规模训练或测试训练")


if __name__ == "__main__":
    demo_deep_generation()
