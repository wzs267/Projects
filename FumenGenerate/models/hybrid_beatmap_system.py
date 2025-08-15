#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合深度学习系统：结合随机森林特征工程 + Transformer序列建模

核心创新：
1. 使用随机森林模型的特征重要性指导特征选择
2. 集成随机森林预测作为Transformer的额外输入
3. 多尺度特征提取：短期（单帧）+ 长期（序列）
4. 专家系统融合：传统ML + 深度学习
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from models.beatmap_learning_system import BeatmapLearningSystem
import warnings
warnings.filterwarnings('ignore')

class EnhancedFeatureExtractor:
    """增强特征提取器：结合音频分析和随机森林特征工程"""
    
    def __init__(self):
        self.rf_system = BeatmapLearningSystem()
        self.feature_scaler = StandardScaler()
        self.rf_models_trained = False
    
    def train_rf_feature_extractors(self, training_data_path: str):
        """训练随机森林特征提取器"""
        print("🌳 训练随机森林特征提取器...")
        
        # 使用现有的训练数据
        aligned_datasets = self.rf_system.collect_training_data(
            training_data_path, 'extracted_audio'
        )
        
        if not aligned_datasets:
            print("❌ 无法加载训练数据")
            return False
        
        # 准备机器学习数据
        X, y_note, y_column, y_long = self.rf_system.prepare_machine_learning_data(aligned_datasets)
        
        # 训练随机森林模型
        self.rf_system.train_models(X, y_note, y_column, y_long)
        
        # 拟合特征标准化器
        self.feature_scaler.fit(X)
        
        self.rf_models_trained = True
        print("✅ 随机森林特征提取器训练完成")
        return True
    
    def extract_enhanced_features(self, audio_features: np.ndarray, 
                                difficulty_params: Dict[str, float]) -> np.ndarray:
        """
        提取增强特征：原始音频特征 + 随机森林预测 + 特征工程
        
        Args:
            audio_features: 原始15维音频特征 [N, 15]
            difficulty_params: 难度参数
            
        Returns:
            enhanced_features: 增强特征 [N, enhanced_dim]
        """
        if not self.rf_models_trained:
            print("⚠️ 随机森林模型未训练，返回原始特征")
            return audio_features
        
        # 1. 标准化原始特征
        normalized_features = self.feature_scaler.transform(audio_features)
        
        # 2. 随机森林预测作为特征
        rf_note_probs = self.rf_system.note_placement_model.predict_proba(normalized_features)
        rf_note_features = rf_note_probs[:, 1:2]  # 音符放置概率
        
        # 只对有音符的位置预测轨道
        rf_column_features = np.zeros((len(audio_features), 4))
        has_note_mask = rf_note_features.flatten() > 0.5
        
        if np.sum(has_note_mask) > 0:
            rf_column_probs = self.rf_system.column_selection_model.predict_proba(
                normalized_features[has_note_mask]
            )
            rf_column_features[has_note_mask] = rf_column_probs
        
        rf_long_probs = self.rf_system.long_note_model.predict_proba(normalized_features)
        rf_long_features = rf_long_probs[:, 1:2]  # 长条音符概率
        
        # 3. 时序特征工程
        temporal_features = self._extract_temporal_features(normalized_features)
        
        # 4. 音乐理论特征
        music_theory_features = self._extract_music_theory_features(normalized_features)
        
        # 5. 统计特征
        statistical_features = self._extract_statistical_features(normalized_features)
        
        # 6. 难度相关特征
        difficulty_features = self._extract_difficulty_features(
            normalized_features, difficulty_params
        )
        
        # 合并所有特征
        enhanced_features = np.concatenate([
            normalized_features,      # 15维：原始音频特征
            rf_note_features,        # 1维：RF音符预测
            rf_column_features,      # 4维：RF轨道预测
            rf_long_features,        # 1维：RF长条预测
            temporal_features,       # 10维：时序特征
            music_theory_features,   # 8维：音乐理论特征
            statistical_features,    # 6维：统计特征
            difficulty_features      # 5维：难度特征
        ], axis=1)
        
        return enhanced_features
    
    def _extract_temporal_features(self, features: np.ndarray) -> np.ndarray:
        """提取时序特征"""
        N = len(features)
        temporal_features = np.zeros((N, 10))
        
        # 使用RMS能量（第一个特征）进行时序分析
        rms_energy = features[:, 0]
        
        for i in range(N):
            # 当前帧的时序特征
            window_size = min(10, i + 1, N - i)
            start_idx = max(0, i - window_size // 2)
            end_idx = min(N, i + window_size // 2 + 1)
            window_energy = rms_energy[start_idx:end_idx]
            
            # 1-3. 能量统计
            temporal_features[i, 0] = np.mean(window_energy)      # 局部平均能量
            temporal_features[i, 1] = np.std(window_energy)       # 局部能量方差
            temporal_features[i, 2] = np.max(window_energy) - np.min(window_energy)  # 能量范围
            
            # 4-5. 能量变化
            if i > 0:
                temporal_features[i, 3] = rms_energy[i] - rms_energy[i-1]  # 一阶差分
            if i > 1:
                temporal_features[i, 4] = temporal_features[i, 3] - temporal_features[i-1, 3]  # 二阶差分
            
            # 6-7. 趋势特征
            if len(window_energy) > 3:
                # 线性回归斜率
                x = np.arange(len(window_energy))
                slope = np.polyfit(x, window_energy, 1)[0]
                temporal_features[i, 5] = slope
                
                # 能量峰值检测
                peaks = (window_energy[1:-1] > window_energy[:-2]) & (window_energy[1:-1] > window_energy[2:])
                temporal_features[i, 6] = np.sum(peaks) / len(window_energy)
            
            # 8-10. 节拍相关
            beat_phase = (i * 0.05) % 1.0  # 假设1秒周期
            temporal_features[i, 7] = np.sin(2 * np.pi * beat_phase)     # 节拍相位sin
            temporal_features[i, 8] = np.cos(2 * np.pi * beat_phase)     # 节拍相位cos
            temporal_features[i, 9] = beat_phase                         # 节拍相位线性
        
        return temporal_features
    
    def _extract_music_theory_features(self, features: np.ndarray) -> np.ndarray:
        """提取音乐理论特征"""
        N = len(features)
        music_features = np.zeros((N, 8))
        
        # 使用MFCC特征进行音乐分析
        mfcc_features = features[:, 4:9]  # MFCC 1-5
        spectral_centroid = features[:, 6]  # 频谱质心
        
        for i in range(N):
            # 1-2. 和声特征
            if i >= 4:
                # 和弦稳定性（MFCC变化小表示和弦稳定）
                mfcc_window = mfcc_features[i-4:i+1]
                music_features[i, 0] = 1.0 / (1.0 + np.std(mfcc_window))
                
                # 音色一致性
                centroid_window = spectral_centroid[i-4:i+1]
                music_features[i, 1] = 1.0 / (1.0 + np.std(centroid_window))
            
            # 3-4. 旋律特征
            if i > 0:
                # 旋律方向
                centroid_change = spectral_centroid[i] - spectral_centroid[i-1]
                music_features[i, 2] = np.tanh(centroid_change / 1000)  # 标准化
                
                # MFCC1变化（音色变化）
                mfcc1_change = mfcc_features[i, 0] - mfcc_features[i-1, 0]
                music_features[i, 3] = np.tanh(mfcc1_change)
            
            # 5-6. 节奏特征
            # 强拍位置（基于能量）
            beat_position = (i * 0.05) % 1.0
            music_features[i, 4] = 1.0 if beat_position < 0.1 else 0.0  # 强拍
            music_features[i, 5] = 1.0 if 0.4 < beat_position < 0.6 else 0.0  # 弱拍
            
            # 7-8. 复杂度特征
            # MFCC复杂度（所有MFCC的标准差）
            music_features[i, 6] = np.std(mfcc_features[i])
            
            # 频谱复杂度（基于频谱质心的相对位置）
            music_features[i, 7] = spectral_centroid[i] / 4000.0  # 标准化到[0,1]
        
        return music_features
    
    def _extract_statistical_features(self, features: np.ndarray) -> np.ndarray:
        """提取统计特征"""
        N = len(features)
        stat_features = np.zeros((N, 6))
        
        window_size = 20  # 1秒窗口
        
        for i in range(N):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(N, i + window_size // 2 + 1)
            window_features = features[start_idx:end_idx]
            
            if len(window_features) > 1:
                # 1. 能量百分位数
                energies = window_features[:, 0]
                stat_features[i, 0] = np.percentile(energies, 75) - np.percentile(energies, 25)
                
                # 2. 特征相关性（RMS与频谱质心）
                if len(window_features) > 3:
                    correlation = np.corrcoef(window_features[:, 0], window_features[:, 6])[0, 1]
                    stat_features[i, 1] = correlation if not np.isnan(correlation) else 0.0
                
                # 3-4. 多维特征分布
                # 使用PCA的第一主成分方差（复杂度指标）
                centered_features = window_features - np.mean(window_features, axis=0)
                if centered_features.shape[0] > centered_features.shape[1]:
                    cov_matrix = np.cov(centered_features.T)
                    eigenvals = np.linalg.eigvals(cov_matrix)
                    stat_features[i, 2] = np.max(eigenvals) / np.sum(eigenvals)  # 主成分贡献率
                    stat_features[i, 3] = np.sum(eigenvals > 0.1)  # 有效维度数
                
                # 5-6. 时序稳定性
                if len(window_features) > 5:
                    # 特征变化率
                    feature_changes = np.diff(window_features, axis=0)
                    stat_features[i, 4] = np.mean(np.std(feature_changes, axis=0))
                    
                    # 周期性检测（自相关）
                    autocorr = np.correlate(energies, energies, mode='full')
                    autocorr = autocorr[autocorr.size // 2:]
                    if len(autocorr) > 2:
                        stat_features[i, 5] = np.max(autocorr[1:3]) / autocorr[0]
        
        return stat_features
    
    def _extract_difficulty_features(self, features: np.ndarray, 
                                   difficulty_params: Dict[str, float]) -> np.ndarray:
        """提取难度相关特征"""
        N = len(features)
        diff_features = np.zeros((N, 5))
        
        note_density = difficulty_params.get('note_density', 0.5)
        note_threshold = difficulty_params.get('note_threshold', 0.5)
        
        for i in range(N):
            # 1. 难度参数直接特征
            diff_features[i, 0] = note_density
            diff_features[i, 1] = note_threshold
            
            # 2. 能量与难度的交互特征
            energy = features[i, 0]
            diff_features[i, 2] = energy * note_density  # 能量-密度交互
            
            # 3. 复杂度调整
            complexity = np.std(features[i, 4:9])  # MFCC标准差作为复杂度
            diff_features[i, 3] = complexity * (1 + note_density)
            
            # 4. 自适应阈值
            if i >= 10:
                recent_energy = features[i-10:i, 0]
                adaptive_threshold = np.percentile(recent_energy, 70) * note_threshold
                diff_features[i, 4] = adaptive_threshold
            else:
                diff_features[i, 4] = note_threshold
        
        return diff_features


class HybridTransformerGenerator(nn.Module):
    """混合Transformer生成器：多尺度特征 + 专家融合"""
    
    def __init__(self, enhanced_input_dim: int = 50, d_model: int = 256,
                 num_heads: int = 8, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        
        self.enhanced_input_dim = enhanced_input_dim
        self.d_model = d_model
        
        # 多路径特征处理
        self.audio_projection = nn.Linear(15, d_model // 4)      # 原始音频特征
        self.rf_projection = nn.Linear(6, d_model // 4)          # RF预测特征  
        self.temporal_projection = nn.Linear(10, d_model // 4)   # 时序特征
        self.context_projection = nn.Linear(enhanced_input_dim - 31, d_model // 4)  # 其他特征
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        # 多尺度注意力
        self.short_attention = self._make_transformer_layer(d_model, num_heads, dropout)  # 短期模式
        self.long_attention = self._make_transformer_layer(d_model, num_heads, dropout)   # 长期模式
        
        # 主Transformer层
        self.transformer_layers = nn.ModuleList([
            self._make_transformer_layer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 专家系统融合
        self.expert_gate = nn.Sequential(
            nn.Linear(d_model, 3),
            nn.Softmax(dim=-1)
        )
        
        # 专家网络
        self.rf_expert = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64)
        )
        
        self.transformer_expert = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64)
        )
        
        self.fusion_expert = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64)
        )
        
        # 输出头
        self.note_placement_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 4),
            nn.Sigmoid()
        )
        
        self.event_type_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3),
            nn.Softmax(dim=-1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def _make_transformer_layer(self, d_model: int, num_heads: int, dropout: float):
        """创建Transformer层"""
        return nn.ModuleDict({
            'attention': nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True),
            'norm1': nn.LayerNorm(d_model),
            'ff': nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model)
            ),
            'norm2': nn.LayerNorm(d_model),
            'dropout': nn.Dropout(dropout)
        })
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, enhanced_input_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # 分离不同类型的特征
        audio_features = x[:, :, :15]           # 原始音频特征
        rf_features = x[:, :, 15:21]            # RF预测特征
        temporal_features = x[:, :, 21:31]      # 时序特征
        context_features = x[:, :, 31:]         # 其他特征
        
        # 多路径特征投影
        audio_emb = self.audio_projection(audio_features)
        rf_emb = self.rf_projection(rf_features)
        temporal_emb = self.temporal_projection(temporal_features)
        context_emb = self.context_projection(context_features)
        
        # 特征融合
        combined_features = torch.cat([audio_emb, rf_emb, temporal_emb, context_emb], dim=-1)
        fused_features = self.feature_fusion(combined_features)
        
        # 添加位置编码
        x = fused_features + self.pos_encoding[:seq_len].unsqueeze(0)
        x = self.dropout(x)
        
        # 多尺度注意力
        # 短期注意力（局部模式）
        short_attn, _ = self.short_attention['attention'](x, x, x)
        x_short = self.short_attention['norm1'](x + self.short_attention['dropout'](short_attn))
        ff_short = self.short_attention['ff'](x_short)
        x_short = self.short_attention['norm2'](x_short + self.short_attention['dropout'](ff_short))
        
        # 长期注意力（全局模式）
        long_attn, _ = self.long_attention['attention'](x, x, x)
        x_long = self.long_attention['norm1'](x + self.long_attention['dropout'](long_attn))
        ff_long = self.long_attention['ff'](x_long)
        x_long = self.long_attention['norm2'](x_long + self.long_attention['dropout'](ff_long))
        
        # 多尺度融合
        x = (x_short + x_long) / 2
        
        # 主Transformer层
        for layer in self.transformer_layers:
            attn_out, _ = layer['attention'](x, x, x)
            x = layer['norm1'](x + layer['dropout'](attn_out))
            ff_out = layer['ff'](x)
            x = layer['norm2'](x + layer['dropout'](ff_out))
        
        # 使用最后时间步的输出
        final_hidden = x[:, -1, :]  # [batch_size, d_model]
        
        # 专家系统
        gate_weights = self.expert_gate(final_hidden)  # [batch_size, 3]
        
        rf_output = self.rf_expert(final_hidden)
        transformer_output = self.transformer_expert(final_hidden)
        fusion_output = self.fusion_expert(final_hidden)
        
        # 专家融合
        expert_outputs = torch.stack([rf_output, transformer_output, fusion_output], dim=2)  # [batch_size, 64, 3]
        final_output = torch.sum(expert_outputs * gate_weights.unsqueeze(1), dim=2)  # [batch_size, 64]
        
        # 最终预测
        note_probs = self.note_placement_head(final_output)
        event_probs = self.event_type_head(final_output)
        
        return note_probs, event_probs, gate_weights


class HybridBeatmapLearningSystem:
    """混合深度学习系统：RF特征工程 + Transformer序列建模"""
    
    def __init__(self, sequence_length: int = 64, batch_size: int = 32,
                 learning_rate: float = 0.0005, device: str = None):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 混合系统使用设备: {self.device}")
        
        # 特征提取器
        self.feature_extractor = EnhancedFeatureExtractor()
        
        # 模型
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # 训练历史
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'note_accuracy': [],
            'event_accuracy': [],
            'expert_weights': []
        }
    
    def prepare_hybrid_training(self, traindata_dir: str, training_data_path: str):
        """准备混合训练：先训练RF，再准备深度学习数据"""
        print("🔄 准备混合深度学习训练...")
        
        # 1. 训练随机森林特征提取器
        success = self.feature_extractor.train_rf_feature_extractors(training_data_path)
        if not success:
            return None, None
        
        # 2. 加载和增强特征
        print("🔧 加载并增强特征...")
        # 这里需要重新实现数据加载，使用增强特征
        # 为简化演示，我们使用模拟数据
        
        return self._create_mock_enhanced_data()
    
    def _create_mock_enhanced_data(self):
        """创建模拟增强数据用于演示"""
        print("🎲 创建模拟增强数据...")
        
        # 模拟增强特征：50维
        num_samples = 5000
        enhanced_dim = 50
        
        enhanced_features = np.random.randn(num_samples, enhanced_dim).astype(np.float32)
        beatmap_labels = np.zeros((num_samples, 7), dtype=np.float32)
        
        # 模拟真实的音符模式
        for i in range(0, num_samples, 15):
            if np.random.random() > 0.6:
                column = np.random.randint(0, 4)
                beatmap_labels[i, column] = 1.0
                event_type = np.random.choice([0, 1, 2], p=[0.8, 0.15, 0.05])
                beatmap_labels[i, 4 + event_type] = 1.0
        
        return enhanced_features, beatmap_labels
    
    def create_hybrid_model(self, enhanced_input_dim: int = 50):
        """创建混合模型"""
        print(f"🏗️ 创建混合Transformer模型 (输入维度: {enhanced_input_dim})")
        
        self.model = HybridTransformerGenerator(
            enhanced_input_dim=enhanced_input_dim,
            d_model=256,
            num_heads=8,
            num_layers=6,
            dropout=0.1
        ).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        print(f"📊 混合模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")


def demo_hybrid_system():
    """演示混合系统"""
    print("🎮 混合深度学习系统演示")
    print("=" * 60)
    
    # 创建混合系统
    hybrid_system = HybridBeatmapLearningSystem(
        sequence_length=32,
        batch_size=16,
        learning_rate=0.001
    )
    
    # 准备训练（使用模拟数据）
    print("🔧 准备混合训练数据...")
    enhanced_features, beatmap_labels = hybrid_system._create_mock_enhanced_data()
    
    # 创建混合模型
    hybrid_system.create_hybrid_model(enhanced_input_dim=enhanced_features.shape[1])
    
    print("✅ 混合深度学习系统创建成功！")
    print(f"🎯 核心优势:")
    print(f"   • 结合随机森林的特征工程智慧")
    print(f"   • 利用Transformer的序列建模能力") 
    print(f"   • 多尺度注意力机制")
    print(f"   • 专家系统融合预测")
    print(f"   • 增强特征维度: {enhanced_features.shape[1]}维")
    
    return hybrid_system


if __name__ == "__main__":
    hybrid_system = demo_hybrid_system()
