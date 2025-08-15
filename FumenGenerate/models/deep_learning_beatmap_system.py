#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于深度学习的音游谱面生成系统

使用大规模数据集(700+ MCZ文件)进行深度学习训练
实现序列到序列的音频-谱面映射
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class BeatmapDataset(Dataset):
    """谱面数据集类"""
    
    def __init__(self, audio_features: np.ndarray, beatmap_labels: np.ndarray, 
                 sequence_length: int = 64):
        """
        Args:
            audio_features: 音频特征序列 [N, feature_dim]
            beatmap_labels: 谱面标签 [N, 4+3] (4轨道+3事件类型)
            sequence_length: 序列长度（时间步数）
        """
        self.sequence_length = sequence_length
        self.audio_features = torch.FloatTensor(audio_features)
        self.beatmap_labels = torch.FloatTensor(beatmap_labels)
        
        # 确保数据长度足够创建序列
        self.valid_indices = list(range(sequence_length, len(audio_features)))
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        
        # 提取序列
        start_idx = actual_idx - self.sequence_length
        end_idx = actual_idx
        
        audio_seq = self.audio_features[start_idx:end_idx]  # [seq_len, feature_dim]
        beatmap_target = self.beatmap_labels[actual_idx]    # [7] (4轨道+3事件类型)
        
        return audio_seq, beatmap_target


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # 线性变换
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.W_o(context)


class TransformerBeatmapGenerator(nn.Module):
    """基于Transformer的谱面生成器"""
    
    def __init__(self, input_dim: int = 15, d_model: int = 256, 
                 num_heads: int = 8, num_layers: int = 6, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        # Transformer层
        self.transformer_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': MultiHeadAttention(d_model, num_heads, dropout),
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
            for _ in range(num_layers)
        ])
        
        # 输出头
        self.note_placement_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4),  # 4个轨道的音符放置概率
            nn.Sigmoid()
        )
        
        self.event_type_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),  # note, long_start, long_end
            nn.Softmax(dim=-1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            note_probs: [batch_size, 4] - 各轨道音符概率
            event_probs: [batch_size, 3] - 事件类型概率
        """
        batch_size, seq_len, _ = x.size()
        
        # 输入投影
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # 添加位置编码
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        x = self.dropout(x)
        
        # Transformer层
        for layer in self.transformer_layers:
            # 多头注意力
            attn_out = layer['attention'](x)
            x = layer['norm1'](x + layer['dropout'](attn_out))
            
            # 前馈网络
            ff_out = layer['ff'](x)
            x = layer['norm2'](x + layer['dropout'](ff_out))
        
        # 使用最后一个时间步的输出
        final_hidden = x[:, -1, :]  # [batch_size, d_model]
        
        # 输出预测
        note_probs = self.note_placement_head(final_hidden)     # [batch_size, 4]
        event_probs = self.event_type_head(final_hidden)        # [batch_size, 3]
        
        return note_probs, event_probs


class DeepBeatmapLearningSystem:
    """深度学习谱面生成系统"""
    
    def __init__(self, sequence_length: int = 64, batch_size: int = 64, 
                 learning_rate: float = 0.001, device: str = None):
        """
        Args:
            sequence_length: 输入序列长度
            batch_size: 批次大小
            learning_rate: 学习率
            device: 计算设备
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # 设备选择
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🚀 使用设备: {self.device}")
        
        # 模型和优化器
        self.model = None
        self.optimizer = None
        self.scaler = StandardScaler()
        
        # 训练历史
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'note_accuracy': [],
            'event_accuracy': []
        }
    
    def create_model(self, input_dim: int = 15):
        """创建深度学习模型"""
        self.model = TransformerBeatmapGenerator(
            input_dim=input_dim,
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
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        print(f"📊 模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def load_large_dataset(self, traindata_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载大规模数据集（700+ MCZ文件）
        
        Args:
            traindata_dir: 训练数据目录
            
        Returns:
            (audio_features, beatmap_labels): 音频特征和谱面标签
        """
        print("🔍 扫描大规模数据集...")
        
        # 导入必要的模块
        from core.mcz_parser import MCZParser
        from core.four_k_extractor import FourKBeatmapExtractor
        from core.audio_beatmap_analyzer import AudioBeatmapAnalyzer
        
        parser = MCZParser()
        extractor = FourKBeatmapExtractor()
        analyzer = AudioBeatmapAnalyzer(time_resolution=0.05)
        
        # 扫描所有MCZ文件
        mcz_files = [f for f in os.listdir(traindata_dir) if f.endswith('.mcz')]
        print(f"📂 发现 {len(mcz_files)} 个MCZ文件")
        
        all_audio_features = []
        all_beatmap_labels = []
        processed_count = 0
        target_count = min(100, len(mcz_files))  # 先处理100个文件进行测试
        
        for i, mcz_file in enumerate(mcz_files[:target_count]):
            try:
                mcz_path = os.path.join(traindata_dir, mcz_file)
                print(f"⚡ 处理 [{i+1}/{target_count}]: {mcz_file}")
                
                # 解析MCZ文件
                song_data = parser.parse_mcz_file(mcz_path)
                if not song_data:
                    continue
                
                # 提取4K谱面
                beatmaps_4k = extractor.extract_4k_beatmaps(song_data)
                if not beatmaps_4k:
                    continue
                
                # 提取音频文件到临时目录
                temp_audio_dir = "temp_audio_extraction"
                os.makedirs(temp_audio_dir, exist_ok=True)
                
                extracted_audio = parser.extract_audio_files(mcz_path, temp_audio_dir)
                if not extracted_audio:
                    continue
                
                # 处理每个4K谱面
                for beatmap in beatmaps_4k:
                    for audio_file in extracted_audio:
                        try:
                            # 分析音频和谱面
                            aligned_data = analyzer.align_audio_and_beatmap(
                                audio_file, beatmap, {}
                            )
                            
                            if aligned_data and len(aligned_data.audio_features) > self.sequence_length:
                                # 添加到训练数据
                                audio_features = aligned_data.audio_features
                                beatmap_events = aligned_data.beatmap_events
                                
                                all_audio_features.append(audio_features)
                                all_beatmap_labels.append(beatmap_events)
                                
                                processed_count += 1
                                break  # 每个谱面只用一个音频文件
                                
                        except Exception as e:
                            print(f"   ⚠️ 处理音频失败: {e}")
                            continue
                
                # 清理临时文件
                import shutil
                if os.path.exists(temp_audio_dir):
                    shutil.rmtree(temp_audio_dir)
                    
            except Exception as e:
                print(f"   ❌ 处理MCZ失败: {e}")
                continue
        
        print(f"✅ 成功处理 {processed_count} 个谱面样本")
        
        if processed_count == 0:
            raise ValueError("没有成功处理任何数据，请检查数据格式")
        
        # 合并所有数据
        audio_features = np.vstack(all_audio_features)
        beatmap_labels = np.vstack(all_beatmap_labels)
        
        print(f"📊 数据集大小: {audio_features.shape[0]:,} 个时间步")
        print(f"🎵 音频特征维度: {audio_features.shape[1]}")
        print(f"🎮 谱面标签维度: {beatmap_labels.shape[1]}")
        
        return audio_features, beatmap_labels
    
    def prepare_training_data(self, audio_features: np.ndarray, beatmap_labels: np.ndarray, 
                            train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        """
        准备训练和验证数据
        
        Args:
            audio_features: 音频特征
            beatmap_labels: 谱面标签
            train_ratio: 训练集比例
            
        Returns:
            (train_loader, val_loader): 训练和验证数据加载器
        """
        print("🔧 准备深度学习训练数据...")
        
        # 标准化音频特征
        audio_features_scaled = self.scaler.fit_transform(audio_features)
        
        # 分割训练和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            audio_features_scaled, beatmap_labels, 
            test_size=1-train_ratio, random_state=42, shuffle=True
        )
        
        print(f"📈 训练集大小: {len(X_train):,}")
        print(f"📉 验证集大小: {len(X_val):,}")
        
        # 创建数据集
        train_dataset = BeatmapDataset(X_train, y_train, self.sequence_length)
        val_dataset = BeatmapDataset(X_val, y_val, self.sequence_length)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, 
            shuffle=False, num_workers=0, pin_memory=True
        )
        
        print(f"🎯 训练批次数: {len(train_loader)}")
        print(f"🎯 验证批次数: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def compute_loss(self, note_pred: torch.Tensor, event_pred: torch.Tensor, 
                    beatmap_target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算损失函数
        
        Args:
            note_pred: 音符预测 [batch_size, 4]
            event_pred: 事件预测 [batch_size, 3]
            beatmap_target: 目标标签 [batch_size, 7]
            
        Returns:
            (total_loss, loss_dict): 总损失和各项损失
        """
        # 分离目标标签
        note_target = beatmap_target[:, :4]      # 4个轨道
        event_target = beatmap_target[:, 4:]     # 3个事件类型
        
        # 音符放置损失（二元交叉熵）
        note_loss = F.binary_cross_entropy(note_pred, note_target, reduction='mean')
        
        # 事件类型损失（交叉熵）
        event_target_indices = torch.argmax(event_target, dim=1)
        event_loss = F.cross_entropy(event_pred, event_target_indices, reduction='mean')
        
        # 总损失（加权组合）
        total_loss = 0.7 * note_loss + 0.3 * event_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'note': note_loss.item(),
            'event': event_loss.item()
        }
        
        return total_loss, loss_dict
    
    def compute_accuracy(self, note_pred: torch.Tensor, event_pred: torch.Tensor, 
                        beatmap_target: torch.Tensor) -> Dict[str, float]:
        """计算准确率"""
        with torch.no_grad():
            # 音符放置准确率（阈值0.5）
            note_target = beatmap_target[:, :4]
            note_pred_binary = (note_pred > 0.5).float()
            note_accuracy = (note_pred_binary == note_target).float().mean().item()
            
            # 事件类型准确率
            event_target = beatmap_target[:, 4:]
            event_target_indices = torch.argmax(event_target, dim=1)
            event_pred_indices = torch.argmax(event_pred, dim=1)
            event_accuracy = (event_pred_indices == event_target_indices).float().mean().item()
            
            return {
                'note_accuracy': note_accuracy,
                'event_accuracy': event_accuracy
            }
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_note_accuracy = 0
        total_event_accuracy = 0
        num_batches = 0
        
        for batch_idx, (audio_seq, beatmap_target) in enumerate(train_loader):
            audio_seq = audio_seq.to(self.device)
            beatmap_target = beatmap_target.to(self.device)
            
            # 前向传播
            note_pred, event_pred = self.model(audio_seq)
            
            # 计算损失
            loss, loss_dict = self.compute_loss(note_pred, event_pred, beatmap_target)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 计算准确率
            accuracy_dict = self.compute_accuracy(note_pred, event_pred, beatmap_target)
            
            # 累计统计
            total_loss += loss_dict['total']
            total_note_accuracy += accuracy_dict['note_accuracy']
            total_event_accuracy += accuracy_dict['event_accuracy']
            num_batches += 1
            
            # 打印进度
            if batch_idx % 50 == 0:
                print(f"   批次 [{batch_idx:4d}/{len(train_loader)}] "
                      f"损失: {loss_dict['total']:.4f} "
                      f"音符准确率: {accuracy_dict['note_accuracy']:.3f} "
                      f"事件准确率: {accuracy_dict['event_accuracy']:.3f}")
        
        return {
            'loss': total_loss / num_batches,
            'note_accuracy': total_note_accuracy / num_batches,
            'event_accuracy': total_event_accuracy / num_batches
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        total_note_accuracy = 0
        total_event_accuracy = 0
        num_batches = 0
        
        with torch.no_grad():
            for audio_seq, beatmap_target in val_loader:
                audio_seq = audio_seq.to(self.device)
                beatmap_target = beatmap_target.to(self.device)
                
                # 前向传播
                note_pred, event_pred = self.model(audio_seq)
                
                # 计算损失和准确率
                loss, loss_dict = self.compute_loss(note_pred, event_pred, beatmap_target)
                accuracy_dict = self.compute_accuracy(note_pred, event_pred, beatmap_target)
                
                # 累计统计
                total_loss += loss_dict['total']
                total_note_accuracy += accuracy_dict['note_accuracy']
                total_event_accuracy += accuracy_dict['event_accuracy']
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'note_accuracy': total_note_accuracy / num_batches,
            'event_accuracy': total_event_accuracy / num_batches
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = 50, save_path: str = None):
        """
        训练深度学习模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            save_path: 模型保存路径
        """
        print(f"🚀 开始深度学习训练 ({num_epochs} 轮)")
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        for epoch in range(num_epochs):
            print(f"\n📊 Epoch [{epoch+1}/{num_epochs}]")
            
            # 训练
            train_metrics = self.train_epoch(train_loader)
            
            # 验证
            val_metrics = self.validate_epoch(val_loader)
            
            # 学习率调度
            self.scheduler.step(val_metrics['loss'])
            
            # 记录历史
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['note_accuracy'].append(val_metrics['note_accuracy'])
            self.training_history['event_accuracy'].append(val_metrics['event_accuracy'])
            
            # 打印结果
            print(f"🔥 训练 - 损失: {train_metrics['loss']:.4f}, "
                  f"音符准确率: {train_metrics['note_accuracy']:.3f}, "
                  f"事件准确率: {train_metrics['event_accuracy']:.3f}")
            print(f"✅ 验证 - 损失: {val_metrics['loss']:.4f}, "
                  f"音符准确率: {val_metrics['note_accuracy']:.3f}, "
                  f"事件准确率: {val_metrics['event_accuracy']:.3f}")
            
            # 早停检查
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                # 保存最佳模型
                if save_path:
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scaler': self.scaler,
                        'epoch': epoch,
                        'val_loss': val_metrics['loss']
                    }, save_path)
                    print(f"💾 保存最佳模型到: {save_path}")
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                print(f"⏰ 早停触发 (patience={max_patience})")
                break
        
        print("🎉 训练完成！")
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.training_history['train_loss'], label='训练损失')
        axes[0, 0].plot(self.training_history['val_loss'], label='验证损失')
        axes[0, 0].set_title('损失函数变化')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 音符准确率
        axes[0, 1].plot(self.training_history['note_accuracy'], label='音符准确率')
        axes[0, 1].set_title('音符放置准确率')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 事件准确率
        axes[1, 0].plot(self.training_history['event_accuracy'], label='事件准确率', color='orange')
        axes[1, 0].set_title('事件类型准确率')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 综合指标
        axes[1, 1].plot(self.training_history['val_loss'], label='验证损失', alpha=0.7)
        ax2 = axes[1, 1].twinx()
        ax2.plot(self.training_history['note_accuracy'], label='音符准确率', color='green', alpha=0.7)
        ax2.plot(self.training_history['event_accuracy'], label='事件准确率', color='orange', alpha=0.7)
        
        axes[1, 1].set_title('综合训练指标')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        ax2.set_ylabel('Accuracy')
        axes[1, 1].legend(loc='upper left')
        ax2.legend(loc='upper right')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('deep_learning_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 训练历史图表已保存为 'deep_learning_training_history.png'")


def main():
    """主函数：演示深度学习训练流程"""
    print("🎮 深度学习音游谱面生成系统")
    print("=" * 60)
    
    # 创建系统
    system = DeepBeatmapLearningSystem(
        sequence_length=64,
        batch_size=32,
        learning_rate=0.001
    )
    
    # 加载大规模数据集
    try:
        audio_features, beatmap_labels = system.load_large_dataset('trainData')
        
        # 创建模型
        system.create_model(input_dim=audio_features.shape[1])
        
        # 准备训练数据
        train_loader, val_loader = system.prepare_training_data(
            audio_features, beatmap_labels, train_ratio=0.8
        )
        
        # 开始训练
        system.train(
            train_loader, val_loader, 
            num_epochs=50,
            save_path='best_deep_beatmap_model.pth'
        )
        
        # 绘制训练历史
        system.plot_training_history()
        
        print("\n🎉 深度学习训练完成！")
        print("💾 最佳模型已保存为 'best_deep_beatmap_model.pth'")
        
    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
