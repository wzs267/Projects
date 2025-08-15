#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
权重融合快速训练脚本 (RF:NN = 2:8)
使用预处理数据进行高效训练
"""

import os
import sys
import time
im    def create_weighted_fusion_model(self, input_dim: int = 15):
        """创建权重融合模型 - 与完整版本对齐"""
        print("🏗️ 创建权重融合模型...")
        
        self.model = WeightedFusionTransformer(
            input_dim=input_dim,
            d_model=256,          # 与完整版本对齐：更大的特征维度
            num_heads=8,          # 与完整版本对齐：更多注意力头
            num_layers=6,         # 与完整版本对齐：更深的网络
            dropout=0.1,
            rf_weight=self.rf_weight,
            nn_weight=self.nn_weight,
            learnable_weights=True
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01      # 添加权重衰减
        )
        
        # 学习率调度器 - 与完整版本对齐
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )ch
import torch.nn as nn
import numpy as np
import json
from models.deep_learning_beatmap_system import DeepBeatmapLearningSystem
from models.weighted_fusion_model import WeightedFusionTransformer

class WeightedFusionQuickSystem(DeepBeatmapLearningSystem):
    """权重融合快速训练系统"""
    
    def __init__(self, rf_weight=0.3, nn_weight=0.7, **kwargs):
        super().__init__(**kwargs)
        self.rf_weight = rf_weight
        self.nn_weight = nn_weight
        print(f"🤝 权重融合系统初始化:")
        print(f"   🌲 RF权重: {rf_weight}")
        print(f"   🧠 NN权重: {nn_weight}")
    
    def validate_epoch(self, val_loader):
        """验证一个epoch - 修正输入维度问题"""
        self.model.eval()
        total_loss = 0
        total_note_accuracy = 0
        total_event_accuracy = 0
        num_batches = 0
        
        with torch.no_grad():
            for audio_seq, beatmap_target in val_loader:
                audio_seq = audio_seq.to(self.device)
                beatmap_target = beatmap_target.to(self.device)
                
                # 修正：使用最后一个时间步，与训练保持一致
                note_pred, event_pred = self.model(audio_seq[:, -1, :])
                
                # 计算损失和准确率
                loss, loss_dict = self.compute_loss(note_pred, event_pred, beatmap_target)
                accuracy_dict = self.compute_accuracy(note_pred, event_pred, beatmap_target)
                
                total_loss += loss_dict['total']
                total_note_accuracy += accuracy_dict['note_accuracy']
                total_event_accuracy += accuracy_dict['event_accuracy']
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'note_accuracy': total_note_accuracy / num_batches,
            'event_accuracy': total_event_accuracy / num_batches
        }
    
    def load_preprocessed_data(self):
        """加载预处理数据"""
        print("📂 加载预处理数据...")
        
        # 加载谱面数据
        beatmap_file = "preprocessed_data/all_4k_beatmaps.json"
        if not os.path.exists(beatmap_file):
            print("❌ 未找到预处理谱面数据")
            return None, None
        
        with open(beatmap_file, 'r', encoding='utf-8') as f:
            beatmap_data = json.load(f)
        
        print(f"📊 找到 {len(beatmap_data)} 个预处理谱面")
        
        # 模拟音频特征 (在实际应用中应该是真实的音频特征)
        # 这里我们创建合理的音频特征以进行训练演示
        all_audio_features = []
        all_beatmap_labels = []
        
        for i, beatmap in enumerate(beatmap_data):
            if i >= 100:  # 限制训练样本数量以加快演示
                break
                
            # 生成模拟音频特征
            # 15个特征: [能量, 频谱质心, 零交叉率, 梅尔频率倒谱系数(12个)]
            num_frames = len(beatmap.get('notes', []))
            if num_frames == 0:
                continue
                
            # 确保至少有序列长度的帧数
            if num_frames < self.sequence_length:
                continue
            
            # 音频特征矩阵 [num_frames, 15]
            audio_features = np.random.randn(num_frames, 15) * 0.5 + 0.5
            
            # 谱面标签 [num_frames, 7] (4轨道 + 3事件类型)
            beatmap_labels = np.zeros((num_frames, 7))
            
            # 从谱面数据中提取标签
            notes = beatmap.get('notes', [])
            for j, note in enumerate(notes):
                if j >= num_frames:
                    break
                
                # 音符信息 (4轨道)
                if isinstance(note, dict):
                    track = note.get('track', 0)
                    if 0 <= track <= 3:
                        beatmap_labels[j, track] = 1
                    
                    # 事件类型 (可以根据音符密度等信息推断)
                    if j % 8 == 0:  # 强拍
                        beatmap_labels[j, 4] = 1
                    elif j % 4 == 0:  # 中拍
                        beatmap_labels[j, 5] = 1
                    else:  # 弱拍
                        beatmap_labels[j, 6] = 1
            
            all_audio_features.append(audio_features)
            all_beatmap_labels.append(beatmap_labels)
            
            if i % 20 == 0:
                print(f"   📁 处理进度: {i+1}/{min(100, len(beatmap_data))}")
        
        if not all_audio_features:
            print("❌ 没有有效的训练数据")
            return None, None
        
        # 合并数据
        audio_features = np.vstack(all_audio_features)
        beatmap_labels = np.vstack(all_beatmap_labels)
        
        print(f"✅ 预处理数据加载完成:")
        print(f"   📊 音频特征: {audio_features.shape}")
        print(f"   🎮 谱面标签: {beatmap_labels.shape}")
        
        return audio_features, beatmap_labels
    
    def create_weighted_fusion_model(self, input_dim: int = 15):
        """创建权重融合模型"""
        print("🏗️ 创建权重融合模型...")
        
        self.model = WeightedFusionTransformer(
            input_dim=input_dim,
            d_model=128,          # 较小的模型以加快训练
            num_heads=4,
            num_layers=3,
            dropout=0.1,
            rf_weight=self.rf_weight,
            nn_weight=self.nn_weight,
            learnable_weights=True
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        print(f"📊 模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def train_epoch(self, train_loader):
        """训练一个epoch，增加权重监控"""
        self.model.train()
        total_loss = 0
        total_note_accuracy = 0
        total_event_accuracy = 0
        num_batches = 0
        
        # 权重追踪
        rf_weights = []
        nn_weights = []
        
        for batch_idx, (audio_seq, beatmap_target) in enumerate(train_loader):
            audio_seq = audio_seq.to(self.device)
            beatmap_target = beatmap_target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            note_pred, event_pred = self.model(audio_seq[:, -1, :])  # 使用最后一个时间步
            
            # 计算损失和准确率
            loss, loss_dict = self.compute_loss(note_pred, event_pred, beatmap_target)
            accuracy_dict = self.compute_accuracy(note_pred, event_pred, beatmap_target)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 累计统计
            total_loss += loss_dict['total']
            total_note_accuracy += accuracy_dict['note_accuracy']
            total_event_accuracy += accuracy_dict['event_accuracy']
            num_batches += 1
            
            # 记录权重变化
            if hasattr(self.model, 'get_weights'):
                weights_info = self.model.get_weights()
                rf_weights.append(weights_info['rf_weight'])
                nn_weights.append(weights_info['nn_weight'])
            
            # 每10个批次打印一次权重信息
            if batch_idx % 10 == 0 and len(rf_weights) > 0:
                current_rf = rf_weights[-1]
                current_nn = nn_weights[-1]
                print(f"    批次 {batch_idx}: Loss={loss_dict['total']:.4f}, "
                      f"RF权重={current_rf:.3f}, NN权重={current_nn:.3f}")
        
        # 添加权重信息到历史记录
        if rf_weights:
            self.training_history.setdefault('rf_weights', []).append(np.mean(rf_weights))
            self.training_history.setdefault('nn_weights', []).append(np.mean(nn_weights))
        
        return {
            'loss': total_loss / num_batches,
            'note_accuracy': total_note_accuracy / num_batches,
            'event_accuracy': total_event_accuracy / num_batches,
            'avg_rf_weight': np.mean(rf_weights) if rf_weights else self.rf_weight,
            'avg_nn_weight': np.mean(nn_weights) if nn_weights else self.nn_weight
        }
    
    def analyze_weight_evolution(self):
        """分析权重演化"""
        if 'rf_weights' not in self.training_history:
            return
        
        rf_weights = self.training_history['rf_weights']
        nn_weights = self.training_history['nn_weights']
        
        print(f"\n⚖️ 权重演化分析:")
        print(f"   初始权重: RF={rf_weights[0]:.3f}, NN={nn_weights[0]:.3f}")
        print(f"   最终权重: RF={rf_weights[-1]:.3f}, NN={nn_weights[-1]:.3f}")
        print(f"   RF变化: {rf_weights[-1] - rf_weights[0]:+.3f}")
        print(f"   NN变化: {nn_weights[-1] - nn_weights[0]:+.3f}")

def weighted_fusion_quick_training():
    """权重融合快速训练"""
    print("🎮 权重融合快速训练 (RF:NN = 3:7)")
    print("=" * 50)
    
    # 创建系统实例
    system = WeightedFusionQuickSystem(
        rf_weight=0.3,           # 30%
        nn_weight=0.7,           # 70%
        sequence_length=32,      # 较短序列以加快训练
        batch_size=32,
        learning_rate=0.001
    )
    
    start_time = time.time()
    
    try:
        # 阶段1: 加载预处理数据
        print("\n🔍 阶段1: 加载预处理数据...")
        audio_features, beatmap_labels = system.load_preprocessed_data()
        
        if audio_features is None:
            print("❌ 数据加载失败")
            return None
        
        # 阶段2: 创建权重融合模型
        print("\n🏗️ 阶段2: 创建权重融合模型...")
        system.create_weighted_fusion_model(input_dim=audio_features.shape[1])
        
        # 阶段3: 准备训练数据
        print("\n🔧 阶段3: 准备训练数据...")
        train_loader, val_loader = system.prepare_training_data(
            audio_features, beatmap_labels, train_ratio=0.8
        )
        
        # 阶段4: 权重融合训练
        print("\n🚀 阶段4: 开始权重融合训练...")
        print("📋 训练配置:")
        print(f"   🤝 权重比例: RF={system.rf_weight:.1f} : NN={system.nn_weight:.1f}")
        print(f"   📊 训练批次: {len(train_loader)}")
        print(f"   📊 验证批次: {len(val_loader)}")
        
        # 开始训练
        system.train(
            train_loader, val_loader,
            num_epochs=30,  # 较少的轮数以快速演示
            save_path='quick_weighted_fusion_model_2_8.pth'
        )
        
        # 阶段5: 权重演化分析
        print("\n📊 阶段5: 权重演化分析...")
        system.analyze_weight_evolution()
        
        # 分支性能分析
        if hasattr(system.model, 'get_branch_predictions'):
            print("\n🔬 分支性能分析...")
            system.model.eval()
            sample_batch = next(iter(val_loader))
            with torch.no_grad():
                audio_seq = sample_batch[0].to(system.device)
                branch_results = system.model.get_branch_predictions(audio_seq[:, -1, :])
                
                print(f"   🌲 RF分支权重: {branch_results['rf_weight']:.3f}")
                print(f"   🧠 NN分支权重: {branch_results['nn_weight']:.3f}")
        
        # 绘制训练历史
        system.plot_training_history()
        
        # 保存训练结果
        results_data = {
            'training_config': {
                'rf_weight': system.rf_weight,
                'nn_weight': system.nn_weight,
                'sequence_length': system.sequence_length,
                'batch_size': system.batch_size,
                'learning_rate': system.learning_rate
            },
            'training_history': system.training_history,
            'final_performance': {
                'final_train_loss': system.training_history['train_loss'][-1],
                'final_val_loss': system.training_history['val_loss'][-1],
                'best_val_loss': min(system.training_history['val_loss'])
            }
        }
        
        with open('quick_weighted_fusion_results_2_8.json', 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        # 计算训练时间
        total_time = time.time() - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        
        print("\n🎉 权重融合快速训练完成！")
        print("=" * 50)
        print("📈 最终结果:")
        print(f"   ⏱️ 总训练时间: {minutes:02d}:{seconds:02d}")
        print(f"   🤝 权重比例: RF={system.rf_weight:.1f} : NN={system.nn_weight:.1f}")
        
        if hasattr(system.training_history, 'rf_weights') and system.training_history.get('rf_weights'):
            final_rf = system.training_history['rf_weights'][-1]
            final_nn = system.training_history['nn_weights'][-1]
            print(f"   ⚖️ 学习后权重: RF={final_rf:.3f} : NN={final_nn:.3f}")
        
        print(f"   📊 最终验证损失: {system.training_history['val_loss'][-1]:.4f}")
        print(f"   🎯 最佳验证损失: {min(system.training_history['val_loss']):.4f}")
        
        print("\n💾 保存文件:")
        print("   • quick_weighted_fusion_model_2_8.pth - 权重融合模型")
        print("   • quick_weighted_fusion_results_2_8.json - 训练历史")
        print("   • deep_learning_training_history.png - 训练图表")
        
        return system
        
    except Exception as e:
        print(f"\n❌ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🔧 准备权重融合快速训练环境...")
    
    # 开始权重融合快速训练
    trained_system = weighted_fusion_quick_training()
    
    if trained_system:
        print("\n🎊 权重融合快速训练成功完成！")
        print("🚀 系统现在使用 RF:NN = 2:8 权重比例进行谱面生成！")
        print("📊 通过权重学习优化了老师傅和学生的贡献比例")
    else:
        print("\n💥 训练过程中遇到问题，请检查日志信息")
