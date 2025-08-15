#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度学习系统测试脚本
首先用少量数据验证系统可行性
"""

import os
import numpy as np
import torch
from scripts.deep_learning_beatmap_system import DeepBeatmapLearningSystem

def test_deep_learning_system():
    """测试深度学习系统基本功能"""
    print("🧪 测试深度学习音游谱面生成系统")
    print("=" * 50)
    
    # 检查PyTorch安装
    print(f"🔥 PyTorch版本: {torch.__version__}")
    print(f"🖥️  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU设备: {torch.cuda.get_device_name()}")
    
    # 创建系统实例
    system = DeepBeatmapLearningSystem(
        sequence_length=32,  # 减小序列长度以便快速测试
        batch_size=16,       # 减小批次大小
        learning_rate=0.001
    )
    
    # 生成模拟数据用于测试
    print("\n🎲 生成模拟数据进行架构测试...")
    
    # 模拟音频特征：[时间步, 15维特征]
    num_timesteps = 1000
    feature_dim = 15
    audio_features = np.random.randn(num_timesteps, feature_dim).astype(np.float32)
    
    # 模拟谱面标签：[时间步, 7维] (4轨道 + 3事件类型)
    beatmap_labels = np.zeros((num_timesteps, 7), dtype=np.float32)
    
    # 随机设置一些音符事件（模拟真实数据模式）
    for i in range(0, num_timesteps, 10):  # 每10个时间步放一个音符
        if np.random.random() > 0.5:  # 50%概率放置音符
            # 随机选择轨道
            column = np.random.randint(0, 4)
            beatmap_labels[i, column] = 1.0
            
            # 随机选择事件类型
            event_type = np.random.choice([0, 1, 2], p=[0.8, 0.15, 0.05])  # note, long_start, long_end
            beatmap_labels[i, 4 + event_type] = 1.0
    
    print(f"📊 模拟数据大小: {audio_features.shape}")
    print(f"🎵 音符密度: {np.sum(beatmap_labels[:, :4]) / num_timesteps:.3f}")
    
    # 创建模型
    print("\n🏗️  创建深度学习模型...")
    system.create_model(input_dim=feature_dim)
    
    # 准备训练数据
    print("🔧 准备训练数据...")
    train_loader, val_loader = system.prepare_training_data(
        audio_features, beatmap_labels, train_ratio=0.8
    )
    
    # 进行少量轮次的训练测试
    print("\n🚀 开始短期训练测试 (5轮)...")
    system.train(
        train_loader, val_loader,
        num_epochs=5,  # 只训练5轮进行测试
        save_path='test_model.pth'
    )
    
    # 绘制训练历史
    system.plot_training_history()
    
    print("\n✅ 深度学习系统测试完成！")
    print("🔍 关键测试结果:")
    print(f"   • 模型成功创建并训练")
    print(f"   • 参数数量: {sum(p.numel() for p in system.model.parameters()):,}")
    print(f"   • 最终验证损失: {system.training_history['val_loss'][-1]:.4f}")
    print(f"   • 最终音符准确率: {system.training_history['note_accuracy'][-1]:.3f}")
    print(f"   • 最终事件准确率: {system.training_history['event_accuracy'][-1]:.3f}")
    
    return system

if __name__ == "__main__":
    test_system = test_deep_learning_system()
