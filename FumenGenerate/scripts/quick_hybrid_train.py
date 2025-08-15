#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实际训练混合系统
使用真实数据快速训练和测试
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def quick_train_hybrid():
    """快速训练混合系统"""
    print("🚀 快速训练混合学习系统")
    print("=" * 50)
    
    # 加载现有训练数据
    try:
        from scripts.beatmap_learning_system import BeatmapLearningSystem
        
        print("📥 加载训练数据...")
        system = BeatmapLearningSystem()
        aligned_datasets = system.collect_training_data('test_4k_beatmaps.json', 'extracted_audio')
        
        if not aligned_datasets:
            print("❌ 无法加载数据")
            return
        
        X, y_note, y_column, y_long = system.prepare_machine_learning_data(aligned_datasets)
        print(f"✅ 数据加载完成: {len(X):,} 个样本")
        
        # 1. 训练随机森林获取特征重要性
        print("\n🌲 训练随机森林特征提取器...")
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_model.fit(X, y_note)
        
        # 获取特征重要性排序
        feature_importance = rf_model.feature_importances_
        rf_probs = rf_model.predict_proba(X)[:, 1]  # 音符概率
        
        print(f"🎯 RF准确率: {rf_model.score(X, y_note):.3f}")
        
        # 2. 构建增强特征
        enhanced_features = np.column_stack([
            X,  # 原始音频特征
            rf_probs,  # RF音符概率
            feature_importance.reshape(1, -1).repeat(len(X), axis=0),  # 特征重要性权重
            np.gradient(rf_probs),  # RF概率梯度
        ])
        
        print(f"📊 增强特征维度: {X.shape[1]} → {enhanced_features.shape[1]}")
        
        # 3. 简化神经网络
        class SimpleHybridNet(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.network(x).squeeze()
        
        # 4. 训练混合模型
        print("\n🧠 训练混合神经网络...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"设备: {device}")
        
        # 数据准备
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(enhanced_features)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_note, test_size=0.2, random_state=42
        )
        
        # 转换为Tensor
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)
        
        # 创建模型
        model = SimpleHybridNet(enhanced_features.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        print(f"📊 模型参数: {sum(p.numel() for p in model.parameters()):,}")
        
        # 训练
        model.train()
        for epoch in range(20):  # 快速训练20轮
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0:
                # 验证
                model.eval()
                with torch.no_grad():
                    test_outputs = model(X_test_tensor)
                    test_loss = criterion(test_outputs, y_test_tensor)
                    
                    # 计算准确率
                    test_pred = (test_outputs > 0.5).float()
                    accuracy = (test_pred == y_test_tensor).float().mean()
                    
                print(f"Epoch {epoch:2d}: 训练损失={loss:.4f}, 测试损失={test_loss:.4f}, 准确率={accuracy:.3f}")
                model.train()
        
        # 最终测试
        model.eval()
        with torch.no_grad():
            final_outputs = model(X_test_tensor)
            final_pred = (final_outputs > 0.5).float()
            final_accuracy = (final_pred == y_test_tensor).float().mean()
            
            # 与RF对比（使用原始特征）
            rf_test_probs = rf_model.predict_proba(X_test[:, :X.shape[1]])[:, 1]  # 只用原始特征
            rf_test_pred = (rf_test_probs > 0.5).astype(float)
            rf_accuracy = (rf_test_pred == y_test).mean()
        
        print(f"\n🎉 训练完成！")
        print(f"🌲 随机森林准确率: {rf_accuracy:.3f}")
        print(f"🧠 混合神经网络准确率: {final_accuracy:.3f}")
        print(f"📈 提升幅度: {(final_accuracy - rf_accuracy):.3f}")
        
        # 保存模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'accuracy': final_accuracy.item()
        }, 'quick_hybrid_model.pth')
        
        # 单独保存其他组件
        import pickle
        with open('hybrid_components.pkl', 'wb') as f:
            pickle.dump({
                'scaler': scaler,
                'rf_model': rf_model
            }, f)
        
        print("💾 模型已保存: quick_hybrid_model.pth")
        
        # 特征重要性分析
        print(f"\n🔍 特征重要性Top5:")
        feature_names = [f"音频特征{i}" for i in range(X.shape[1])] + ['RF概率', 'RF重要性权重', 'RF概率梯度']
        importance_idx = np.argsort(feature_importance)[::-1][:5]
        for i, idx in enumerate(importance_idx):
            if idx < len(feature_names):
                print(f"   {i+1}. {feature_names[idx]}: {feature_importance[idx]:.4f}")
        
        return model, scaler, rf_model
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def demo_generation():
    """演示生成效果"""
    if not os.path.exists('quick_hybrid_model.pth'):
        print("❌ 未找到训练好的模型")
        return
    
    print("\n🎮 演示混合系统生成效果")
    
    # 加载模型
    checkpoint = torch.load('quick_hybrid_model.pth', map_location='cpu', weights_only=False)
    
    # 加载其他组件
    import pickle
    with open('hybrid_components.pkl', 'rb') as f:
        components = pickle.load(f)
    
    scaler = components['scaler']
    rf_model = components['rf_model']
    
    print(f"📥 加载模型 (准确率: {checkpoint['accuracy']:.3f})")
    
    # 模拟音频特征进行测试
    print("🎵 模拟音频特征测试...")
    
    # 生成一些测试特征
    test_features = np.random.randn(100, 15)  # 模拟100个时间步的音频特征
    
    # 使用RF模型获取概率
    rf_probs = rf_model.predict_proba(test_features)[:, 1]
    
    # 构建增强特征
    feature_importance = rf_model.feature_importances_
    enhanced_test = np.column_stack([
        test_features,
        rf_probs,
        feature_importance.reshape(1, -1).repeat(len(test_features), axis=0),
        np.gradient(rf_probs),
    ])
    
    # 标准化
    test_scaled = scaler.transform(enhanced_test)
    
    # 创建模型并加载权重
    class SimpleHybridNet(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.network(x).squeeze()
    
    model = SimpleHybridNet(enhanced_test.shape[1])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        predictions = model(torch.FloatTensor(test_scaled))
        note_predictions = (predictions > 0.5).float().numpy()
    
    print(f"🎯 预测结果: {np.sum(note_predictions):.0f}/100 个位置放置音符")
    print(f"📊 音符密度: {np.sum(note_predictions)/100:.2f}")
    
    # 显示预测概率分布
    prob_high = np.sum(predictions.numpy() > 0.7)
    prob_medium = np.sum((predictions.numpy() > 0.3) & (predictions.numpy() <= 0.7))
    prob_low = np.sum(predictions.numpy() <= 0.3)
    
    print(f"🎼 置信度分布:")
    print(f"   高置信(>0.7): {prob_high} 个")
    print(f"   中置信(0.3-0.7): {prob_medium} 个") 
    print(f"   低置信(<0.3): {prob_low} 个")

if __name__ == "__main__":
    # 训练混合系统
    result = quick_train_hybrid()
    
    if result:
        print("\n" + "="*50)
        # 演示生成效果
        demo_generation()
    else:
        print("❌ 训练失败，无法演示")
