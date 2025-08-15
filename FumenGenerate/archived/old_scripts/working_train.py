#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工作目录修复后的快速训练脚本
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 添加路径以便导入
sys.path.append('.')
sys.path.append('core')
sys.path.append('scripts')

def load_existing_data():
    """加载现有的训练数据"""
    try:
        from models.beatmap_learning_system import BeatmapLearningSystem
        
        print("📥 加载现有训练数据...")
        system = BeatmapLearningSystem()
        aligned_datasets = system.collect_training_data('test_4k_beatmaps.json', 'extracted_audio')
        
        if not aligned_datasets:
            print("❌ 无法加载数据")
            return None, None
        
        X, y_note, y_column, y_long = system.prepare_machine_learning_data(aligned_datasets)
        return X, y_note
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None, None

class QuickHybridNet(nn.Module):
    """快速混合网络"""
    
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x).squeeze()

def quick_train():
    """快速训练"""
    print("🚀 快速混合训练")
    print("=" * 40)
    
    # 加载数据
    X, y = load_existing_data()
    if X is None:
        print("❌ 无法获取训练数据")
        return
    
    print(f"✅ 数据加载完成: {len(X):,} 样本")
    
    # 训练RF
    print("🌲 训练随机森林...")
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X, y)
    rf_probs = rf.predict_proba(X)[:, 1]
    
    # 构建增强特征
    enhanced_X = np.column_stack([X, rf_probs, np.gradient(rf_probs)])
    
    # 标准化
    scaler = StandardScaler()
    enhanced_X = scaler.fit_transform(enhanced_X)
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        enhanced_X, y, test_size=0.2, random_state=42
    )
    
    # 训练神经网络
    print("🧠 训练神经网络...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = QuickHybridNet(enhanced_X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # 转换数据
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # 训练
    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                test_pred = (test_outputs > 0.5).float()
                accuracy = (test_pred == y_test_tensor).float().mean()
                print(f"Epoch {epoch}: 损失={loss:.4f}, 准确率={accuracy:.3f}")
    
    # 最终测试
    model.eval()
    with torch.no_grad():
        final_outputs = model(X_test_tensor)
        final_pred = (final_outputs > 0.5).float()
        final_accuracy = (final_pred == y_test_tensor).float().mean()
        
        rf_acc = rf.score(X_test[:, :X.shape[1]], y_test)
    
    print(f"\n🎉 训练完成！")
    print(f"🌲 随机森林准确率: {rf_acc:.3f}")
    print(f"🧠 混合模型准确率: {final_accuracy:.3f}")
    print(f"📈 提升: {final_accuracy - rf_acc:.3f}")
    
    # 保存模型
    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': final_accuracy.item()
    }, 'models/working_hybrid_model.pth')
    
    import pickle
    with open('models/working_components.pkl', 'wb') as f:
        pickle.dump({'rf': rf, 'scaler': scaler}, f)
    
    print("💾 模型已保存到 models/")

if __name__ == "__main__":
    quick_train()
