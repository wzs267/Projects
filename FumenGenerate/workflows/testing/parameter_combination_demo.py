#!/usr/bin/env python3
"""
混合模型参数组合的具体示例演示
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def demonstrate_parameter_combination():
    """演示混合模型参数组合的具体过程"""
    print("🔄 混合模型参数组合演示")
    print("=" * 50)
    
    # 模拟一批训练数据
    batch_size = 5
    feature_dim = 18
    
    # 假设我们有5个时间步的音频特征
    sample_features = np.array([
        # 特征: [RMS, ZCR, SpectralCentroid, ...其他15个特征]
        [0.8, 0.02, 2500, 1200, 3000, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.15, 0.1, 0.05, 0.6],  # 强音符
        [0.2, 0.01, 1800, 800, 2200, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.08, 0.1, 0.05, 0.6],  # 弱音
        [0.9, 0.03, 3200, 1500, 3800, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.25, 0.1, 0.05, 0.6],    # 很强
        [0.1, 0.005, 1000, 500, 1500, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.03, 0.1, 0.05, 0.6],  # 静音
        [0.6, 0.015, 2200, 1000, 2800, 0.08, 0.16, 0.24, 0.32, 0.4, 0.48, 0.56, 0.64, 0.72, 0.12, 0.1, 0.05, 0.6]   # 中等
    ])
    
    # 对应的标签 (是否有音符)
    sample_labels = np.array([1, 0, 1, 0, 1])  # 强音符处有音符
    
    print(f"📊 输入数据:")
    print(f"   特征矩阵形状: {sample_features.shape}")
    print(f"   标签形状: {sample_labels.shape}")
    
    # 阶段1: 训练随机森林
    print(f"\n🌲 阶段1: 随机森林特征提取")
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sample_features)
    
    # 训练随机森林
    rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
    rf_model.fit(X_scaled, sample_labels)
    
    # 获取特征重要性
    feature_importance = rf_model.feature_importances_
    print(f"   特征重要性: {feature_importance[:5]}... (显示前5个)")
    
    # 选择最重要的15个特征
    top_indices = np.argsort(feature_importance)[-15:]
    print(f"   最重要的15个特征索引: {top_indices}")
    
    # 获取随机森林预测概率
    rf_probabilities = rf_model.predict_proba(X_scaled)[:, 1]
    print(f"   随机森林预测概率: {rf_probabilities}")
    
    # 构建增强特征
    X_top_features = X_scaled[:, top_indices]  # 15维
    rf_enhanced_features = np.column_stack([X_top_features, rf_probabilities])  # 16维
    
    print(f"   原始特征维度: {X_scaled.shape}")
    print(f"   筛选后特征维度: {X_top_features.shape}")
    print(f"   增强特征维度: {rf_enhanced_features.shape}")
    
    # 阶段2: 训练神经网络
    print(f"\n🧠 阶段2: 神经网络融合")
    
    # 定义简化的混合网络
    class SimplifiedHybridNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.audio_branch = nn.Sequential(
                nn.Linear(18, 32),
                nn.ReLU(),
                nn.Linear(32, 16)
            )
            self.rf_branch = nn.Sequential(
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, 8)
            )
            self.fusion = nn.Sequential(
                nn.Linear(16 + 8, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        
        def forward(self, audio_features, rf_features):
            audio_out = self.audio_branch(audio_features)
            rf_out = self.rf_branch(rf_features)
            combined = torch.cat([audio_out, rf_out], dim=1)
            output = self.fusion(combined)
            return output
    
    # 创建模型
    model = SimplifiedHybridNetwork()
    
    # 转换为张量
    audio_tensor = torch.FloatTensor(X_scaled)
    rf_tensor = torch.FloatTensor(rf_enhanced_features)
    labels_tensor = torch.FloatTensor(sample_labels).reshape(-1, 1)
    
    print(f"   音频分支输入: {audio_tensor.shape}")
    print(f"   随机森林分支输入: {rf_tensor.shape}")
    
    # 前向传播演示
    with torch.no_grad():
        # 分支1: 音频特征处理
        audio_out = model.audio_branch(audio_tensor)
        print(f"   音频分支输出: {audio_out.shape} = {audio_out[0][:5].numpy()}")
        
        # 分支2: 随机森林特征处理
        rf_out = model.rf_branch(rf_tensor)
        print(f"   RF分支输出: {rf_out.shape} = {rf_out[0][:5].numpy()}")
        
        # 特征融合
        combined = torch.cat([audio_out, rf_out], dim=1)
        print(f"   融合后特征: {combined.shape}")
        
        # 最终预测
        predictions = model.fusion(combined)
        print(f"   最终预测: {predictions.flatten().numpy()}")
        print(f"   真实标签: {sample_labels}")
    
    # 参数数量统计
    print(f"\n📊 模型参数统计:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   总参数数量: {total_params:,}")
    
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        print(f"   {name}: {module_params:,} 参数")
    
    return model, scaler, rf_model, top_indices

def demonstrate_inference_process():
    """演示推理时的参数组合过程"""
    print(f"\n🚀 推理过程演示")
    print("=" * 30)
    
    # 假设我们有一个新的音频片段
    new_audio_features = np.array([[
        0.7, 0.025, 2800, 1300, 3200, 0.15, 0.25, 0.35, 0.45, 0.55, 
        0.65, 0.75, 0.85, 0.95, 0.18, 0.1, 0.05, 0.6
    ]])
    
    print(f"📥 新音频特征: {new_audio_features.shape}")
    
    # 模拟已训练的组件
    model, scaler, rf_model, top_indices = demonstrate_parameter_combination()
    
    # 推理步骤
    print(f"\n🔄 推理步骤:")
    
    # 1. 特征标准化
    X_scaled = scaler.transform(new_audio_features)
    print(f"   1. 标准化后: {X_scaled[0][:5]}...")
    
    # 2. 随机森林处理
    rf_prob = rf_model.predict_proba(X_scaled)[:, 1]
    X_top = X_scaled[:, top_indices]
    rf_enhanced = np.column_stack([X_top, rf_prob])
    print(f"   2. RF增强: 概率={rf_prob[0]:.3f}, 维度={rf_enhanced.shape}")
    
    # 3. 神经网络预测
    model.eval()
    with torch.no_grad():
        audio_tensor = torch.FloatTensor(X_scaled)
        rf_tensor = torch.FloatTensor(rf_enhanced)
        prediction = model(audio_tensor, rf_tensor)
        print(f"   3. 最终预测: {prediction.item():.3f}")
        
        # 决策
        has_note = prediction.item() > 0.5
        print(f"   4. 决策结果: {'有音符' if has_note else '无音符'}")

def explain_parameter_flow():
    """解释参数流动过程"""
    print(f"\n📋 参数流动解释")
    print("=" * 30)
    
    flow_explanation = """
    数据流动路径:
    
    原始特征 (18维)
        ↓ [标准化]
    标准化特征 (18维)
        ↓ [随机森林分析]
    ├─→ 特征重要性排序 → 选择前15个特征
    └─→ 预测概率 (1维) ─┐
                        ├─→ 增强特征 (16维)
        ↓ [神经网络双分支]    ↑
    ┌─→ 音频分支 (18维输入) → 16维输出
    └─→ RF分支 (16维输入) → 8维输出
        ↓ [特征融合]
    合并特征 (24维)
        ↓ [最终预测层]
    预测结果 (1维)
    
    关键组合点:
    • 随机森林提供"专家意见"概率
    • 神经网络学习更复杂的模式
    • 双分支避免信息损失
    • 融合层优化组合权重
    """
    
    print(flow_explanation)

if __name__ == "__main__":
    # 运行演示
    demonstrate_parameter_combination()
    demonstrate_inference_process()
    explain_parameter_flow()
