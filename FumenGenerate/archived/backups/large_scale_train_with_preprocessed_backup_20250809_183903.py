#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于预处理数据的大规模训练脚本
使用与小批量训练相同的成功模式
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_preprocessed_data():
    """加载预处理的训练数据"""
    try:
        from models.beatmap_learning_system import BeatmapLearningSystem
        
        system = BeatmapLearningSystem()
        system.traindata_dir = "preprocessed_data"
        
        # 加载JSON数据
        json_file = "preprocessed_data/all_4k_beatmaps.json"
        
        if not os.path.exists(json_file):
            print(f"❌ 找不到预处理数据文件: {json_file}")
            return None, None
        
        with open(json_file, 'r', encoding='utf-8') as f:
            beatmaps_data = json.load(f)
        
        print(f"📁 加载了 {len(beatmaps_data)} 个预处理谱面")
        
        # 使用系统的处理方法（与小批量训练相同）
        collected_data = []
        
        # 使用进度条处理所有谱面
        with tqdm(beatmaps_data, desc="🎵 处理谱面", unit="谱面") as pbar:
            for i, beatmap_data in enumerate(pbar):
                # 更新进度条描述
                title_short = beatmap_data['title'][:15] + "..." if len(beatmap_data['title']) > 15 else beatmap_data['title']
                pbar.set_postfix({
                    "当前": title_short,
                    "成功": len(collected_data)
                })
                
                # 构建音频文件路径
                audio_file = os.path.join("preprocessed_data/audio", beatmap_data['audio_file'])
                
                if not os.path.exists(audio_file):
                    continue
                
                # 字段映射：将预处理数据的字段名映射到BeatmapLearningSystem期望的字段名
                mapped_beatmap_data = {
                    'song_title': beatmap_data.get('title', 'Unknown'),
                    'difficulty_version': beatmap_data.get('difficulty_name', 'Unknown'),
                    'notes': beatmap_data.get('notes', []),
                    'timing_points': beatmap_data.get('timing_points', []),
                    'note_count': beatmap_data.get('note_count', 0),
                    'note_density': beatmap_data.get('note_density', 0),
                    'long_notes_ratio': beatmap_data.get('long_notes_ratio', 0),
                    'avg_bpm': beatmap_data.get('avg_bpm', 120),
                    'duration': beatmap_data.get('duration', 0),
                    'initial_bpm': beatmap_data.get('initial_bpm', 120)
                }
                
                # 使用系统的处理方法
                mcz_name = beatmap_data['source_mcz']
                data = system.process_single_beatmap(mcz_name, mapped_beatmap_data, audio_file)
                
                if data:
                    collected_data.append(data)
            else:
                print(f"⚠️ 处理失败: {beatmap_data['title']}")
        
        print(f"\n收集到 {len(collected_data)} 个有效的训练样本")
        
        if not collected_data:
            return None, None
        
        # 合并所有特征和标签
        all_features = []
        all_labels = []
        
        for data in collected_data:
            # AlignedData对象，使用属性访问而不是字典访问
            features = data.audio_features  # 音频特征矩阵
            events = data.beatmap_events     # 谱面事件矩阵
            
            # 构建标签：将多列的事件矩阵转换为单列的note标签
            # 假设events矩阵的每一行代表一个时间步，每一列代表一个键道
            # 标签为1表示该时间步有音符，0表示没有
            labels = np.any(events > 0, axis=1).astype(int)
            
            all_features.append(features)
            all_labels.append(labels)
        
        # 拼接所有数据
        combined_features = np.vstack(all_features)
        combined_labels = np.hstack(all_labels)
        
        print(f"✅ 数据合并完成: {combined_features.shape[0]:,} 样本")
        return combined_features, combined_labels
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# 混合模型定义（与working_train.py相同）
class HybridNeuralNetwork(nn.Module):
    """混合神经网络：结合随机森林特征和音频特征"""
    
    def __init__(self, audio_features_dim=15, rf_features_dim=15):
        super(HybridNeuralNetwork, self).__init__()
        
        # 音频特征分支
        self.audio_branch = nn.Sequential(
            nn.Linear(audio_features_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 随机森林特征分支
        self.rf_branch = nn.Sequential(
            nn.Linear(rf_features_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(32 + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, audio_features, rf_features):
        audio_out = self.audio_branch(audio_features)
        rf_out = self.rf_branch(rf_features)
        
        combined = torch.cat([audio_out, rf_out], dim=1)
        output = self.fusion(combined)
        
        return output

def train_large_scale_model(features, labels):
    """训练大规模混合模型"""
    print(f"\n🚀 开始大规模混合模型训练...")
    print(f"   📊 训练样本: {len(features):,}")
    print(f"   🎯 正样本比例: {np.mean(labels):.3f}")
    
    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 阶段1: 训练随机森林
    print(f"\n🌲 阶段1: 训练随机森林...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train_scaled, y_train)
    rf_accuracy = rf_model.score(X_test_scaled, y_test)
    print(f"   🌲 随机森林准确率: {rf_accuracy:.3f}")
    
    # 获取随机森林预测概率作为特征
    rf_train_probs = rf_model.predict_proba(X_train_scaled)[:, 1].reshape(-1, 1)
    rf_test_probs = rf_model.predict_proba(X_test_scaled)[:, 1].reshape(-1, 1)
    
    # 特征重要性前15个特征
    feature_importance = rf_model.feature_importances_
    top_indices = np.argsort(feature_importance)[-15:]
    
    X_train_top = X_train_scaled[:, top_indices]
    X_test_top = X_test_scaled[:, top_indices]
    
    # 构建增强特征
    rf_enhanced_train = np.hstack([
        X_train_top,
        rf_train_probs
    ])
    rf_enhanced_test = np.hstack([
        X_test_top,
        rf_test_probs
    ])
    
    # 阶段2: 训练混合神经网络
    print(f"\n🧠 阶段2: 训练混合神经网络...")
    
    # 转换为PyTorch张量
    X_audio_train = torch.FloatTensor(X_train_scaled)
    X_rf_train = torch.FloatTensor(rf_enhanced_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    
    X_audio_test = torch.FloatTensor(X_test_scaled)
    X_rf_test = torch.FloatTensor(rf_enhanced_test)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)
    
    # 创建模型
    model = HybridNeuralNetwork(
        audio_features_dim=X_train_scaled.shape[1],
        rf_features_dim=rf_enhanced_train.shape[1]
    )
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    epochs = 20
    batch_size = 512
    
    # 使用进度条显示训练过程
    with tqdm(range(epochs), desc="🧠 训练神经网络", unit="epoch") as epoch_bar:
        for epoch in epoch_bar:
            model.train()
            total_loss = 0
            
            # 批量训练
            num_batches = (len(X_audio_train) + batch_size - 1) // batch_size
            for i in range(0, len(X_audio_train), batch_size):
                batch_audio = X_audio_train[i:i+batch_size]
                batch_rf = X_rf_train[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_audio, batch_rf)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # 计算验证准确率
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_audio_test, X_rf_test)
                test_predictions = (test_outputs > 0.5).float()
                accuracy = (test_predictions == y_test_tensor).float().mean().item()
            
            # 更新进度条
            epoch_bar.set_postfix({
                "损失": f"{total_loss/num_batches:.4f}",
                "准确率": f"{accuracy:.3f}"
            })
    
    # 最终评估
    model.eval()
    with torch.no_grad():
        final_outputs = model(X_audio_test, X_rf_test)
        final_predictions = (final_outputs > 0.5).float()
        final_accuracy = (final_predictions == y_test_tensor).float().mean().item()
    
    print(f"\n🎉 训练完成！")
    print(f"🌲 随机森林准确率: {rf_accuracy:.3f}")
    print(f"🧠 混合模型准确率: {final_accuracy:.3f}")
    print(f"📈 提升: {(final_accuracy - rf_accuracy):.3f}")
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'rf_model': rf_model,
        'top_indices': top_indices
    }, 'models/large_scale_hybrid_model.pth')
    
    print(f"💾 模型已保存到 models/large_scale_hybrid_model.pth")
    
    return final_accuracy

def main():
    """主函数"""
    print("🎮 大规模预处理数据训练")
    print("=" * 50)
    print("📝 使用与小批量训练相同的成功模式")
    print()
    
    # 加载预处理数据
    print("📥 加载预处理训练数据...")
    features, labels = load_preprocessed_data()
    
    if features is None or labels is None:
        print("❌ 无法获取训练数据")
        return
    
    # 训练模型
    accuracy = train_large_scale_model(features, labels)
    
    if accuracy > 0.90:
        print(f"\n🏆 恭喜！达到90%以上准确率！")
    elif accuracy > 0.85:
        print(f"\n🎯 很好！达到85%以上准确率！")
    else:
        print(f"\n💡 准确率为{accuracy:.1%}，可以进一步优化")

if __name__ == "__main__":
    main()
