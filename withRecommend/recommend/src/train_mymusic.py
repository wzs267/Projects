#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于mymusic数据库的音乐推荐模型训练脚本
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Flatten, Lambda
from tensorflow.keras.models import Model
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.utils.mymusic_data_loader import load_mymusic_interactions, generate_mymusic_training_data
except ImportError:
    print("⚠️  无法导入mymusic_data_loader，将使用本地数据加载方法")

class MyMusicConfig:
    """mymusic数据库配置"""
    # 模型参数
    EMBEDDING_DIM = 64
    LSTM_UNITS = 128
    DENSE_UNITS = 64
    USER_TOWER_OUTPUT_DIM = 64
    SONG_TOWER_OUTPUT_DIM = 64
    
    # 训练参数
    EPOCHS = 10
    BATCH_SIZE = 32
    
    # 路径配置
    MODEL_SAVE_PATH = "models/mymusic_twin_tower.keras"

def dot_product_func(inputs):
    """点积函数，必须与predict脚本中的定义完全一致"""
    user_vec, song_vec = inputs
    return tf.reduce_sum(user_vec * song_vec, axis=1, keepdims=True)

def build_mymusic_twin_tower_model(num_users: int, num_songs: int):
    """
    构建适配mymusic数据的双塔模型
    
    Args:
        num_users: 用户数量
        num_songs: 歌曲数量
        
    Returns:
        tensorflow.keras.Model: 编译后的模型
    """
    config = MyMusicConfig()
    
    # 用户塔
    user_input = Input(shape=(1,), name='user_input')
    user_embed = Embedding(num_users, config.EMBEDDING_DIM, name='user_embedding')(user_input)
    user_lstm = LSTM(config.LSTM_UNITS, name='user_lstm')(user_embed)
    user_vec = Dense(config.USER_TOWER_OUTPUT_DIM, activation='relu', name='user_dense')(user_lstm)
    
    # 歌曲塔
    song_input = Input(shape=(1,), name='song_input')
    song_embed = Embedding(num_songs, config.EMBEDDING_DIM, name='song_embedding')(song_input)
    song_vec = Dense(config.SONG_TOWER_OUTPUT_DIM, activation='relu', name='song_dense')(song_embed)
    song_vec = Flatten(name='song_flatten')(song_vec)
    
    # 交互层：使用点积计算相似度
    dot_product = Lambda(
        dot_product_func,
        name="dot_product",
        output_shape=(1,)
    )([user_vec, song_vec])
    
    # 构建并编译模型
    model = Model(inputs=[user_input, song_input], outputs=dot_product, name='mymusic_twin_tower')
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def load_mymusic_data_fallback():
    """
    备用数据加载方法（当无法使用数据库时）
    """
    import json
    import numpy as np
    
    # 尝试从JSON文件加载
    json_file = Path("data/mymusic_generated/user_plays.json")
    if not json_file.exists():
        raise FileNotFoundError(f"找不到数据文件: {json_file}")
    
    print("📁 从JSON文件加载数据...")
    with open(json_file, 'r', encoding='utf-8') as f:
        plays = json.load(f)
    
    # 处理数据
    interactions = []
    user_ids = set()
    song_ids = set()
    
    for play in plays:
        user_id = f"user_{play['user_id']}"
        song_id = f"song_{play['song_id']}"
        
        completion_rate = play['play_duration'] / play['full_duration'] if play['full_duration'] > 0 else 0
        label = 1 if (completion_rate > 0.6 or play.get('is_liked', 0) == 1) else 0
        
        interactions.append({
            'user_id': user_id,
            'song_id': song_id,
            'label': label
        })
        
        user_ids.add(user_id)
        song_ids.add(song_id)
    
    # 创建映射
    user_to_idx = {user_id: idx for idx, user_id in enumerate(sorted(user_ids))}
    song_to_idx = {song_id: idx for idx, song_id in enumerate(sorted(song_ids))}
    
    # 生成训练数据
    positive_samples = [(user_to_idx[item['user_id']], song_to_idx[item['song_id']], item['label']) 
                       for item in interactions]
    
    # 生成负样本
    positive_pairs = set((item[0], item[1]) for item in positive_samples)
    negative_samples = []
    
    while len(negative_samples) < len(positive_samples):
        user_idx = np.random.randint(0, len(user_to_idx))
        song_idx = np.random.randint(0, len(song_to_idx))
        
        if (user_idx, song_idx) not in positive_pairs:
            negative_samples.append((user_idx, song_idx, 0))
    
    # 合并并打乱
    all_samples = positive_samples + negative_samples
    np.random.shuffle(all_samples)
    
    users = np.array([sample[0] for sample in all_samples])
    songs = np.array([sample[1] for sample in all_samples])
    labels = np.array([sample[2] for sample in all_samples])
    
    print(f"✅ 数据加载完成: {len(user_to_idx)} 用户, {len(song_to_idx)} 歌曲, {len(all_samples)} 样本")
    
    return interactions, user_to_idx, song_to_idx, users, songs, labels

def train_mymusic_model():
    """训练mymusic推荐模型"""
    print("🎵 开始训练mymusic音乐推荐模型...")
    print("=" * 60)
    
    try:
        # 尝试使用数据库加载器
        interactions, user_to_idx, song_to_idx = load_mymusic_interactions()
        users, songs, labels = generate_mymusic_training_data(interactions, user_to_idx, song_to_idx)
    except:
        # 使用备用加载方法
        print("⚠️  使用备用数据加载方法...")
        interactions, user_to_idx, song_to_idx, users, songs, labels = load_mymusic_data_fallback()
    
    print(f"\n📊 数据统计:")
    print(f"   用户数量: {len(user_to_idx)}")
    print(f"   歌曲数量: {len(song_to_idx)}")
    print(f"   训练样本: {len(users)}")
    print(f"   正样本比例: {labels.mean():.2%}")
    
    # 构建模型
    print(f"\n🏗️  构建双塔模型...")
    model = build_mymusic_twin_tower_model(len(user_to_idx), len(song_to_idx))
    
    # 显示模型结构
    model.summary()
    
    # 训练模型
    print(f"\n🧠 开始训练模型...")
    config = MyMusicConfig()
    
    history = model.fit(
        [users, songs], 
        labels,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_split=0.2,
        verbose=1
    )
    
    # 保存模型
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    model.save(config.MODEL_SAVE_PATH)
    print(f"✅ 模型已保存到: {config.MODEL_SAVE_PATH}")
    
    # 保存映射关系（用于预测）
    mappings_file = Path("data/mymusic_processed/mappings.npy")
    mappings_file.parent.mkdir(exist_ok=True)
    
    import numpy as np
    mappings = {
        'user_to_idx': user_to_idx,
        'song_to_idx': song_to_idx
    }
    np.save(mappings_file, mappings)
    print(f"✅ 映射关系已保存到: {mappings_file}")
    
    # 显示训练结果
    final_loss = history.history['loss'][-1]
    final_accuracy = history.history['accuracy'][-1]
    val_loss = history.history['val_loss'][-1]
    val_accuracy = history.history['val_accuracy'][-1]
    
    print(f"\n📈 训练结果:")
    print(f"   训练损失: {final_loss:.4f}")
    print(f"   训练准确率: {final_accuracy:.4f}")
    print(f"   验证损失: {val_loss:.4f}")
    print(f"   验证准确率: {val_accuracy:.4f}")
    
    print(f"\n🎉 mymusic模型训练完成！")

if __name__ == "__main__":
    train_mymusic_model()
