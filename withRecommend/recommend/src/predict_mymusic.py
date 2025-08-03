#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于mymusic数据库的音乐推荐预测脚本
"""

import tensorflow as tf
import numpy as np
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def dot_product_func(inputs):
    """点积函数，必须与训练脚本中的定义完全一致"""
    user_vec, song_vec = inputs
    return tf.reduce_sum(user_vec * song_vec, axis=1, keepdims=True)

def load_mymusic_model():
    """加载训练好的mymusic模型"""
    model_path = "models/mymusic_twin_tower.keras"
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'dot_product_func': dot_product_func}
        )
        print(f"模型加载成功: {model_path}")
        return model
    except Exception as e:
        raise Exception(f"模型加载失败: {e}")

def load_mymusic_mappings():
    """加载用户和歌曲映射关系"""
    mappings_path = "data/mymusic_processed/mappings.npy"
    
    if not Path(mappings_path).exists():
        raise FileNotFoundError(f"映射文件不存在: {mappings_path}")
    
    try:
        mappings = np.load(mappings_path, allow_pickle=True).item()
        user_to_idx = mappings['user_to_idx']
        song_to_idx = mappings['song_to_idx']
        
        # 创建反向映射
        idx_to_user = {v: k for k, v in user_to_idx.items()}
        idx_to_song = {v: k for k, v in song_to_idx.items()}
        
        print(f"映射加载成功: {len(user_to_idx)} 用户, {len(song_to_idx)} 歌曲")
        
        return user_to_idx, song_to_idx, idx_to_user, idx_to_song
    except Exception as e:
        raise Exception(f"映射加载失败: {e}")

def predict_for_mymusic_user(user_id: str, top_k: int = 5):
    """
    为mymusic用户生成推荐
    
    Args:
        user_id (str): 用户ID，格式为 "user_3" 或直接传入数字字符串如 "3"
        top_k (int): 推荐歌曲数量
        
    Returns:
        list: 推荐的歌曲ID列表
    """
    print(f"为用户 {user_id} 生成推荐...")
    
    # 加载模型和映射
    model = load_mymusic_model()
    user_to_idx, song_to_idx, idx_to_user, idx_to_song = load_mymusic_mappings()
    
    # 规范化用户ID格式
    if not user_id.startswith('user_'):
        user_id = f"user_{user_id}"
    
    # 检查用户是否存在
    if user_id not in user_to_idx:
        available_users = list(user_to_idx.keys())[:10]  # 显示前10个用户
        raise ValueError(f"用户 {user_id} 不存在。可用用户示例: {available_users}")
    
    # 准备预测数据：目标用户与所有歌曲的组合
    user_idx = user_to_idx[user_id]
    user_array = np.array([user_idx] * len(song_to_idx))
    song_array = np.array(list(song_to_idx.values()))
    
    print(f"预测用户对 {len(song_to_idx)} 首歌曲的偏好...")
    
    # 预测所有歌曲的得分
    predictions = model.predict([user_array, song_array], verbose=0).flatten()
    
    # 获取Top-K推荐
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_scores = predictions[top_indices]
    
    # 转换为歌曲ID
    recommended_songs = []
    for i, idx in enumerate(top_indices):
        song_id = idx_to_song[idx]
        score = top_scores[i]
        recommended_songs.append({
            'song_id': song_id,
            'score': float(score),
            'raw_song_id': int(song_id.replace('song_', ''))  # 提取原始歌曲ID
        })
    
    print(f"推荐完成！为用户 {user_id} 推荐 {len(recommended_songs)} 首歌曲")
    
    return recommended_songs

def batch_predict_mymusic(user_ids: list, top_k: int = 5):
    """
    批量为多个用户生成推荐
    
    Args:
        user_ids (list): 用户ID列表
        top_k (int): 每个用户推荐歌曲数量
        
    Returns:
        dict: 用户ID -> 推荐列表的字典
    """
    print(f"批量为 {len(user_ids)} 个用户生成推荐...")
    
    # 加载模型和映射
    model = load_mymusic_model()
    user_to_idx, song_to_idx, idx_to_user, idx_to_song = load_mymusic_mappings()
    
    results = {}
    
    for user_id in user_ids:
        try:
            # 规范化用户ID
            if not user_id.startswith('user_'):
                user_id = f"user_{user_id}"
            
            if user_id in user_to_idx:
                user_idx = user_to_idx[user_id]
                user_array = np.array([user_idx] * len(song_to_idx))
                song_array = np.array(list(song_to_idx.values()))
                
                # 预测
                predictions = model.predict([user_array, song_array], verbose=0).flatten()
                top_indices = np.argsort(predictions)[-top_k:][::-1]
                top_scores = predictions[top_indices]
                
                # 格式化结果
                recommendations = []
                for i, idx in enumerate(top_indices):
                    song_id = idx_to_song[idx]
                    recommendations.append({
                        'song_id': song_id,
                        'score': float(top_scores[i]),
                        'raw_song_id': int(song_id.replace('song_', ''))
                    })
                
                results[user_id] = recommendations
                
            else:
                print(f"用户 {user_id} 不存在，跳过")
                results[user_id] = []
                
        except Exception as e:
            print(f"❌ 用户 {user_id} 推荐失败: {e}")
            results[user_id] = []
    
    print(f"批量推荐完成！成功为 {len([r for r in results.values() if r])} 个用户生成推荐")
    
    return results

def get_song_info_from_db(song_ids: list):
    """
    从数据库获取歌曲详细信息（可选功能）
    
    Args:
        song_ids (list): 歌曲ID列表
        
    Returns:
        dict: 歌曲ID -> 歌曲信息的字典
    """
    try:
        import mysql.connector
        
        db_config = {
            'host': '127.0.0.1',
            'user': 'root',
            'password': '123456', 
            'database': 'mymusic',
            'charset': 'utf8mb4'
        }
        
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        # 查询歌曲信息
        placeholders = ','.join(['%s'] * len(song_ids))
        query = f"""
        SELECT s.id, s.name, s.introduction, s.pic, si.name as singer_name
        FROM songs s
        LEFT JOIN singers si ON s.singer_id = si.id
        WHERE s.id IN ({placeholders})
        """
        
        cursor.execute(query, song_ids)
        results = cursor.fetchall()
        
        song_info = {}
        for row in results:
            song_id, name, introduction, pic, singer_name = row
            song_info[song_id] = {
                'id': song_id,
                'name': name,
                'introduction': introduction,
                'pic': pic,
                'singer': singer_name
            }
        
        cursor.close()
        conn.close()
        
        return song_info
        
    except Exception as e:
        print(f"无法获取歌曲详细信息: {e}")
        return {}

def main():
    """主函数 - 演示推荐功能"""
    print("mymusic音乐推荐系统")
    print("=" * 50)
    
    try:
        # 示例1: 为单个用户推荐
        print("\n📱 示例1: 为用户推荐音乐")
        user_id = "3"  # 对应数据库中consumers表的ID
        recommendations = predict_for_mymusic_user(user_id, top_k=5)
        
        print(f"\n🎯 用户 {user_id} 的推荐结果:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. 歌曲ID: {rec['raw_song_id']}, 得分: {rec['score']:.4f}")
        
        # 获取歌曲详细信息
        song_ids = [rec['raw_song_id'] for rec in recommendations]
        song_info = get_song_info_from_db(song_ids)
        
        if song_info:
            print(f"\n推荐歌曲详情:")
            for i, rec in enumerate(recommendations, 1):
                song_id = rec['raw_song_id']
                if song_id in song_info:
                    info = song_info[song_id]
                    print(f"   {i}. {info['name']} - {info['singer']} (得分: {rec['score']:.4f})")
        
        # 示例2: 批量推荐
        print(f"\n📱 示例2: 批量推荐")
        user_ids = ["3", "4", "5"]
        batch_results = batch_predict_mymusic(user_ids, top_k=3)
        
        for user_id, recs in batch_results.items():
            if recs:
                print(f"\n   用户 {user_id}: {[rec['raw_song_id'] for rec in recs]}")
        
    except Exception as e:
        print(f"❌ 推荐失败: {e}")
        print(f"   请确保已运行训练脚本生成模型文件")

if __name__ == "__main__":
    main()
