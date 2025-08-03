# utils/preprocessing.py
import numpy as np
import json
import random
import os
from src.config import Config

def generate_mock_data():
    """生成模拟数据并确保目录存在"""
    os.makedirs(os.path.dirname(Config.DATA_PATH), exist_ok=True)
    
    data = []
    for user_id in range(Config.NUM_USERS):
        plays = random.randint(5, 50)
        for _ in range(plays):
            data.append({
                "user_id": f"user_{user_id}",
                "song_id": f"song_{random.randint(0, Config.NUM_SONGS-1)}",
                "duration": random.randint(10, 300)
            })
    
    with open(Config.DATA_PATH, 'w') as f:
        json.dump(data, f)

def preprocess_data():
    """处理数据并强制保存到文件"""
    os.makedirs(os.path.dirname(Config.PROCESSED_PATH), exist_ok=True)
    
    with open(Config.DATA_PATH) as f:
        data = json.load(f)
    
    # 生成映射和交互矩阵
    user_ids = sorted({d["user_id"] for d in data})
    song_ids = sorted({d["song_id"] for d in data})
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    song_to_idx = {s: i for i, s in enumerate(song_ids)}
    
    interactions = np.zeros((len(user_ids), len(song_ids)))
    for d in data:
        # 去掉时长限制，增加模型探索倾向
        interactions[user_to_idx[d["user_id"]], song_to_idx[d["song_id"]]] += 1
    
    # 归一化并保存
    interactions /= np.max(interactions)
    np.save(Config.PROCESSED_PATH, interactions)
    
    # 保存映射字典
    with open(Config.MAPPINGS_PATH, 'wb') as f:
        np.save(f, user_to_idx)
        np.save(f, song_to_idx)
    
    return interactions, user_to_idx, song_to_idx