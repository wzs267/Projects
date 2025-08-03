import numpy as np
from typing import Tuple

def load_interactions() -> Tuple[np.ndarray, dict, dict]:
    """加载处理后的数据"""
    interactions = np.load("data/processed/interactions.npy", allow_pickle=True)
    with open("data/processed/mappings.npy", 'rb') as f:
        user_to_idx = np.load(f, allow_pickle=True).item()
        song_to_idx = np.load(f, allow_pickle=True).item()
    return interactions, user_to_idx, song_to_idx

def generate_training_data(interactions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成训练数据（正负样本）"""
    positive_pairs = []
    negative_pairs = []
    
    # 正样本：交互值>0的用户-歌曲对
    user_indices, song_indices = np.where(interactions > 0)
    for u, s in zip(user_indices, song_indices):
        positive_pairs.append((u, s, 1))  # 标签1表示正样本
    
    # 负样本：随机采样未交互的对
    num_negatives = len(positive_pairs)
    for _ in range(num_negatives):
        u = np.random.randint(0, interactions.shape[0])
        s = np.random.randint(0, interactions.shape[1])
        if interactions[u, s] == 0:
            negative_pairs.append((u, s, 0))  # 标签0表示负样本
    
    all_pairs = positive_pairs + negative_pairs
    np.random.shuffle(all_pairs)
    
    # 转换为numpy数组
    users = np.array([x[0] for x in all_pairs])
    songs = np.array([x[1] for x in all_pairs])
    labels = np.array([x[2] for x in all_pairs])
    
    return users, songs, labels