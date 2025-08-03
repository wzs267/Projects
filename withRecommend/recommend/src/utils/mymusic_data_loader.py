#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mymusic数据库数据加载器
适配实际的mymusic数据库结构，加载用户播放记录数据
"""

import numpy as np
import json
import pymysql
from pathlib import Path
import sys
import os

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class MyMusicDataLoader:
    """mymusic数据库数据加载器"""
    
    def __init__(self, db_config=None):
        """
        初始化数据加载器
        
        Args:
            db_config (dict): 数据库配置
        """
        self.db_config = db_config or {
            'host': '127.0.0.1',
            'user': 'root', 
            'password': '123456',
            'database': 'mymusic',
            'charset': 'utf8mb4'
        }
        
    def load_from_database(self):
        """从数据库加载播放记录数据"""
        try:
            # 连接数据库
            conn = pymysql.connect(**self.db_config)
            cursor = conn.cursor()
            
            # 查询播放记录数据
            query = """
            SELECT 
                up.user_id,
                up.song_id, 
                up.play_duration,
                up.full_duration,
                up.is_liked,
                up.play_time
            FROM user_plays up
            ORDER BY up.play_time
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            # 转换为标准格式
            interactions = []
            for row in results:
                user_id, song_id, play_duration, full_duration, is_liked, play_time = row
                
                # 计算播放完成度
                completion_rate = play_duration / full_duration if full_duration > 0 else 0
                
                # 根据播放完成度和is_liked字段判断是否为正样本
                # 播放超过60%或明确标记为喜欢的为正样本
                label = 1 if (completion_rate > 0.6 or is_liked == 1) else 0
                
                interactions.append({
                    'user_id': f"user_{user_id}",
                    'song_id': f"song_{song_id}",
                    'rating': completion_rate,  # 使用播放完成度作为评分
                    'label': label,
                    'play_time': str(play_time)
                })
            
            cursor.close()
            conn.close()
            
            print(f"✅ 从数据库加载了 {len(interactions)} 条播放记录")
            return interactions
            
        except pymysql.Error as e:
            print(f"❌ 数据库连接错误: {e}")
            return []
        except Exception as e:
            print(f"❌ 数据加载错误: {e}")
            return []
    
    def load_from_json(self, json_file):
        """从JSON文件加载数据（备用方案）"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                plays = json.load(f)
            
            interactions = []
            for play in plays:
                completion_rate = play['play_duration'] / play['full_duration'] if play['full_duration'] > 0 else 0
                label = 1 if (completion_rate > 0.6 or play.get('is_liked', 0) == 1) else 0
                
                interactions.append({
                    'user_id': f"user_{play['user_id']}",
                    'song_id': f"song_{play['song_id']}", 
                    'rating': completion_rate,
                    'label': label,
                    'play_time': play['play_time']
                })
            
            print(f"✅ 从JSON文件加载了 {len(interactions)} 条播放记录")
            return interactions
            
        except Exception as e:
            print(f"❌ JSON文件加载错误: {e}")
            return []
    
    def create_mappings(self, interactions):
        """创建用户和歌曲的ID映射"""
        user_ids = list(set([item['user_id'] for item in interactions]))
        song_ids = list(set([item['song_id'] for item in interactions]))
        
        # 排序确保一致性
        user_ids.sort()
        song_ids.sort()
        
        user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
        song_to_idx = {song_id: idx for idx, song_id in enumerate(song_ids)}
        
        print(f"✅ 创建映射: {len(user_ids)} 个用户, {len(song_ids)} 首歌曲")
        
        return user_to_idx, song_to_idx
    
    def generate_training_data(self, interactions, user_to_idx, song_to_idx):
        """生成训练数据"""
        # 正样本（实际播放记录）
        positive_samples = []
        for item in interactions:
            user_idx = user_to_idx[item['user_id']]
            song_idx = song_to_idx[item['song_id']]
            positive_samples.append((user_idx, song_idx, item['label']))
        
        # 负样本（随机生成未播放的用户-歌曲对）
        positive_pairs = set((item[0], item[1]) for item in positive_samples)
        negative_samples = []
        
        # 生成与正样本相同数量的负样本
        while len(negative_samples) < len(positive_samples):
            user_idx = np.random.randint(0, len(user_to_idx))
            song_idx = np.random.randint(0, len(song_to_idx))
            
            if (user_idx, song_idx) not in positive_pairs:
                negative_samples.append((user_idx, song_idx, 0))
        
        # 合并正负样本
        all_samples = positive_samples + negative_samples
        np.random.shuffle(all_samples)
        
        # 分离用户、歌曲和标签
        users = np.array([sample[0] for sample in all_samples])
        songs = np.array([sample[1] for sample in all_samples])
        labels = np.array([sample[2] for sample in all_samples])
        
        print(f"✅ 生成训练数据: {len(positive_samples)} 正样本, {len(negative_samples)} 负样本")
        
        return users, songs, labels
    
    def save_processed_data(self, interactions, user_to_idx, song_to_idx):
        """保存处理后的数据"""
        output_dir = Path("data/mymusic_processed")
        output_dir.mkdir(exist_ok=True)
        
        # 保存交互数据
        np.save(output_dir / "interactions.npy", interactions)
        
        # 保存映射关系
        mappings = {
            'user_to_idx': user_to_idx,
            'song_to_idx': song_to_idx
        }
        np.save(output_dir / "mappings.npy", mappings)
        
        print(f"✅ 处理后的数据已保存到 {output_dir}")

def load_mymusic_interactions():
    """
    加载mymusic数据库的交互数据
    
    Returns:
        tuple: (interactions, user_to_idx, song_to_idx)
    """
    loader = MyMusicDataLoader()
    
    # 首先尝试从数据库加载
    interactions = loader.load_from_database()
    
    # 如果数据库加载失败，尝试从JSON文件加载
    if not interactions:
        json_file = "data/mymusic_generated/user_plays.json"
        if Path(json_file).exists():
            print("⚠️  数据库加载失败，尝试从JSON文件加载...")
            interactions = loader.load_from_json(json_file)
        else:
            raise FileNotFoundError("无法从数据库或JSON文件加载数据")
    
    # 创建映射
    user_to_idx, song_to_idx = loader.create_mappings(interactions)
    
    # 保存处理后的数据
    loader.save_processed_data(interactions, user_to_idx, song_to_idx)
    
    return interactions, user_to_idx, song_to_idx

def generate_mymusic_training_data(interactions, user_to_idx, song_to_idx):
    """
    生成mymusic的训练数据
    
    Args:
        interactions: 交互数据
        user_to_idx: 用户映射
        song_to_idx: 歌曲映射
        
    Returns:
        tuple: (users, songs, labels)
    """
    loader = MyMusicDataLoader()
    return loader.generate_training_data(interactions, user_to_idx, song_to_idx)

def main():
    """测试数据加载功能"""
    print("🎵 测试mymusic数据加载器...")
    
    try:
        interactions, user_to_idx, song_to_idx = load_mymusic_interactions()
        users, songs, labels = generate_mymusic_training_data(interactions, user_to_idx, song_to_idx)
        
        print("\n📊 数据统计:")
        print(f"   用户数量: {len(user_to_idx)}")
        print(f"   歌曲数量: {len(song_to_idx)}")
        print(f"   交互记录: {len(interactions)}")
        print(f"   训练样本: {len(users)}")
        print(f"   正样本比例: {labels.mean():.2%}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    main()
