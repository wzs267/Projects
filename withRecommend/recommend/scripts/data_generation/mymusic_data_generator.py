#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为实际数据库mymusic生成模拟数据
包括：
1. consumers表的模拟用户数据
2. user_plays播放记录表（基于现有10首歌曲）
"""

import random
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

def generate_consumers_data(num_users=100):
    """生成consumers表的模拟用户数据"""
    consumers = []
    
    # 用户名前缀
    username_prefixes = ['user', 'music_lover', 'melody', 'rhythm', 'harmony', 'beat', 'tune', 'sound']
    
    # 地区列表
    locations = ['北京', '上海', '广州', '深圳', '杭州', '南京', '成都', '武汉', '西安', '天津', 
                '重庆', '苏州', '青岛', '长沙', '宁波', '大连', '厦门', '无锡', '福州', '济南']
    
    # 座右铭/个人介绍
    introductions = [
        '音乐是生活的调味剂', '用音乐治愈心灵', '每一首歌都有故事', '音乐让生活更美好',
        '热爱音乐的普通人', '音乐是我的精神食粮', '在音符中寻找共鸣', '音乐无国界',
        '用心感受每一个音符', '音乐是最好的陪伴', '生活需要音乐点缀', '在旋律中找到自己',
        '音乐让世界更精彩', '用音乐记录生活', '音乐是心灵的语言', '热爱一切美好的声音'
    ]
    
    for i in range(num_users):
        user_id = i + 3  # 从3开始，因为现有数据库已有user1和user2
        
        consumer = {
            'id': user_id,
            'username': f"{random.choice(username_prefixes)}_{user_id}",
            'password': 'e10adc3949ba59abbe56e057f20f883e',  # 默认密码 123456 的MD5
            'sex': str(random.choice([0, 1])),  # 0=女, 1=男
            'phone_num': f"1{random.randint(3,9)}{random.randint(0,9)}{random.randint(10000000,99999999)}",
            'email': f"user{user_id}@example.com",
            'birth': f"{random.randint(1980,2005)}-{random.randint(1,12):02d}-{random.randint(1,28):02d} 00:00:00",
            'introduction': random.choice(introductions),
            'location': random.choice(locations),
            'avatar': '/img/singerPic/1635182970215liudehua.jpg',  # 默认头像
            'create_time': '2025-07-22 00:00:00',
            'update_time': '2025-07-22 00:00:00',
            'createdAt': '2025-07-22 00:00:00',
            'updatedAt': '2025-07-22 00:00:00'
        }
        consumers.append(consumer)
    
    return consumers

def generate_user_plays_data(consumers, num_plays=5000):
    """
    生成用户播放记录数据
    基于现有的10首歌曲 (song_id: 1-10)
    """
    plays = []
    song_ids = list(range(1, 11))  # 10首歌曲
    
    # 为每个用户生成播放记录
    for consumer in consumers:
        user_id = consumer['id']
        # 每个用户播放30-80次
        user_play_count = random.randint(30, 80)
        
        for _ in range(user_play_count):
            song_id = random.choice(song_ids)
            
            # 生成播放时长 (秒)，一般歌曲3-5分钟
            full_duration = random.randint(180, 300)
            play_duration = random.randint(30, full_duration)  # 实际播放时长
            
            # 根据播放时长判断是否喜欢 (播放超过60%认为喜欢)
            is_liked = 1 if play_duration / full_duration > 0.6 else 0
            
            # 生成随机播放时间 (最近3个月内)
            base_time = datetime.now() - timedelta(days=90)
            random_days = random.randint(0, 90)
            random_hours = random.randint(0, 23)
            random_minutes = random.randint(0, 59)
            play_time = base_time + timedelta(days=random_days, hours=random_hours, minutes=random_minutes)
            
            play = {
                'user_id': user_id,
                'song_id': song_id,
                'play_duration': play_duration,
                'full_duration': full_duration,
                'is_liked': is_liked,
                'play_time': play_time.strftime('%Y-%m-%d %H:%M:%S'),
                'device_type': random.choice(['mobile', 'web', 'desktop']),
                'created_at': play_time.strftime('%Y-%m-%d %H:%M:%S'),
                'updated_at': play_time.strftime('%Y-%m-%d %H:%M:%S')
            }
            plays.append(play)
    
    # 打乱播放记录顺序
    random.shuffle(plays)
    return plays[:num_plays]  # 限制总数

def save_json_data(consumers, plays):
    """保存JSON格式的数据"""
    output_dir = Path("data/mymusic_generated")
    output_dir.mkdir(exist_ok=True)
    
    # 保存用户数据
    with open(output_dir / "consumers.json", 'w', encoding='utf-8') as f:
        json.dump(consumers, f, ensure_ascii=False, indent=2)
    
    # 保存播放记录数据
    with open(output_dir / "user_plays.json", 'w', encoding='utf-8') as f:
        json.dump(plays, f, ensure_ascii=False, indent=2)
    
    print(f"✅ JSON数据已保存到 {output_dir}")
    print(f"   - 用户数量: {len(consumers)}")
    print(f"   - 播放记录数量: {len(plays)}")

def generate_sql_scripts(consumers, plays):
    """生成SQL插入脚本"""
    sql_dir = Path("sql_scripts/mymusic")
    sql_dir.mkdir(exist_ok=True)
    
    # 生成consumers表插入脚本
    consumers_sql = """-- 插入consumers表数据
-- 生成时间: {}

""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    for consumer in consumers:
        consumers_sql += f"""INSERT INTO `consumers` (`id`, `username`, `password`, `sex`, `phone_num`, `email`, `birth`, `introduction`, `location`, `avatar`, `create_time`, `update_time`, `createdAt`, `updatedAt`) 
VALUES ({consumer['id']}, '{consumer['username']}', '{consumer['password']}', '{consumer['sex']}', '{consumer['phone_num']}', '{consumer['email']}', '{consumer['birth']}', '{consumer['introduction']}', '{consumer['location']}', '{consumer['avatar']}', '{consumer['create_time']}', '{consumer['update_time']}', '{consumer['createdAt']}', '{consumer['updatedAt']}');
"""
    
    with open(sql_dir / "insert_consumers.sql", 'w', encoding='utf-8') as f:
        f.write(consumers_sql)
    
    # 生成user_plays表创建和插入脚本
    plays_sql = """-- 创建user_plays播放记录表
-- 生成时间: {}

DROP TABLE IF EXISTS `user_plays`;
CREATE TABLE `user_plays` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL COMMENT '用户ID',
  `song_id` int NOT NULL COMMENT '歌曲ID', 
  `play_duration` int NOT NULL COMMENT '播放时长(秒)',
  `full_duration` int NOT NULL COMMENT '歌曲总时长(秒)',
  `is_liked` tinyint(1) DEFAULT 0 COMMENT '是否喜欢(0=否,1=是)',
  `play_time` datetime NOT NULL COMMENT '播放时间',
  `device_type` varchar(20) DEFAULT 'mobile' COMMENT '设备类型',
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_user_id` (`user_id`),
  KEY `idx_song_id` (`song_id`),
  KEY `idx_play_time` (`play_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='用户播放记录表';

-- 插入播放记录数据
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # 分批插入播放记录，每1000条一个批次
    batch_size = 1000
    for i in range(0, len(plays), batch_size):
        batch = plays[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        batch_sql = f"\n-- 批次 {batch_num}\nINSERT INTO `user_plays` (`user_id`, `song_id`, `play_duration`, `full_duration`, `is_liked`, `play_time`, `device_type`, `created_at`, `updated_at`) VALUES\n"
        
        values = []
        for play in batch:
            values.append(f"({play['user_id']}, {play['song_id']}, {play['play_duration']}, {play['full_duration']}, {play['is_liked']}, '{play['play_time']}', '{play['device_type']}', '{play['created_at']}', '{play['updated_at']}')")
        
        batch_sql += ",\n".join(values) + ";\n"
        plays_sql += batch_sql
        
        # 每个批次单独保存一个文件
        batch_filename = f"insert_user_plays_batch_{batch_num}.sql"
        with open(sql_dir / batch_filename, 'w', encoding='utf-8') as f:
            f.write(f"-- 用户播放记录插入脚本 - 批次 {batch_num}\n")
            f.write(f"-- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(batch_sql)
    
    # 保存完整的播放记录脚本
    with open(sql_dir / "create_and_insert_user_plays.sql", 'w', encoding='utf-8') as f:
        f.write(plays_sql)
    
    print(f"✅ SQL脚本已生成到 {sql_dir}")
    print(f"   - insert_consumers.sql: {len(consumers)} 用户数据")
    print(f"   - create_and_insert_user_plays.sql: 完整播放记录脚本")
    print(f"   - insert_user_plays_batch_*.sql: 分批播放记录脚本")

def generate_mymusic_config():
    """生成适配mymusic数据库的配置文件"""
    config_content = '''# mymusic数据库配置
class MyMusicConfig:
    # 数据库配置
    DATABASE_NAME = "mymusic"
    
    # 数据表配置
    USERS_TABLE = "consumers"
    SONGS_TABLE = "songs" 
    PLAYS_TABLE = "user_plays"
    
    # 数据范围
    NUM_USERS = 102  # 包含现有的2个用户 + 100个新用户
    NUM_SONGS = 10   # 现有的10首歌曲
    
    # 模型参数 (与原配置保持一致)
    EMBEDDING_DIM = 64
    LSTM_UNITS = 128
    DENSE_UNITS = 64
    USER_TOWER_OUTPUT_DIM = 64
    SONG_TOWER_OUTPUT_DIM = 64
    
    # 训练参数
    EPOCHS = 10
    BATCH_SIZE = 32
    
    # 路径配置
    DATA_PATH = "data/mymusic_generated/user_plays.json"
    PROCESSED_PATH = "data/mymusic_processed/interactions.npy"
    MAPPINGS_PATH = "data/mymusic_processed/mappings.npy"
    MODEL_SAVE_PATH = "models/mymusic_twin_tower.keras"
'''
    
    config_dir = Path("src")
    config_dir.mkdir(exist_ok=True)
    
    with open(config_dir / "mymusic_config.py", 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"✅ 配置文件已生成: src/mymusic_config.py")

def main():
    """主函数"""
    print("🎵 开始生成mymusic数据库的模拟数据...")
    print("=" * 60)
    
    # 生成数据
    consumers = generate_consumers_data(num_users=100)
    plays = generate_user_plays_data(consumers, num_plays=5000)
    
    # 保存数据
    save_json_data(consumers, plays)
    generate_sql_scripts(consumers, plays)
    generate_mymusic_config()
    
    print("\n" + "=" * 60)
    print("🎉 数据生成完成！")
    print("\n📋 数据统计:")
    print(f"   👥 用户数量: {len(consumers)} (ID: 3-102)")
    print(f"   🎵 歌曲数量: 10 (使用现有歌曲)")
    print(f"   📊 播放记录: {len(plays)} 条")
    
    print("\n📁 生成的文件:")
    print("   📄 data/mymusic_generated/consumers.json")
    print("   📄 data/mymusic_generated/user_plays.json")
    print("   📄 sql_scripts/mymusic/insert_consumers.sql")
    print("   📄 sql_scripts/mymusic/create_and_insert_user_plays.sql")
    print("   📄 sql_scripts/mymusic/insert_user_plays_batch_*.sql")
    print("   📄 src/mymusic_config.py")
    
    print("\n🚀 下一步操作:")
    print("   1. 执行SQL脚本创建播放记录表并插入数据")
    print("   2. 修改推荐系统配置文件使用mymusic数据库")
    print("   3. 更新数据加载器以适配新的数据库结构")

if __name__ == "__main__":
    main()
