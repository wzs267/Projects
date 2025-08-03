# data_generator.py
# 音乐推荐系统数据生成器
# 生成用户、歌曲和播放记录的测试数据
import json
import random
from datetime import datetime, timedelta
import os

def generate_test_data():
    """生成简化的测试数据"""
    
    # 基础数据
    genres = ['流行', '摇滚', '民谣', '电子', '爵士', '古典', '说唱', '乡村']
    artists = ['周杰伦', '林俊杰', '邓紫棋', '陈奕迅', '薛之谦', '张学友', '王力宏', '李荣浩']
    song_titles = ['青花瓷', '稻香', '告白气球', '可惜没如果', '演员', '默', '体面', '凉凉']
    
    # 生成用户数据
    users = []
    for i in range(100):
        users.append({
            'user_id': f'user_{i}',
            'username': f'用户{i:03d}',
            'email': f'user{i}@example.com'
        })
    
    # 生成歌曲数据
    songs = []
    for i in range(500):
        songs.append({
            'song_id': f'song_{i}',
            'title': f'{random.choice(song_titles)}_{i}',
            'artist': random.choice(artists),
            'album': f'专辑{random.randint(1, 50)}',
            'genre': random.choice(genres),
            'duration_seconds': random.randint(180, 360)
        })
    
    # 生成播放记录
    plays = []
    device_types = ['mobile', 'desktop', 'tablet', 'smart_speaker']
    
    for user in users:
        # 每个用户随机播放一些歌曲
        user_songs = random.sample(songs, random.randint(20, 80))
        
        for song in user_songs:
            # 每首歌可能被播放多次
            play_count = random.choices([1, 2, 3, 4, 5], weights=[50, 25, 15, 7, 3])[0]
            
            for _ in range(play_count):
                # 生成播放时长
                duration_type = random.choices(['full', 'partial', 'skip'], weights=[40, 35, 25])[0]
                
                if duration_type == 'full':
                    play_duration = random.randint(int(song['duration_seconds'] * 0.8), song['duration_seconds'])
                elif duration_type == 'partial':
                    play_duration = random.randint(30, int(song['duration_seconds'] * 0.7))
                else:  # skip
                    play_duration = random.randint(5, 29)
                
                plays.append({
                    'user_id': user['user_id'],
                    'song_id': song['song_id'],
                    'play_duration': play_duration,
                    'session_id': f'session_{random.randint(1000, 9999)}',
                    'device_type': random.choice(device_types)
                })
    
    return users, songs, plays

def save_json_data(users, songs, plays):
    """保存数据到JSON文件"""
    os.makedirs('test_data', exist_ok=True)
    
    with open('test_data/users.json', 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)
    
    with open('test_data/songs.json', 'w', encoding='utf-8') as f:
        json.dump(songs, f, ensure_ascii=False, indent=2)
    
    with open('test_data/user_plays.json', 'w', encoding='utf-8') as f:
        json.dump(plays, f, ensure_ascii=False, indent=2)
    
    print(f"生成测试数据:")
    print(f"- 用户数据: {len(users)} 条")
    print(f"- 歌曲数据: {len(songs)} 条")
    print(f"- 播放记录: {len(plays)} 条")

def generate_sql_insert_files(users, songs, plays):
    """生成SQL插入文件"""
    os.makedirs('sql_scripts', exist_ok=True)
    
    # 生成用户插入SQL
    with open('sql_scripts/insert_users.sql', 'w', encoding='utf-8') as f:
        f.write("-- 插入用户数据\n")
        f.write("USE music_recommendation;\n\n")
        for user in users:
            f.write(f"INSERT INTO users (user_id, username, email) VALUES ('{user['user_id']}', '{user['username']}', '{user['email']}');\n")
    
    # 生成歌曲插入SQL
    with open('sql_scripts/insert_songs.sql', 'w', encoding='utf-8') as f:
        f.write("-- 插入歌曲数据\n")
        f.write("USE music_recommendation;\n\n")
        for song in songs:
            f.write(f"INSERT INTO songs (song_id, title, artist, album, genre, duration_seconds) VALUES ('{song['song_id']}', '{song['title']}', '{song['artist']}', '{song['album']}', '{song['genre']}', {song['duration_seconds']});\n")

    # 生成播放记录插入SQL（分批）
    batch_size = 1000
    for i in range(0, len(plays), batch_size):
        batch_num = i // batch_size + 1
        with open(f'sql_scripts/insert_plays_batch_{batch_num}.sql', 'w', encoding='utf-8') as f:
            f.write(f"-- 插入播放记录数据 (批次 {batch_num})\n")
            f.write("USE music_recommendation;\n\n")
            
            batch_plays = plays[i:i + batch_size]
            for play in batch_plays:
                f.write(f"INSERT INTO user_plays (user_id, song_id, play_duration, session_id, device_type) VALUES ('{play['user_id']}', '{play['song_id']}', {play['play_duration']}, '{play['session_id']}', '{play['device_type']}');\n")
    
    print(f"SQL脚本已生成到 sql_scripts/ 目录")

def generate_recommendation_data(plays):
    """生成推荐算法所需的数据格式"""
    os.makedirs('data/raw', exist_ok=True)
    
    # 转换为推荐算法需要的格式
    algo_data = []
    for play in plays:
        algo_data.append({
            "user_id": play['user_id'],
            "song_id": play['song_id'],
            "duration": play['play_duration']
        })
    
    with open('data/raw/user_plays.json', 'w', encoding='utf-8') as f:
        json.dump(algo_data, f, ensure_ascii=False, indent=2)
    
    print(f"推荐算法数据已保存到: data/raw/user_plays.json ({len(algo_data)} 条记录)")

if __name__ == "__main__":
    print("正在生成测试数据...")
    
    # 生成数据
    users, songs, plays = generate_test_data()
    
    # 保存JSON格式
    save_json_data(users, songs, plays)
    
    # 生成SQL插入脚本
    generate_sql_insert_files(users, songs, plays)
    
    # 生成推荐算法数据
    generate_recommendation_data(plays)
    
    print("\n所有文件生成完成！")
    print("\n使用说明:")
    print("1. 首先执行 database_schema.sql 创建数据库和表")
    print("2. 然后依次执行 sql_scripts/ 目录下的SQL文件插入数据")
    print("3. 或者使用 database_setup.py 脚本自动插入（需要安装 mysql-connector-python）")
    print("4. data/raw/user_plays.json 可直接用于推荐算法训练")
