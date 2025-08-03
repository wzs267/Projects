#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mymusic音乐推荐系统主程序
统一管理数据生成、训练、预测等操作
"""

import sys
import os
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def generate_mymusic_data():
    """生成mymusic测试数据"""
    print("🎵 生成mymusic测试数据...")
    try:
        from scripts.data_generation.mymusic_data_generator import main as generate_main
        generate_main()
        print("✅ mymusic数据生成完成")
    except Exception as e:
        print(f"❌ 数据生成失败: {e}")

def train_mymusic_model():
    """训练mymusic模型"""
    print("🧠 训练mymusic模型...")
    try:
        from src.train_mymusic import train_mymusic_model
        train_mymusic_model()
        print("✅ mymusic模型训练完成")
    except Exception as e:
        print(f"❌ 模型训练失败: {e}")

def predict_mymusic():
    """生成mymusic推荐"""
    print("🔮 生成mymusic推荐...")
    try:
        from src.predict_mymusic import main as predict_main
        predict_main()
        print("✅ mymusic推荐生成完成")
    except Exception as e:
        print(f"❌ 推荐生成失败: {e}")

def check_mymusic_system():
    """检查mymusic系统状态"""
    print("🔍 检查mymusic系统状态...")
    
    # 检查必要的目录和文件
    checks = [
        ("数据目录", "data/mymusic_generated"),
        ("SQL脚本目录", "sql_scripts/mymusic"),
        ("模型目录", "models"),
        ("消费者数据", "data/mymusic_generated/consumers.json"),
        ("播放记录数据", "data/mymusic_generated/user_plays.json"),
        ("消费者SQL脚本", "sql_scripts/mymusic/insert_consumers.sql"),
        ("播放记录SQL脚本", "sql_scripts/mymusic/create_and_insert_user_plays.sql"),
        ("训练模型", "models/mymusic_twin_tower.keras"),
        ("映射文件", "data/mymusic_processed/mappings.npy")
    ]
    
    print(f"\n📋 系统检查结果:")
    all_good = True
    
    for name, path in checks:
        if Path(path).exists():
            print(f"   ✅ {name}: {path}")
        else:
            print(f"   ❌ {name}: {path} (不存在)")
            all_good = False
    
    if all_good:
        print(f"\n🎉 系统状态良好！")
    else:
        print(f"\n⚠️  系统存在缺失文件，请运行相应命令生成")
    
    # 检查数据库连接（可选）
    try:
        import pymysql
        db_config = {
            'host': '127.0.0.1',
            'user': 'root',
            'password': '123456',
            'database': 'mymusic'
        }
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
        
        # 检查表是否存在
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]
        
        required_tables = ['consumers', 'songs', 'singers', 'user_plays']
        print(f"\n📊 数据库检查:")
        
        for table in required_tables:
            if table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"   ✅ {table}: {count} 条记录")
            else:
                print(f"   ❌ {table}: 表不存在")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"\n⚠️  数据库连接检查失败: {e}")

def run_mymusic_all():
    """运行完整的mymusic流程"""
    print("🚀 开始mymusic完整流程...")
    print("=" * 60)
    
    try:
        # 1. 生成数据
        generate_mymusic_data()
        print()
        
        # 2. 训练模型
        train_mymusic_model()
        print()
        
        # 3. 生成推荐
        predict_mymusic()
        print()
        
        print("🎉 mymusic完整流程执行完毕！")
        
    except Exception as e:
        print(f"❌ 流程执行失败: {e}")

def show_mymusic_usage():
    """显示使用说明"""
    print("""
🎵 mymusic音乐推荐系统使用指南
=" * 60

📋 命令说明:
   generate  - 生成mymusic测试数据(consumers + user_plays)
   train     - 训练推荐模型
   predict   - 生成推荐结果
   check     - 检查系统状态
   all       - 运行完整流程

🚀 快速开始:
   1. python main_mymusic.py generate    # 生成测试数据
   2. 在MySQL中执行生成的SQL脚本        # 初始化数据库
   3. python main_mymusic.py train      # 训练模型
   4. python main_mymusic.py predict    # 生成推荐

📁 重要文件:
   - data/mymusic_generated/            # 生成的JSON数据
   - sql_scripts/mymusic/              # SQL插入脚本  
   - models/mymusic_twin_tower.keras   # 训练好的模型
   - src/predict_mymusic.py            # 推荐预测脚本

💡 提示:
   - 确保MySQL数据库'mymusic'已创建
   - 数据库配置: host=127.0.0.1, user=root, password=123456
   - 系统基于现有的10首歌曲生成推荐
""")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='mymusic音乐推荐系统')
    parser.add_argument('command', nargs='?', choices=[
        'generate', 'train', 'predict', 'check', 'all', 'help'
    ], help='执行的命令', default='help')
    
    args = parser.parse_args()
    
    print("🎵 mymusic音乐推荐系统")
    print("=" * 50)
    
    if args.command == 'generate':
        generate_mymusic_data()
    elif args.command == 'train':
        train_mymusic_model()
    elif args.command == 'predict':
        predict_mymusic()
    elif args.command == 'check':
        check_mymusic_system()
    elif args.command == 'all':
        run_mymusic_all()
    elif args.command == 'help':
        show_mymusic_usage()
    else:
        show_mymusic_usage()

if __name__ == "__main__":
    main()
