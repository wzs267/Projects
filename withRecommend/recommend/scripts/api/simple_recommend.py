#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为后端API提供的简单推荐脚本
通过命令行参数接收用户ID，返回JSON格式的推荐结果
"""

import sys
import json
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.predict_mymusic import predict_for_mymusic_user, get_song_info_from_db
except ImportError:
    print(json.dumps({"error": "推荐模块导入失败，请确保模型已训练"}, ensure_ascii=True))
    sys.exit(1)

def main():
    """主函数 - 处理命令行参数并返回推荐结果"""
    try:
        # 检查命令行参数
        if len(sys.argv) < 2:
            print(json.dumps({"error": "缺少用户ID参数"}, ensure_ascii=True))
            sys.exit(1)
        
        user_id = sys.argv[1]
        top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        
        # 生成推荐
        recommendations = predict_for_mymusic_user(user_id, top_k)
        
        # 获取歌曲详细信息（如果可以连接数据库）
        song_ids = [rec['raw_song_id'] for rec in recommendations]
        song_info = get_song_info_from_db(song_ids)
        
        # 合并推荐和歌曲信息
        result = []
        for rec in recommendations:
            song_id = rec['raw_song_id']
            item = {
                'song_id': song_id,
                'score': rec['score'],
                'song_info': song_info.get(song_id, None)
            }
            result.append(item)
        
        # 输出JSON结果
        print(json.dumps({
            "success": True,
            "user_id": user_id,
            "recommendations": result,
            "count": len(result)
        }, ensure_ascii=True))
        
    except ValueError as e:
        print(json.dumps({"error": f"参数错误: {str(e)}"}, ensure_ascii=True))
        sys.exit(1)
    except FileNotFoundError as e:
        print(json.dumps({"error": f"文件不存在: {str(e)}"}, ensure_ascii=True))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"推荐生成失败: {str(e)}"}, ensure_ascii=True))
        sys.exit(1)

if __name__ == "__main__":
    main()
