#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量推荐脚本 - 为多个用户同时生成推荐
"""

import sys
import json
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.predict_mymusic import batch_predict_mymusic, get_song_info_from_db
except ImportError:
    print(json.dumps({"error": "推荐模块导入失败，请确保模型已训练"}))
    sys.exit(1)

def main():
    """批量推荐主函数"""
    try:
        # 检查命令行参数
        if len(sys.argv) < 2:
            print(json.dumps({"error": "缺少用户ID列表参数"}))
            sys.exit(1)
        
        user_ids_json = sys.argv[1]
        top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        
        # 解析用户ID列表
        user_ids = json.loads(user_ids_json)
        
        if not isinstance(user_ids, list):
            print(json.dumps({"error": "用户ID必须是列表格式"}))
            sys.exit(1)
        
        # 批量生成推荐
        batch_results = batch_predict_mymusic(user_ids, top_k)
        
        # 收集所有需要查询的歌曲ID
        all_song_ids = set()
        for user_id, recommendations in batch_results.items():
            for rec in recommendations:
                all_song_ids.add(rec['raw_song_id'])
        
        # 获取歌曲详细信息
        song_info = get_song_info_from_db(list(all_song_ids))
        
        # 格式化结果
        formatted_results = {}
        for user_id, recommendations in batch_results.items():
            user_results = []
            for rec in recommendations:
                song_id = rec['raw_song_id']
                item = {
                    'song_id': song_id,
                    'score': rec['score'],
                    'song_info': song_info.get(song_id, None)
                }
                user_results.append(item)
            formatted_results[user_id] = user_results
        
        # 输出JSON结果
        print(json.dumps({
            "success": True,
            "results": formatted_results,
            "user_count": len(formatted_results),
            "total_recommendations": sum(len(recs) for recs in formatted_results.values())
        }, ensure_ascii=False))
        
    except json.JSONDecodeError:
        print(json.dumps({"error": "用户ID列表JSON格式错误"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"批量推荐失败: {str(e)}"}))
        sys.exit(1)

if __name__ == "__main__":
    main()
