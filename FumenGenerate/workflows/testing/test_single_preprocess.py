#!/usr/bin/env python3
"""
简单测试预处理单个文件
"""

import os
import sys

# 添加项目根目录和相关路径到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'core'))
sys.path.insert(0, os.path.join(project_root, 'workflows', 'preprocessing'))

from batch_mcz_preprocessor import MCZBatchPreprocessor

def test_single_file():
    """测试单个文件预处理"""
    # 切换到项目根目录以确保路径正确
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    os.chdir(project_root)
    
    preprocessor = MCZBatchPreprocessor()
    
    mcz_file = "_song_10088.mcz"
    print(f"🔍 测试预处理: {mcz_file}")
    print(f"📁 当前工作目录: {os.getcwd()}")
    
    try:
        results = preprocessor.process_single_mcz(mcz_file)
        print(f"✅ 结果数量: {len(results)}")
        
        if results:
            print(f"📊 第一个结果:")
            result = results[0]
            for key, value in result.items():
                if key in ['notes', 'timing_points']:
                    print(f"   {key}: {len(value)} 项")
                else:
                    print(f"   {key}: {value}")
        else:
            print("❌ 没有返回结果")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_file()
