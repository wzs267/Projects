#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试批量分析器 - 只分析前5个MCZ文件
"""

import os
from batch_analyzer import TrainingDataAnalyzer

def quick_test():
    """快速测试分析器"""
    data_dir = r"d:\Projects\FumenGenerate\trainData"
    
    # 获取前5个MCZ文件
    mcz_files = [f for f in os.listdir(data_dir) if f.endswith('.mcz')][:5]
    
    print(f"将分析以下文件: {mcz_files}")
    
    # 创建分析器
    analyzer = TrainingDataAnalyzer(data_dir)
    
    # 只分析选定的文件
    for mcz_file in mcz_files:
        mcz_path = os.path.join(data_dir, mcz_file)
        try:
            print(f"正在分析: {mcz_file}")
            mcz_data = analyzer.parser.parse_mcz_file(mcz_path)
            analysis = analyzer._analyze_single_mcz(mcz_data)
            analyzer.analysis_results.extend(analysis)
        except Exception as e:
            print(f"分析失败 {mcz_file}: {e}")
    
    # 保存结果
    df = analyzer.save_analysis_results("quick_test_results.csv")
    
    # 生成报告
    analyzer.generate_summary_report(df)
    
    return df

if __name__ == "__main__":
    df = quick_test()
