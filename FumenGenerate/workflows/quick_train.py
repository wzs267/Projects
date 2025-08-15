#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速训练启动器
"""

import os
import sys

def main():
    print("🎮 FumenGenerate 快速训练启动器")
    print("=" * 40)
    print("1. enhanced_weighted_fusion_training_3_7.py - 最新权重融合训练 ⭐推荐")
    print("2. large_scale_real_training.py - 大规模真实数据训练") 
    print("3. weighted_fusion_large_scale_training_2_8.py - 2:8权重融合训练")
    print("4. large_scale_training.py - 基础深度学习训练")
    
    choice = input("\n请选择训练方案 (1-4): ").strip()
    
    training_scripts = {
        '1': 'enhanced_weighted_fusion_training_3_7.py',
        '2': 'large_scale_real_training.py', 
        '3': 'weighted_fusion_large_scale_training_2_8.py',
        '4': 'large_scale_training.py'
    }
    
    if choice in training_scripts:
        script = training_scripts[choice]
        print(f"\n🚀 启动训练: {script}")
        os.system(f"python {script}")
    else:
        print("❌ 无效选择")

if __name__ == "__main__":
    main()
