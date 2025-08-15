#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速生成启动器
"""

import os
import sys

def main():
    print("🎵 FumenGenerate 快速生成启动器")
    print("=" * 40)
    print("1. final_demo.py - 完整系统演示 ⭐推荐")
    print("2. deep_beatmap_generator.py - 深度学习生成器")
    print("3. improved_precise_generator.py - 精确生成器")
    print("4. high_density_beatmap_generator.py - 高密度生成器")
    
    choice = input("\n请选择生成方案 (1-4): ").strip()
    
    generation_scripts = {
        '1': 'final_demo.py',
        '2': 'deep_beatmap_generator.py',
        '3': 'improved_precise_generator.py', 
        '4': 'high_density_beatmap_generator.py'
    }
    
    if choice in generation_scripts:
        script = generation_scripts[choice]
        print(f"\n🎵 启动生成: {script}")
        os.system(f"python {script}")
    else:
        print("❌ 无效选择")

if __name__ == "__main__":
    main()
