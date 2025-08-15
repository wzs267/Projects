#!/usr/bin/env python3
"""
详细对比标准MC文件和生成的MC文件的meta结构
"""

import zipfile
import json
import os

def detailed_meta_comparison():
    """详细对比meta结构"""
    
    # 检查标准文件
    standard_mcz = "trainData/_song_4833.mcz"
    generated_mcz = "generated_beatmaps/fixed_song_4833.mcz"
    
    print("🔍 详细对比Meta结构\n")
    
    # 分析标准文件
    print("=" * 50)
    print("📋 标准MCZ文件的MC文件分析")
    print("=" * 50)
    
    try:
        with zipfile.ZipFile(standard_mcz, 'r') as mcz:
            mc_files = [f for f in mcz.namelist() if f.endswith('.mc')]
            
            for i, mc_file in enumerate(mc_files):
                print(f"\n📄 MC文件 {i+1}: {mc_file}")
                
                with mcz.open(mc_file, 'r') as f:
                    mc_data = json.loads(f.read().decode('utf-8'))
                
                meta = mc_data.get('meta', {})
                print(f"   Meta字段: {list(meta.keys())}")
                
                # 详细显示meta内容
                for key, value in meta.items():
                    if isinstance(value, dict):
                        print(f"   {key}: {value}")
                    else:
                        print(f"   {key}: {value}")
                        
                # 检查是否有其他音频相关字段
                print(f"   完整meta结构:")
                print(f"   {json.dumps(meta, ensure_ascii=False, indent=2)}")
                
    except Exception as e:
        print(f"❌ 标准文件分析失败: {e}")
    
    # 分析生成文件
    print("\n" + "=" * 50)
    print("📋 生成MCZ文件的MC文件分析")
    print("=" * 50)
    
    try:
        with zipfile.ZipFile(generated_mcz, 'r') as mcz:
            mc_files = [f for f in mcz.namelist() if f.endswith('.mc')]
            
            for mc_file in mc_files:
                print(f"\n📄 MC文件: {mc_file}")
                
                with mcz.open(mc_file, 'r') as f:
                    mc_data = json.loads(f.read().decode('utf-8'))
                
                meta = mc_data.get('meta', {})
                print(f"   Meta字段: {list(meta.keys())}")
                
                # 详细显示meta内容
                print(f"   完整meta结构:")
                print(f"   {json.dumps(meta, ensure_ascii=False, indent=2)}")
                
    except Exception as e:
        print(f"❌ 生成文件分析失败: {e}")

def check_file_naming_pattern():
    """检查文件命名模式"""
    print("\n" + "=" * 50)
    print("🔍 分析文件命名模式")
    print("=" * 50)
    
    mcz_path = "trainData/_song_4833.mcz"
    
    try:
        with zipfile.ZipFile(mcz_path, 'r') as mcz:
            all_files = mcz.namelist()
            
            print("📁 所有文件:")
            audio_files = []
            mc_files = []
            
            for file in all_files:
                print(f"   {file}")
                if file.endswith('.ogg'):
                    audio_files.append(file)
                elif file.endswith('.mc'):
                    mc_files.append(file)
            
            print(f"\n🎵 音频文件: {audio_files}")
            print(f"📄 MC文件: {mc_files}")
            
            # 检查命名模式
            print(f"\n🔍 文件命名分析:")
            for audio_file in audio_files:
                audio_name = os.path.basename(audio_file).replace('.ogg', '')
                print(f"   音频文件基名: {audio_name}")
                
                # 查找同名的MC文件
                matching_mc = None
                for mc_file in mc_files:
                    mc_name = os.path.basename(mc_file).replace('.mc', '')
                    if mc_name == audio_name:
                        matching_mc = mc_file
                        break
                
                if matching_mc:
                    print(f"   ✅ 找到同名MC文件: {matching_mc}")
                    print(f"   💡 可能的命名约定: 音频和MC文件使用相同的基名")
                else:
                    print(f"   ❌ 未找到同名MC文件")
                    print(f"   🤔 游戏可能使用其他方式关联音频")
    
    except Exception as e:
        print(f"❌ 分析失败: {e}")

def main():
    detailed_meta_comparison()
    check_file_naming_pattern()

if __name__ == "__main__":
    main()
