#!/usr/bin/env python3
"""
对比分析生成的MCZ和训练集MCZ的格式差异
"""
import zipfile
import json
import os

def analyze_mcz_structure(mcz_path, label):
    """分析MCZ文件结构"""
    print(f"\n{'='*50}")
    print(f"分析 {label}: {mcz_path}")
    print(f"{'='*50}")
    
    if not os.path.exists(mcz_path):
        print(f"❌ 文件不存在: {mcz_path}")
        return
    
    try:
        with zipfile.ZipFile(mcz_path, 'r') as z:
            files = z.namelist()
            print(f"📁 文件数量: {len(files)}")
            print(f"📁 文件列表:")
            for f in sorted(files):
                print(f"   {f}")
            
            # 分析JSON文件
            json_files = [f for f in files if f.endswith('.json')]
            mc_files = [f for f in files if f.endswith('.mc')]
            ogg_files = [f for f in files if f.endswith('.ogg')]
            
            print(f"\n📄 JSON文件: {len(json_files)}")
            print(f"🎼 MC文件: {len(mc_files)}")
            print(f"🎵 OGG文件: {len(ogg_files)}")
            
            # 分析JSON内容
            for json_file in json_files:
                print(f"\n--- {json_file} 内容 ---")
                try:
                    with z.open(json_file) as f:
                        data = json.load(f)
                        print(json.dumps(data, indent=2, ensure_ascii=False))
                except Exception as e:
                    print(f"❌ 读取JSON失败: {e}")
            
            # 分析MC文件内容
            for mc_file in mc_files[:3]:  # 只看前3个
                print(f"\n--- {mc_file} 内容 (前20行) ---")
                try:
                    with z.open(mc_file) as f:
                        content = f.read().decode('utf-8', errors='ignore')
                        lines = content.split('\n')[:20]
                        for i, line in enumerate(lines, 1):
                            print(f"{i:2d}: {line}")
                        if len(content.split('\n')) > 20:
                            print("    ... (更多内容)")
                except Exception as e:
                    print(f"❌ 读取MC文件失败: {e}")
                    
    except Exception as e:
        print(f"❌ 分析失败: {e}")

def main():
    # 分析生成的MCZ
    analyze_mcz_structure("generated_beatmaps/generated_song_4833.mcz", "AI生成的MCZ")
    
    # 分析训练集中的标准MCZ
    analyze_mcz_structure("trainData/_song_1203.mcz", "训练集标准MCZ")

if __name__ == "__main__":
    main()
