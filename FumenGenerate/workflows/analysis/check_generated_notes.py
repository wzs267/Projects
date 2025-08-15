#!/usr/bin/env python3
"""
检查生成文件的音符格式
"""

import zipfile
import json

def check_generated_notes():
    """检查生成文件的音符格式"""
    mcz_path = "generated_beatmaps/fixed_song_4833.mcz"
    
    try:
        with zipfile.ZipFile(mcz_path, 'r') as mcz:
            mc_files = [f for f in mcz.namelist() if f.endswith('.mc')]
            if not mc_files:
                print("❌ 未找到MC文件")
                return
            
            with mcz.open(mc_files[0], 'r') as f:
                mc_data = json.loads(f.read().decode('utf-8'))
            
            notes = mc_data.get('note', [])
            print(f"🎵 生成的音符数量: {len(notes)}")
            
            # 检查前10个音符
            print(f"\n📝 前10个生成的音符:")
            for i, note in enumerate(notes[:10]):
                print(f"   Note {i}: {note}")
                if 'endbeat' in note:
                    print(f"      ❌ 包含endbeat字段 (应该是单点音符)")
                else:
                    print(f"      ✅ 单点音符格式正确")
            
            # 统计音符类型
            tap_count = 0
            long_count = 0
            
            for note in notes:
                if 'endbeat' in note:
                    long_count += 1
                else:
                    tap_count += 1
            
            print(f"\n📊 生成的音符类型统计:")
            print(f"   单点音符: {tap_count}")
            print(f"   长按音符: {long_count}")
            
            # 检查音频路径
            meta = mc_data.get('meta', {})
            song = meta.get('song', {})
            if 'file' in song:
                print(f"\n🎵 音频文件引用: {song['file']}")
                print(f"✅ 音频路径已添加到meta中")
            else:
                print(f"\n❌ 未找到音频文件引用")
            
    except Exception as e:
        print(f"❌ 检查失败: {e}")

if __name__ == "__main__":
    check_generated_notes()
