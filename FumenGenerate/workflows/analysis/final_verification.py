#!/usr/bin/env python3
"""
最终验证修复后的MCZ文件
"""

import zipfile
import json
import os

def final_verification():
    """最终验证生成的MCZ文件"""
    mcz_path = "generated_beatmaps/fixed_song_4833.mcz"
    
    print("🔍 最终验证修复后的MCZ文件")
    print("=" * 50)
    
    try:
        with zipfile.ZipFile(mcz_path, 'r') as mcz:
            all_files = mcz.namelist()
            print(f"📁 文件列表:")
            for file in all_files:
                print(f"   {file}")
            
            # 检查文件命名约定
            audio_files = [f for f in all_files if f.endswith('.ogg')]
            mc_files = [f for f in all_files if f.endswith('.mc')]
            
            print(f"\n🎵 音频文件: {audio_files}")
            print(f"📄 MC文件: {mc_files}")
            
            # 验证文件名匹配
            if audio_files and mc_files:
                audio_basename = os.path.basename(audio_files[0]).replace('.ogg', '')
                mc_basename = os.path.basename(mc_files[0]).replace('.mc', '')
                
                if audio_basename == mc_basename:
                    print(f"✅ 文件名匹配: {audio_basename}")
                else:
                    print(f"❌ 文件名不匹配:")
                    print(f"   音频: {audio_basename}")
                    print(f"   MC: {mc_basename}")
            
            # 检查MC文件内容
            if mc_files:
                print(f"\n📋 MC文件内容验证:")
                with mcz.open(mc_files[0], 'r') as f:
                    mc_data = json.loads(f.read().decode('utf-8'))
                
                meta = mc_data.get('meta', {})
                
                # 检查关键字段
                required_fields = ['creator', 'background', 'version', 'id', 'mode', 'time', 'song', 'mode_ext']
                missing_fields = []
                extra_fields = []
                
                for field in required_fields:
                    if field not in meta:
                        missing_fields.append(field)
                
                standard_fields = set(required_fields)
                actual_fields = set(meta.keys())
                extra_fields = actual_fields - standard_fields
                
                if not missing_fields and not extra_fields:
                    print(f"✅ Meta字段完全匹配标准格式")
                else:
                    if missing_fields:
                        print(f"❌ 缺少字段: {missing_fields}")
                    if extra_fields:
                        print(f"⚠️  额外字段: {list(extra_fields)}")
                
                # 显示meta内容
                print(f"\n📊 Meta结构:")
                print(f"   版本: {meta.get('version')}")
                print(f"   模式: {meta.get('mode')}")
                print(f"   键数: {meta.get('mode_ext', {}).get('column')}")
                print(f"   歌曲: {meta.get('song', {}).get('title')} - {meta.get('song', {}).get('artist')}")
                
                # 检查音符
                notes = mc_data.get('note', [])
                print(f"\n🎵 音符验证:")
                print(f"   音符数量: {len(notes)}")
                
                if notes:
                    # 检查音符格式
                    first_note = notes[0]
                    note_fields = set(first_note.keys())
                    expected_tap_fields = {'beat', 'column'}
                    
                    if note_fields == expected_tap_fields:
                        print(f"✅ 音符格式正确 (单点音符)")
                    else:
                        print(f"❌ 音符格式异常:")
                        print(f"   期望字段: {expected_tap_fields}")
                        print(f"   实际字段: {note_fields}")
                    
                    # 统计按键分布
                    column_counts = {}
                    for note in notes:
                        col = note.get('column', -1)
                        column_counts[col] = column_counts.get(col, 0) + 1
                    print(f"   按键分布: {column_counts}")
                
        print(f"\n🎉 验证完成！")
        print(f"📁 文件路径: {mcz_path}")
        print(f"💡 这个文件应该可以在游戏中正常播放音乐并识别为4K谱面")
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    final_verification()
