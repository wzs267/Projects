#!/usr/bin/env python3
"""
详细检查标准MCZ的音符结构
"""

import zipfile
import json
import os

def detailed_note_analysis(mcz_path):
    """详细分析音符结构"""
    print(f"🔍 详细分析: {mcz_path}")
    
    try:
        with zipfile.ZipFile(mcz_path, 'r') as mcz:
            mc_files = [f for f in mcz.namelist() if f.endswith('.mc')]
            if not mc_files:
                return
            
            with mcz.open(mc_files[0], 'r') as f:
                mc_data = json.loads(f.read().decode('utf-8'))
            
            notes = mc_data.get('note', [])
            print(f"📊 总共 {len(notes)} 个音符")
            
            # 分析前10个音符的详细结构
            print(f"\n📝 前10个音符详细结构:")
            for i, note in enumerate(notes[:10]):
                print(f"   Note {i}: {note}")
                
                # 检查每个字段
                beat = note.get('beat')
                endbeat = note.get('endbeat')
                column = note.get('column')
                
                print(f"      字段: beat={beat}, endbeat={endbeat}, column={column}")
                
                # 判断音符类型
                if beat and endbeat:
                    if beat == endbeat:
                        note_type = "单点音符 (Tap)"
                    else:
                        note_type = "长按音符 (Long)"
                        # 计算长度
                        if len(beat) >= 3 and len(endbeat) >= 3:
                            beat_pos = beat[0] * 4 + beat[1] * beat[2] / 4
                            endbeat_pos = endbeat[0] * 4 + endbeat[1] * endbeat[2] / 4
                            duration = endbeat_pos - beat_pos
                            note_type += f" (长度: {duration:.2f}拍)"
                else:
                    note_type = "未知类型"
                
                print(f"      类型: {note_type}")
                print()
            
            # 统计音符类型
            tap_count = 0
            long_count = 0
            unknown_count = 0
            
            for note in notes:
                beat = note.get('beat')
                endbeat = note.get('endbeat')
                
                if beat and endbeat:
                    if beat == endbeat:
                        tap_count += 1
                    else:
                        long_count += 1
                else:
                    unknown_count += 1
            
            print(f"📊 音符类型统计:")
            print(f"   单点音符: {tap_count}")
            print(f"   长按音符: {long_count}")
            print(f"   未知类型: {unknown_count}")
            
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()

def check_audio_path_in_mcz():
    """检查MCZ文件中的音频路径引用"""
    mcz_files = [
        "trainData/_song_1203.mcz",
        "generated_beatmaps/fixed_song_4833.mcz"
    ]
    
    for mcz_path in mcz_files:
        if not os.path.exists(mcz_path):
            continue
            
        print(f"\n🔍 检查音频路径: {mcz_path}")
        
        try:
            with zipfile.ZipFile(mcz_path, 'r') as mcz:
                # 检查所有文件
                all_files = mcz.namelist()
                print(f"📁 所有文件: {all_files}")
                
                # 查找MC文件中的音频引用
                mc_files = [f for f in all_files if f.endswith('.mc')]
                if mc_files:
                    with mcz.open(mc_files[0], 'r') as f:
                        mc_data = json.loads(f.read().decode('utf-8'))
                    
                    # 查看meta中是否有音频文件引用
                    meta = mc_data.get('meta', {})
                    print(f"📋 Meta字段键: {list(meta.keys())}")
                    
                    # 查找可能的音频文件字段
                    audio_fields = ['audio', 'music', 'sound', 'file', 'path']
                    for field in audio_fields:
                        if field in meta:
                            print(f"🎵 找到音频字段 {field}: {meta[field]}")
                    
                    # 检查song字段
                    if 'song' in meta:
                        song = meta['song']
                        print(f"🎵 Song字段: {song}")
                        for key, value in song.items():
                            if 'audio' in key.lower() or 'music' in key.lower() or 'file' in key.lower():
                                print(f"   可能的音频引用 {key}: {value}")
                
        except Exception as e:
            print(f"❌ 检查失败: {e}")

def main():
    # 详细分析一个标准文件
    detailed_note_analysis("trainData/_song_1203.mcz")
    
    # 检查音频路径
    check_audio_path_in_mcz()

if __name__ == "__main__":
    main()
