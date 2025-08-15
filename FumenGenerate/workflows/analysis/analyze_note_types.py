#!/usr/bin/env python3
"""
分析标准MCZ中的单点和长按音符格式
"""

import zipfile
import json
import os

def analyze_note_types(mcz_path):
    """分析MCZ文件中的音符类型"""
    print(f"🔍 分析音符类型: {mcz_path}")
    
    try:
        with zipfile.ZipFile(mcz_path, 'r') as mcz:
            # 查找MC文件
            mc_files = [f for f in mcz.namelist() if f.endswith('.mc')]
            if not mc_files:
                print("❌ 未找到MC文件")
                return
            
            # 读取MC文件
            with mcz.open(mc_files[0], 'r') as f:
                mc_data = json.loads(f.read().decode('utf-8'))
            
            notes = mc_data.get('note', [])
            print(f"📊 总共 {len(notes)} 个音符")
            
            # 分析音符类型
            tap_notes = []  # 单点音符
            long_notes = []  # 长按音符
            
            for i, note in enumerate(notes):
                beat = note.get('beat', [])
                endbeat = note.get('endbeat', [])
                
                # 比较beat和endbeat是否相同
                if beat == endbeat:
                    tap_notes.append((i, note))
                else:
                    long_notes.append((i, note))
            
            print(f"🎵 单点音符 (Tap): {len(tap_notes)} 个")
            print(f"🎹 长按音符 (Long): {len(long_notes)} 个")
            
            # 显示单点音符样例
            if tap_notes:
                print(f"\n📝 单点音符格式样例:")
                for i, (idx, note) in enumerate(tap_notes[:5]):
                    print(f"   {i+1}. Note {idx}: {note}")
            
            # 显示长按音符样例
            if long_notes:
                print(f"\n📝 长按音符格式样例:")
                for i, (idx, note) in enumerate(long_notes[:5]):
                    beat = note.get('beat', [])
                    endbeat = note.get('endbeat', [])
                    # 计算长度
                    if len(beat) >= 3 and len(endbeat) >= 3:
                        beat_pos = beat[0] * 4 + beat[1] + beat[2]/4
                        endbeat_pos = endbeat[0] * 4 + endbeat[1] + endbeat[2]/4
                        duration = endbeat_pos - beat_pos
                        print(f"   {i+1}. Note {idx}: {note}")
                        print(f"      长度: {duration:.2f} 拍")
                    else:
                        print(f"   {i+1}. Note {idx}: {note}")
                        print(f"      格式异常: beat={beat}, endbeat={endbeat}")
            
            # 检查音频文件
            print(f"\n🎵 音频文件分析:")
            audio_files = [f for f in mcz.namelist() if f.endswith(('.ogg', '.mp3', '.wav'))]
            for audio_file in audio_files:
                print(f"   📄 {audio_file}")
                # 检查文件大小
                info = mcz.getinfo(audio_file)
                print(f"      大小: {info.file_size} 字节 ({info.file_size/1024/1024:.2f} MB)")
                print(f"      压缩后: {info.compress_size} 字节")
            
            return {
                'tap_notes': len(tap_notes),
                'long_notes': len(long_notes),
                'audio_files': audio_files,
                'sample_tap': tap_notes[0][1] if tap_notes else None,
                'sample_long': long_notes[0][1] if long_notes else None
            }
            
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_audio_files():
    """比较不同MCZ文件的音频"""
    files_to_check = [
        "trainData/_song_1203.mcz",
        "trainData/_song_1011.mcz", 
        "trainData/_song_4833.mcz",
        "generated_beatmaps/fixed_song_4833.mcz"
    ]
    
    print(f"\n🎵 音频文件对比:")
    for mcz_path in files_to_check:
        if os.path.exists(mcz_path):
            print(f"\n📁 {mcz_path}")
            try:
                with zipfile.ZipFile(mcz_path, 'r') as mcz:
                    audio_files = [f for f in mcz.namelist() if f.endswith(('.ogg', '.mp3', '.wav'))]
                    for audio_file in audio_files:
                        info = mcz.getinfo(audio_file)
                        print(f"   🎶 {audio_file}")
                        print(f"      大小: {info.file_size/1024:.1f} KB")
                        print(f"      路径: {audio_file}")
            except Exception as e:
                print(f"   ❌ 读取失败: {e}")
        else:
            print(f"\n📁 {mcz_path} - 文件不存在")

def main():
    # 分析标准样例
    standard_files = [
        "trainData/_song_1203.mcz",
        "trainData/_song_1011.mcz"
    ]
    
    results = {}
    for mcz_path in standard_files:
        if os.path.exists(mcz_path):
            print(f"\n{'='*60}")
            result = analyze_note_types(mcz_path)
            if result:
                results[mcz_path] = result
        else:
            print(f"⚠️  文件不存在: {mcz_path}")
    
    # 对比音频文件
    compare_audio_files()
    
    # 总结发现
    print(f"\n{'='*60}")
    print(f"📋 分析总结:")
    for mcz_path, result in results.items():
        print(f"\n📁 {os.path.basename(mcz_path)}:")
        print(f"   单点音符: {result['tap_notes']} 个")
        print(f"   长按音符: {result['long_notes']} 个")
        if result['sample_tap']:
            print(f"   单点格式: {result['sample_tap']}")
        if result['sample_long']:
            print(f"   长按格式: {result['sample_long']}")

if __name__ == "__main__":
    main()
