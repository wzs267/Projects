#!/usr/bin/env python3
"""
检查标准文件中的offset参数
"""

import zipfile
import json

def check_offset_in_standard():
    """检查标准文件的offset参数"""
    mcz_path = "trainData/_song_4833.mcz"
    
    try:
        with zipfile.ZipFile(mcz_path, 'r') as mcz:
            # 检查4K Another Lv.27
            target_mc = "0/1511697495.mc"
            
            with mcz.open(target_mc, 'r') as f:
                mc_data = json.loads(f.read().decode('utf-8'))
            
            notes = mc_data.get('note', [])
            
            print(f"🔍 检查标准文件的音频控制音符:")
            
            # 查找音频控制音符
            audio_notes = [note for note in notes if 'sound' in note]
            
            for i, note in enumerate(audio_notes):
                print(f"   Audio Note {i+1}: {note}")
                
                sound = note.get('sound')
                vol = note.get('vol')
                offset = note.get('offset')
                note_type = note.get('type')
                
                print(f"      音频文件: {sound}")
                print(f"      音量: {vol}")
                print(f"      偏移: {offset} 毫秒 = {offset/1000:.2f} 秒")
                print(f"      类型: {note_type}")
                
                if offset and offset > 0:
                    print(f"      ⚠️  音频延迟了 {offset/1000:.2f} 秒！")
                    print(f"      这可能解释了为什么只播放部分音频！")
                
    except Exception as e:
        print(f"❌ 检查失败: {e}")

def check_our_offset():
    """检查我们生成的offset"""
    mcz_path = "generated_beatmaps/high_density_song_4833.mcz"
    
    try:
        with zipfile.ZipFile(mcz_path, 'r') as mcz:
            mc_files = [f for f in mcz.namelist() if f.endswith('.mc')]
            
            with mcz.open(mc_files[0], 'r') as f:
                mc_data = json.loads(f.read().decode('utf-8'))
            
            notes = mc_data.get('note', [])
            audio_notes = [note for note in notes if 'sound' in note]
            
            print(f"\n🤖 检查我们生成的音频控制音符:")
            
            for i, note in enumerate(audio_notes):
                print(f"   Audio Note {i+1}: {note}")
                
                offset = note.get('offset', 0)
                print(f"      我们的偏移: {offset} 毫秒")
                
    except Exception as e:
        print(f"❌ 检查我们的文件失败: {e}")

if __name__ == "__main__":
    check_offset_in_standard()
    check_our_offset()
