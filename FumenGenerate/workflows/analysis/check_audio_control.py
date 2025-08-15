#!/usr/bin/env python3
"""
验证音频控制音符
"""

import zipfile
import json

def check_audio_control():
    """检查音频控制音符"""
    mcz_path = "generated_beatmaps/fixed_song_4833.mcz"
    
    try:
        with zipfile.ZipFile(mcz_path, 'r') as mcz:
            print(f"📁 文件列表: {mcz.namelist()}")
            
            mc_files = [f for f in mcz.namelist() if f.endswith('.mc')]
            if not mc_files:
                print("❌ 未找到MC文件")
                return
            
            print(f"🎼 MC文件: {mc_files}")
            
            with mcz.open(mc_files[0], 'r') as f:
                mc_data = json.loads(f.read().decode('utf-8'))
            
            notes = mc_data.get('note', [])
            print(f"🎵 总音符数: {len(notes)}")
            
            # 查找音频控制音符
            audio_notes = []
            game_notes = []
            
            for i, note in enumerate(notes):
                if 'sound' in note:
                    audio_notes.append((i, note))
                else:
                    game_notes.append((i, note))
            
            print(f"\n🎵 音频控制音符 ({len(audio_notes)}个):")
            for i, (idx, note) in enumerate(audio_notes):
                print(f"   {i+1}. Note {idx}: {note}")
                
                # 检查关键字段
                sound = note.get('sound')
                vol = note.get('vol')
                offset = note.get('offset')
                note_type = note.get('type')
                
                print(f"      音频文件: {sound}")
                print(f"      音量: {vol}")
                print(f"      偏移: {offset}秒")
                print(f"      类型: {note_type}")
            
            print(f"\n🎮 游戏音符 ({len(game_notes)}个):")
            if game_notes:
                print(f"   前3个: {[note[1] for note in game_notes[:3]]}")
            
            # 检查meta信息
            meta = mc_data.get('meta', {})
            song = meta.get('song', {})
            print(f"\n📋 歌曲信息:")
            print(f"   标题: {song.get('title', 'N/A')}")
            print(f"   艺术家: {song.get('artist', 'N/A')}")
            print(f"   版本: {meta.get('version', 'N/A')}")
            
            return len(audio_notes) > 0
            
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = check_audio_control()
    if success:
        print(f"\n✅ 音频控制音符已正确添加！")
        print(f"🎵 现在应该可以播放完整的音频了")
    else:
        print(f"\n❌ 音频控制音符缺失")
