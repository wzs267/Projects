#!/usr/bin/env python3
"""
分析生成音符的时间分布
"""

import zipfile
import json

def analyze_note_timing():
    """分析生成音符的时间分布"""
    mcz_path = "generated_beatmaps/fixed_song_4833.mcz"
    
    try:
        with zipfile.ZipFile(mcz_path, 'r') as mcz:
            mc_files = [f for f in mcz.namelist() if f.endswith('.mc')]
            with mcz.open(mc_files[0], 'r') as f:
                mc_data = json.loads(f.read().decode('utf-8'))
        
        notes = mc_data.get('note', [])
        game_notes = [note for note in notes if 'column' in note]  # 排除音频控制音符
        
        print(f"🎵 游戏音符数量: {len(game_notes)}")
        
        # 分析音符时间分布
        note_times = []
        tempo = 104.2  # BPM
        beats_per_second = tempo / 60
        
        for note in game_notes:
            beat = note.get('beat', [])
            if len(beat) >= 3:
                measure = beat[0]
                sub_beat = beat[1] 
                fraction = beat[2]
                
                # 计算实际时间（秒）
                beat_position = measure * 4 + sub_beat + (1.0 / fraction)
                time_seconds = beat_position / beats_per_second
                note_times.append(time_seconds)
        
        if note_times:
            min_time = min(note_times)
            max_time = max(note_times)
            
            print(f"⏰ 音符时间范围:")
            print(f"   最早: {min_time:.1f}秒")
            print(f"   最晚: {max_time:.1f}秒")
            print(f"   时间跨度: {max_time - min_time:.1f}秒")
            
            # 分析每10秒的音符数量
            print(f"\n📊 每10秒音符分布:")
            for i in range(0, int(max_time) + 10, 10):
                start_time = i
                end_time = i + 10
                count = len([t for t in note_times if start_time <= t < end_time])
                print(f"   {start_time:3d}-{end_time:3d}秒: {count:3d}个音符")
            
            # 显示前10个和后10个音符
            sorted_notes = sorted(zip(note_times, game_notes), key=lambda x: x[0])
            
            print(f"\n📝 前10个音符:")
            for i, (time, note) in enumerate(sorted_notes[:10]):
                print(f"   {i+1:2d}. {time:6.1f}秒 - {note}")
            
            print(f"\n📝 后10个音符:")
            for i, (time, note) in enumerate(sorted_notes[-10:]):
                print(f"   {i+1:2d}. {time:6.1f}秒 - {note}")
                
            return max_time
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return 0

if __name__ == "__main__":
    max_time = analyze_note_timing()
    if max_time > 0:
        print(f"\n🎯 问题发现:")
        if max_time < 100:
            print(f"   音符只覆盖到 {max_time:.1f}秒，需要扩展到140秒！")
        else:
            print(f"   音符覆盖 {max_time:.1f}秒，看起来正常")
    else:
        print(f"❌ 无法分析音符时间")
