#!/usr/bin/env python3
"""
对比标准MCZ和我们生成的音符密度
"""

import zipfile
import json

def compare_note_density():
    """对比音符密度"""
    files = [
        ("标准文件", "trainData/_song_4833.mcz"),
        ("生成文件", "generated_beatmaps/fixed_song_4833.mcz")
    ]
    
    for name, mcz_path in files:
        try:
            print(f"\n🔍 分析 {name}: {mcz_path}")
            
            with zipfile.ZipFile(mcz_path, 'r') as mcz:
                mc_files = [f for f in mcz.namelist() if f.endswith('.mc')]
                
                if name == "标准文件":
                    # 标准文件有多个MC文件，选择4K难度
                    target_mc = None
                    for mc_file in mc_files:
                        with mcz.open(mc_file, 'r') as f:
                            mc_data = json.loads(f.read().decode('utf-8'))
                        meta = mc_data.get('meta', {})
                        version = meta.get('version', '')
                        if '4K' in version and ('Normal' in version or 'Lv.' in version):
                            target_mc = mc_file
                            print(f"   选择难度: {version}")
                            break
                    
                    if not target_mc:
                        target_mc = mc_files[0]  # 如果没找到合适的，用第一个
                        
                    with mcz.open(target_mc, 'r') as f:
                        mc_data = json.loads(f.read().decode('utf-8'))
                else:
                    with mcz.open(mc_files[0], 'r') as f:
                        mc_data = json.loads(f.read().decode('utf-8'))
                
                notes = mc_data.get('note', [])
                game_notes = [note for note in notes if 'column' in note]
                audio_notes = [note for note in notes if 'sound' in note]
                
                print(f"   总音符: {len(notes)}")
                print(f"   游戏音符: {len(game_notes)}")
                print(f"   音频控制: {len(audio_notes)}")
                
                if game_notes:
                    # 计算时间跨度和密度
                    bpm = 104.2  # 假设BPM相同
                    beats_per_second = bpm / 60
                    
                    note_times = []
                    for note in game_notes:
                        beat = note.get('beat', [])
                        if len(beat) >= 3:
                            measure = beat[0]
                            beat_num = beat[1]
                            fraction = beat[2]
                            
                            # 计算时间
                            total_beats = measure * 4 + beat_num / fraction * 4
                            time_seconds = total_beats / beats_per_second
                            note_times.append(time_seconds)
                    
                    if note_times:
                        min_time = min(note_times)
                        max_time = max(note_times)
                        duration = max_time - min_time
                        
                        print(f"   时间范围: {min_time:.1f} - {max_time:.1f}秒")
                        print(f"   持续时间: {duration:.1f}秒")
                        print(f"   音符密度: {len(game_notes)/duration:.2f} 个/秒")
                        
                        # 显示前几个音符
                        print(f"   前3个音符:")
                        sorted_notes = sorted(zip(note_times, game_notes))[:3]
                        for time, note in sorted_notes:
                            print(f"     {time:.1f}秒: {note}")
                    
        except Exception as e:
            print(f"   ❌ 分析失败: {e}")

if __name__ == "__main__":
    compare_note_density()
