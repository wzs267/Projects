#!/usr/bin/env python3
"""
验证高密度谱面
"""

import zipfile
import json

def verify_high_density():
    """验证高密度谱面"""
    mcz_path = "generated_beatmaps/high_density_song_4833.mcz"
    
    try:
        with zipfile.ZipFile(mcz_path, 'r') as mcz:
            print(f"📁 文件列表: {mcz.namelist()}")
            
            mc_files = [f for f in mcz.namelist() if f.endswith('.mc')]
            with mcz.open(mc_files[0], 'r') as f:
                mc_data = json.loads(f.read().decode('utf-8'))
            
            notes = mc_data.get('note', [])
            game_notes = [note for note in notes if 'column' in note]
            audio_notes = [note for note in notes if 'sound' in note]
            
            print(f"🎵 总音符数: {len(notes)}")
            print(f"🎮 游戏音符: {len(game_notes)}")
            print(f"🎵 音频控制: {len(audio_notes)}")
            
            # 检查beat格式
            print(f"\n📝 Beat格式检查:")
            for i, note in enumerate(game_notes[:5]):
                beat = note.get('beat', [])
                column = note.get('column', -1)
                print(f"   Note {i+1}: beat={beat}, column={column}")
            
            # 计算时间跨度
            max_measure = 0
            max_subdivision = 0
            subdivisions = 24
            
            for note in game_notes:
                beat = note.get('beat', [])
                if len(beat) >= 3:
                    measure = beat[0]
                    subdivision = beat[1]
                    max_measure = max(max_measure, measure)
                    if measure == max_measure:
                        max_subdivision = max(max_subdivision, subdivision)
            
            # 计算时长
            time_info = mc_data.get('time', [])
            if time_info:
                bpm = time_info[0].get('bpm', 156)
                beats_per_measure = 4
                total_beats = max_measure * beats_per_measure + (max_subdivision / subdivisions) * beats_per_measure
                total_seconds = total_beats * 60 / bpm
                
                print(f"\n📊 时间分析:")
                print(f"   最大小节: {max_measure}")
                print(f"   最大细分: {max_subdivision}")
                print(f"   BPM: {bpm}")
                print(f"   估算时长: {total_seconds:.1f}秒 ({total_seconds/60:.2f}分钟)")
                print(f"   密度: {len(game_notes)/total_seconds:.1f} 个/秒")
            
            # 检查列分布
            column_counts = {}
            for note in game_notes:
                col = note.get('column', -1)
                column_counts[col] = column_counts.get(col, 0) + 1
            
            print(f"\n🎹 按键分布:")
            for col, count in sorted(column_counts.items()):
                percentage = count / len(game_notes) * 100
                print(f"   键 {col}: {count} 个 ({percentage:.1f}%)")
            
            return len(game_notes) > 1400  # 检查是否接近目标密度
            
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False

if __name__ == "__main__":
    success = verify_high_density()
    if success:
        print(f"\n✅ 高密度谱面验证成功！")
        print(f"🎮 现在应该可以播放完整的2分20秒了")
    else:
        print(f"\n❌ 验证失败")
