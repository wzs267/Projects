#!/usr/bin/env python3
"""
修正beat格式理解后的时长计算
"""

import zipfile
import json

def correct_time_calculation():
    """使用正确的beat格式计算时长"""
    mcz_path = "trainData/_song_4833.mcz"
    
    try:
        with zipfile.ZipFile(mcz_path, 'r') as mcz:
            target_mc = "0/1511697495.mc"
            
            with mcz.open(target_mc, 'r') as f:
                mc_data = json.loads(f.read().decode('utf-8'))
            
            notes = mc_data.get('note', [])
            game_notes = [note for note in notes if 'column' in note]
            time_info = mc_data.get('time', [])
            
            print(f"🔢 使用正确的beat格式计算:")
            
            if time_info:
                bpm = time_info[0].get('bpm', 156)
                print(f"   BPM: {bpm}")
                
                # 找到最大的beat位置 - 使用正确的公式
                max_beat_value = 0
                max_note = None
                
                for note in game_notes:
                    beat = note.get('beat', [])
                    if len(beat) >= 3:
                        x, y, z = beat[0], beat[1], beat[2]
                        # 正确的beat值计算：x + y/z
                        beat_value = x + y / z
                        if beat_value > max_beat_value:
                            max_beat_value = beat_value
                            max_note = note
                
                print(f"   最大beat值: {max_beat_value:.3f}拍")
                print(f"   最大beat音符: {max_note}")
                
                # 计算时长 - 使用正确的公式
                total_seconds = max_beat_value * 60 / bpm
                print(f"   正确计算的时长: {total_seconds:.1f}秒 ({total_seconds/60:.2f}分钟)")
                
                # 验证几个音符的时间
                print(f"\n📝 验证前几个音符的时间:")
                for i, note in enumerate(game_notes[:5]):
                    beat = note.get('beat', [])
                    if len(beat) >= 3:
                        x, y, z = beat[0], beat[1], beat[2]
                        beat_value = x + y / z
                        time_seconds = beat_value * 60 / bpm
                        print(f"   Note {i+1}: beat={beat} -> {beat_value:.3f}拍 -> {time_seconds:.2f}秒")
                
                return total_seconds
                
    except Exception as e:
        print(f"❌ 计算失败: {e}")
        return None

def check_our_time_calculation():
    """检查我们生成文件的时间计算"""
    mcz_path = "generated_beatmaps/high_density_song_4833.mcz"
    
    try:
        with zipfile.ZipFile(mcz_path, 'r') as mcz:
            mc_files = [f for f in mcz.namelist() if f.endswith('.mc')]
            
            with mcz.open(mc_files[0], 'r') as f:
                mc_data = json.loads(f.read().decode('utf-8'))
            
            notes = mc_data.get('note', [])
            game_notes = [note for note in notes if 'column' in note]
            time_info = mc_data.get('time', [])
            
            print(f"\n🤖 我们生成文件的时间计算:")
            
            if time_info:
                bpm = time_info[0].get('bpm', 156)
                print(f"   BPM: {bpm}")
                
                # 找到最大的beat位置
                max_beat_value = 0
                
                for note in game_notes:
                    beat = note.get('beat', [])
                    if len(beat) >= 3:
                        x, y, z = beat[0], beat[1], beat[2]
                        beat_value = x + y / z
                        max_beat_value = max(max_beat_value, beat_value)
                
                total_seconds = max_beat_value * 60 / bpm
                print(f"   最大beat值: {max_beat_value:.3f}拍")
                print(f"   我们的计算时长: {total_seconds:.1f}秒 ({total_seconds/60:.2f}分钟)")
                
                return total_seconds
                
    except Exception as e:
        print(f"❌ 检查我们的文件失败: {e}")
        return None

if __name__ == "__main__":
    standard_time = correct_time_calculation()
    our_time = check_our_time_calculation()
    
    if standard_time and our_time:
        ratio = our_time / standard_time
        print(f"\n📊 时长对比:")
        print(f"   标准文件: {standard_time:.1f}秒")
        print(f"   我们的文件: {our_time:.1f}秒")
        print(f"   比例: {ratio:.3f}")
        
        if abs(ratio - 0.25) < 0.05:  # 接近1/4
            print(f"   🎯 比例接近1/4，这解释了35秒 vs 2分钟的问题！")
        
        # 计算正确的目标
        target_time = 126  # 2分06秒
        print(f"\n🎯 目标分析:")
        print(f"   实际歌曲长度: {target_time}秒")
        print(f"   标准文件过长: {standard_time/target_time:.1f}倍")
        print(f"   我们的文件: {our_time/target_time:.1f}倍")
