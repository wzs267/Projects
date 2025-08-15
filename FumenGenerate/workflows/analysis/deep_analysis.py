#!/usr/bin/env python3
"""
深度检查time、effect和beat格式
"""

import zipfile
import json

def deep_analysis_standard():
    """深度分析标准文件的time和effect"""
    mcz_path = "trainData/_song_4833.mcz"
    
    try:
        with zipfile.ZipFile(mcz_path, 'r') as mcz:
            target_mc = "0/1511697495.mc"
            
            with mcz.open(target_mc, 'r') as f:
                mc_data = json.loads(f.read().decode('utf-8'))
            
            print(f"🕐 深度分析time字段:")
            time_info = mc_data.get('time', [])
            print(f"   BPM变化点数量: {len(time_info)}")
            
            for i, time_point in enumerate(time_info):
                beat = time_point.get('beat', [])
                bpm = time_point.get('bpm', 0)
                
                # 计算实际时间位置
                if len(beat) >= 3:
                    measure = beat[0]
                    numerator = beat[1] 
                    denominator = beat[2]
                    beat_position = measure + numerator / denominator
                    print(f"   {i+1}. Beat: {beat} -> 第{beat_position:.3f}拍, BPM: {bpm}")
                else:
                    print(f"   {i+1}. Beat: {beat}, BPM: {bpm}")
            
            print(f"\n🎛️ 检查effect字段:")
            effect_info = mc_data.get('effect', [])
            print(f"   特效数量: {len(effect_info)}")
            
            for i, effect in enumerate(effect_info):
                beat = effect.get('beat', [])
                scroll = effect.get('scroll')
                jump = effect.get('jump')
                sign = effect.get('sign')
                
                effect_desc = []
                if scroll is not None:
                    effect_desc.append(f"scroll={scroll}")
                if jump is not None:
                    effect_desc.append(f"jump={jump}")
                if sign is not None:
                    effect_desc.append(f"sign={sign}")
                
                if len(beat) >= 3:
                    measure = beat[0]
                    numerator = beat[1]
                    denominator = beat[2]
                    beat_position = measure + numerator / denominator
                    print(f"   {i+1}. Beat: {beat} -> 第{beat_position:.3f}拍, {', '.join(effect_desc)}")
                else:
                    print(f"   {i+1}. Beat: {beat}, {', '.join(effect_desc)}")
            
            print(f"\n🎵 分析beat格式分布:")
            notes = mc_data.get('note', [])
            game_notes = [note for note in notes if 'column' in note]
            
            # 统计不同分母的使用
            denominators = {}
            max_measure = 0
            max_beat_in_measure = 0
            
            for note in game_notes:
                beat = note.get('beat', [])
                if len(beat) >= 3:
                    measure = beat[0]
                    numerator = beat[1]
                    denominator = beat[2]
                    
                    denominators[denominator] = denominators.get(denominator, 0) + 1
                    max_measure = max(max_measure, measure)
                    if measure == max_measure:
                        max_beat_in_measure = max(max_beat_in_measure, numerator)
            
            print(f"   分母使用统计:")
            for denom, count in sorted(denominators.items()):
                percentage = count / len(game_notes) * 100
                print(f"      分母{denom}: {count}个 ({percentage:.1f}%)")
            
            print(f"   最大小节: {max_measure}")
            print(f"   最大小节内位置: {max_beat_in_measure}")
            
            # 重新计算正确的时长
            print(f"\n⏰ 重新计算谱面时长:")
            
            # 使用最后一个BPM来计算
            if time_info:
                last_bpm = time_info[-1].get('bpm', 156)
                print(f"   使用最后的BPM: {last_bpm}")
                
                # 找到最大的beat位置
                max_beat_position = 0
                for note in game_notes:
                    beat = note.get('beat', [])
                    if len(beat) >= 3:
                        measure = beat[0]
                        numerator = beat[1]
                        denominator = beat[2]
                        beat_position = measure * 4 + (numerator / denominator) * 4  # 假设每小节4拍
                        max_beat_position = max(max_beat_position, beat_position)
                
                # 计算时长
                total_seconds = max_beat_position * 60 / last_bpm
                print(f"   最大beat位置: {max_beat_position:.3f}拍")
                print(f"   计算的谱面时长: {total_seconds:.1f}秒 ({total_seconds/60:.2f}分钟)")
            
            return mc_data
            
    except Exception as e:
        print(f"❌ 深度分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_with_our_generation():
    """对比我们的生成"""
    mcz_path = "generated_beatmaps/high_density_song_4833.mcz"
    
    try:
        with zipfile.ZipFile(mcz_path, 'r') as mcz:
            mc_files = [f for f in mcz.namelist() if f.endswith('.mc')]
            
            with mcz.open(mc_files[0], 'r') as f:
                mc_data = json.loads(f.read().decode('utf-8'))
            
            print(f"\n🤖 我们的生成对比:")
            
            time_info = mc_data.get('time', [])
            effect_info = mc_data.get('effect', [])
            
            print(f"   我们的BPM变化点: {len(time_info)}")
            for time_point in time_info:
                print(f"      {time_point}")
            
            print(f"   我们的特效数量: {len(effect_info)}")
            for effect in effect_info:
                print(f"      {effect}")
                
    except Exception as e:
        print(f"❌ 对比失败: {e}")

if __name__ == "__main__":
    deep_analysis_standard()
    compare_with_our_generation()
