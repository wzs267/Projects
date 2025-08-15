#!/usr/bin/env python3
"""
正确分析beat格式和时间计算
"""

import zipfile
import json

def analyze_beat_format():
    """正确分析beat格式"""
    mcz_path = "trainData/_song_4833.mcz"
    
    try:
        with zipfile.ZipFile(mcz_path, 'r') as mcz:
            # 找到4K Another Lv.27的MC文件
            target_mc = "0/1511697495.mc"  # 这应该是4K Another Lv.27
            
            with mcz.open(target_mc, 'r') as f:
                mc_data = json.loads(f.read().decode('utf-8'))
            
            print(f"🎼 分析MC文件: {target_mc}")
            print(f"📋 Meta信息:")
            meta = mc_data.get('meta', {})
            print(f"   版本: {meta.get('version', 'N/A')}")
            if 'song' in meta:
                song = meta['song']
                print(f"   歌曲: {song.get('title', 'N/A')} - {song.get('artist', 'N/A')}")
            
            # 分析time信息（BPM）
            time_info = mc_data.get('time', [])
            print(f"\n⏰ 时间信息 ({len(time_info)}个时间点):")
            for i, time_point in enumerate(time_info):
                print(f"   {i+1}. {time_point}")
            
            # 分析notes
            notes = mc_data.get('note', [])
            game_notes = [note for note in notes if 'column' in note]
            audio_notes = [note for note in notes if 'sound' in note]
            
            print(f"\n🎵 音符统计:")
            print(f"   总音符: {len(notes)}")
            print(f"   游戏音符: {len(game_notes)}")
            print(f"   音频控制: {len(audio_notes)}")
            
            # 详细分析beat格式
            print(f"\n📝 Beat格式分析:")
            print(f"前10个游戏音符的beat格式:")
            for i, note in enumerate(game_notes[:10]):
                beat = note.get('beat', [])
                column = note.get('column', -1)
                print(f"   Note {i+1}: beat={beat}, column={column}")
                
                # 分析beat数组的含义
                if len(beat) >= 3:
                    print(f"      beat[0]={beat[0]} (可能是小节)")
                    print(f"      beat[1]={beat[1]} (可能是小节内位置)")
                    print(f"      beat[2]={beat[2]} (可能是拍子分母)")
            
            # 查找最大的beat值来估算歌曲长度
            max_measure = 0
            max_beat_in_measure = 0
            
            for note in game_notes:
                beat = note.get('beat', [])
                if len(beat) >= 2:
                    measure = beat[0]
                    beat_in_measure = beat[1]
                    max_measure = max(max_measure, measure)
                    if measure == max_measure:
                        max_beat_in_measure = max(max_beat_in_measure, beat_in_measure)
            
            print(f"\n📊 Beat范围统计:")
            print(f"   最大小节: {max_measure}")
            print(f"   最后小节内最大beat: {max_beat_in_measure}")
            
            # 假设BPM计算总时长
            if time_info and len(time_info) > 0:
                bpm = time_info[0].get('bpm', 120)
                print(f"   BPM: {bpm}")
                
                # 估算歌曲长度（假设每小节4拍）
                total_beats = max_measure * 4 + max_beat_in_measure
                total_seconds = total_beats * 60 / bpm
                print(f"   估算总拍数: {total_beats}")
                print(f"   估算时长: {total_seconds:.1f}秒 ({total_seconds/60:.2f}分钟)")
            
            return mc_data
            
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_our_generation():
    """分析我们生成的文件"""
    mcz_path = "generated_beatmaps/fixed_song_4833.mcz"
    
    try:
        with zipfile.ZipFile(mcz_path, 'r') as mcz:
            mc_files = [f for f in mcz.namelist() if f.endswith('.mc')]
            if not mc_files:
                return
            
            with mcz.open(mc_files[0], 'r') as f:
                mc_data = json.loads(f.read().decode('utf-8'))
            
            print(f"\n🤖 我们生成的文件分析:")
            
            notes = mc_data.get('note', [])
            game_notes = [note for note in notes if 'column' in note]
            
            print(f"   游戏音符数: {len(game_notes)}")
            
            # 查找我们的beat范围
            max_measure = 0
            max_beat_in_measure = 0
            
            for note in game_notes:
                beat = note.get('beat', [])
                if len(beat) >= 2:
                    measure = beat[0]
                    beat_in_measure = beat[1]
                    max_measure = max(max_measure, measure)
                    if measure == max_measure:
                        max_beat_in_measure = max(max_beat_in_measure, beat_in_measure)
            
            print(f"   最大小节: {max_measure}")
            print(f"   最后小节内最大beat: {max_beat_in_measure}")
            
            # 计算我们的时长
            time_info = mc_data.get('time', [])
            if time_info:
                bpm = time_info[0].get('bpm', 120)
                total_beats = max_measure * 4 + max_beat_in_measure
                total_seconds = total_beats * 60 / bpm
                print(f"   BPM: {bpm}")
                print(f"   估算时长: {total_seconds:.1f}秒 ({total_seconds/60:.2f}分钟)")
            
    except Exception as e:
        print(f"❌ 分析我们的文件失败: {e}")

def main():
    print("🔍 重新正确分析beat格式和时间计算")
    
    # 分析标准文件
    analyze_beat_format()
    
    # 分析我们的文件
    analyze_our_generation()

if __name__ == "__main__":
    main()
