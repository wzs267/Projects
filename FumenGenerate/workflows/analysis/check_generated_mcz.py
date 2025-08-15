#!/usr/bin/env python3
import zipfile
import json

mcz_path = 'generated_beatmaps/generated_song_4833.mcz'

with zipfile.ZipFile(mcz_path, 'r') as z:
    # 查看谱面文件
    with z.open('Song 4833_4K_AI_Lv20.json') as beatmap_file:
        beatmap_data = json.load(beatmap_file)
        print('=== 谱面文件内容 ===')
        print(f'歌曲标题: {beatmap_data.get("song_title")}')
        print(f'艺术家: {beatmap_data.get("artist")}')
        print(f'难度: {beatmap_data.get("difficulty")}')
        print(f'按键数: {beatmap_data.get("keys")}K')
        print(f'音频文件: {beatmap_data.get("audio_file")}')
        
        notes = beatmap_data.get('notes', [])
        print(f'Notes总数: {len(notes)}')
        
        if notes:
            print('\n前10个Notes:')
            for i, note in enumerate(notes[:10]):
                time_sec = note.get('time', 0) / 1000
                print(f'  {i+1:2d}. {time_sec:6.2f}s - 列{note.get("column", 0)}')
            
            # 统计每列的notes数量
            column_counts = {}
            for note in notes:
                col = note.get('column', 0)
                column_counts[col] = column_counts.get(col, 0) + 1
            
            print('\n各列Notes分布:')
            for col in sorted(column_counts.keys()):
                print(f'  列{col}: {column_counts[col]} 个notes')
                
            # 统计时间分布
            if len(notes) > 1:
                duration_ms = notes[-1]['time'] - notes[0]['time']
                duration_sec = duration_ms / 1000
                print(f'\n谱面时长: {duration_sec:.1f}秒')
                print(f'平均密度: {len(notes)/duration_sec:.2f} notes/秒')
