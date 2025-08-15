#!/usr/bin/env python3
"""
严格分析原始_song_4833.mcz文件结构
"""

import zipfile
import json
import os
import librosa

def analyze_original_mcz(mcz_path):
    """详细分析原始MCZ文件"""
    print(f"🔍 严格分析原始文件: {mcz_path}")
    
    try:
        with zipfile.ZipFile(mcz_path, 'r') as mcz:
            all_files = mcz.namelist()
            print(f"📁 原始文件包含 {len(all_files)} 个文件:")
            for file in all_files:
                info = mcz.getinfo(file)
                print(f"   📄 {file} ({info.file_size} bytes)")
            
            # 查找和分析所有MC文件
            mc_files = [f for f in all_files if f.endswith('.mc')]
            print(f"\n🎼 找到 {len(mc_files)} 个MC文件:")
            
            for mc_file in mc_files:
                print(f"\n📋 分析MC文件: {mc_file}")
                with mcz.open(mc_file, 'r') as f:
                    mc_content = f.read().decode('utf-8')
                    mc_data = json.loads(mc_content)
                
                # 详细分析meta
                meta = mc_data.get('meta', {})
                print(f"   🏷️  Meta字段: {list(meta.keys())}")
                
                # 重要字段
                version = meta.get('version', 'N/A')
                mode = meta.get('mode', 'N/A')
                mode_ext = meta.get('mode_ext', {})
                song = meta.get('song', {})
                
                print(f"   📊 版本: {version}")
                print(f"   🎮 模式: {mode}")
                print(f"   🔧 扩展模式: {mode_ext}")
                print(f"   🎵 歌曲信息: {song}")
                
                # 分析时间信息
                time_info = mc_data.get('time', [])
                print(f"   ⏰ 时间点数量: {len(time_info)}")
                if time_info:
                    print(f"   📍 第一个时间点: {time_info[0]}")
                    if len(time_info) > 1:
                        print(f"   📍 最后一个时间点: {time_info[-1]}")
                
                # 分析音符
                notes = mc_data.get('note', [])
                print(f"   🎵 音符数量: {len(notes)}")
                
                if notes:
                    # 分析音符范围
                    first_note = notes[0]
                    last_note = notes[-1]
                    print(f"   🏁 第一个音符: {first_note}")
                    print(f"   🏁 最后一个音符: {last_note}")
                    
                    # 计算谱面时长
                    def beat_to_seconds(beat, bpm):
                        if len(beat) >= 3:
                            beat_position = beat[0] + beat[1] / beat[2]
                            return beat_position * 60 / bpm
                        return 0
                    
                    if time_info:
                        bpm = time_info[0].get('bpm', 120)
                        first_time = beat_to_seconds(first_note.get('beat', [0, 0, 1]), bpm)
                        last_time = beat_to_seconds(last_note.get('beat', [0, 0, 1]), bpm)
                        print(f"   ⏱️  谱面时长: {first_time:.1f}s - {last_time:.1f}s (总计: {last_time - first_time:.1f}s)")
            
            # 分析音频文件
            audio_files = [f for f in all_files if f.endswith(('.ogg', '.mp3', '.wav'))]
            print(f"\n🎵 音频文件分析:")
            
            for audio_file in audio_files:
                print(f"\n🎶 音频文件: {audio_file}")
                info = mcz.getinfo(audio_file)
                print(f"   💾 文件大小: {info.file_size / 1024 / 1024:.2f} MB")
                
                # 提取并分析音频
                import tempfile
                with tempfile.TemporaryDirectory() as temp_dir:
                    mcz.extract(audio_file, temp_dir)
                    temp_audio_path = os.path.join(temp_dir, audio_file)
                    
                    try:
                        y, sr = librosa.load(temp_audio_path, sr=None)
                        duration = len(y) / sr
                        print(f"   ⏰ 音频时长: {duration:.2f}秒 ({duration/60:.2f}分钟)")
                        
                        # 检查音频是否被截断
                        if duration < 60:
                            print(f"   ⚠️  音频时长异常短！可能被截断")
                        
                    except Exception as e:
                        print(f"   ❌ 音频分析失败: {e}")
                        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()

def compare_with_generated():
    """对比原始文件和生成文件"""
    print(f"\n{'='*60}")
    print(f"🔄 对比原始文件和生成文件")
    
    original_path = "trainData/_song_4833.mcz"
    generated_path = "generated_beatmaps/fixed_song_4833.mcz"
    
    def get_mcz_info(mcz_path):
        try:
            with zipfile.ZipFile(mcz_path, 'r') as mcz:
                # 获取MC文件数据
                mc_files = [f for f in mcz.namelist() if f.endswith('.mc')]
                if mc_files:
                    with mcz.open(mc_files[0], 'r') as f:
                        mc_data = json.loads(f.read().decode('utf-8'))
                    
                    # 获取音频文件信息
                    audio_files = [f for f in mcz.namelist() if f.endswith(('.ogg', '.mp3', '.wav'))]
                    
                    return {
                        'mc_file': mc_files[0],
                        'mc_data': mc_data,
                        'audio_files': audio_files,
                        'all_files': mcz.namelist()
                    }
        except Exception as e:
            print(f"❌ 读取 {mcz_path} 失败: {e}")
            return None
    
    original_info = get_mcz_info(original_path)
    generated_info = get_mcz_info(generated_path)
    
    if original_info and generated_info:
        print(f"\n📊 文件结构对比:")
        print(f"   原始文件: {original_info['all_files']}")
        print(f"   生成文件: {generated_info['all_files']}")
        
        print(f"\n🎼 MC文件名对比:")
        print(f"   原始: {original_info['mc_file']}")
        print(f"   生成: {generated_info['mc_file']}")
        
        print(f"\n🎵 歌曲信息对比:")
        orig_song = original_info['mc_data'].get('meta', {}).get('song', {})
        gen_song = generated_info['mc_data'].get('meta', {}).get('song', {})
        print(f"   原始歌曲: {orig_song}")
        print(f"   生成歌曲: {gen_song}")
        
        print(f"\n🎶 音频文件对比:")
        print(f"   原始音频: {original_info['audio_files']}")
        print(f"   生成音频: {generated_info['audio_files']}")

def main():
    # 分析原始文件
    analyze_original_mcz("trainData/_song_4833.mcz")
    
    # 对比生成文件
    compare_with_generated()

if __name__ == "__main__":
    main()
