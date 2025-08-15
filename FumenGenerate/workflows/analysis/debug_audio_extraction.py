#!/usr/bin/env python3
"""
详细检查MCZ文件解析过程和音频文件
"""

import zipfile
import json
import os
import librosa
import tempfile

def analyze_mcz_audio_files(mcz_path):
    """详细分析MCZ文件中的音频文件"""
    print(f"🔍 详细分析MCZ文件: {mcz_path}")
    
    if not os.path.exists(mcz_path):
        print(f"❌ 文件不存在: {mcz_path}")
        return
    
    try:
        with zipfile.ZipFile(mcz_path, 'r') as mcz:
            all_files = mcz.namelist()
            print(f"📁 MCZ包含 {len(all_files)} 个文件:")
            for file in all_files:
                info = mcz.getinfo(file)
                print(f"   📄 {file}")
                print(f"      大小: {info.file_size} 字节 ({info.file_size/1024/1024:.2f} MB)")
                print(f"      压缩大小: {info.compress_size} 字节")
                print(f"      压缩率: {(1-info.compress_size/info.file_size)*100:.1f}%")
            
            # 查找音频文件
            audio_files = [f for f in all_files if f.endswith(('.ogg', '.mp3', '.wav'))]
            print(f"\n🎵 音频文件 ({len(audio_files)}个):")
            
            for audio_file in audio_files:
                print(f"\n   🎶 分析音频文件: {audio_file}")
                
                # 提取到临时文件
                temp_dir = tempfile.mkdtemp()
                mcz.extract(audio_file, temp_dir)
                temp_audio_path = os.path.join(temp_dir, audio_file)
                
                try:
                    # 使用librosa分析音频
                    y, sr = librosa.load(temp_audio_path, sr=None)
                    duration = len(y) / sr
                    
                    print(f"      时长: {duration:.2f} 秒 ({duration/60:.2f} 分钟)")
                    print(f"      采样率: {sr} Hz")
                    print(f"      样本数: {len(y)}")
                    
                    # 检查音频质量
                    if duration < 30:
                        print(f"      ⚠️  音频时长过短!")
                    if sr < 44100:
                        print(f"      ⚠️  采样率较低!")
                    
                except Exception as e:
                    print(f"      ❌ 音频分析失败: {e}")
                
                # 清理临时文件
                try:
                    os.remove(temp_audio_path)
                    os.rmdir(temp_dir)
                except:
                    pass
            
            # 检查MC文件中的音频引用
            mc_files = [f for f in all_files if f.endswith('.mc')]
            print(f"\n📋 MC文件分析:")
            for mc_file in mc_files:
                print(f"\n   📄 {mc_file}")
                with mcz.open(mc_file, 'r') as f:
                    mc_data = json.loads(f.read().decode('utf-8'))
                
                meta = mc_data.get('meta', {})
                song = meta.get('song', {})
                
                print(f"      歌曲标题: {song.get('title', 'N/A')}")
                print(f"      艺术家: {song.get('artist', 'N/A')}")
                
                # 检查音频文件引用
                if 'file' in song:
                    audio_ref = song['file']
                    print(f"      音频引用: {audio_ref}")
                    
                    # 检查引用的文件是否存在
                    audio_ref_path = f"0/{audio_ref}"
                    if audio_ref_path in all_files:
                        print(f"      ✅ 音频文件存在")
                    else:
                        print(f"      ❌ 音频文件不存在! 引用路径: {audio_ref_path}")
                        print(f"      可用的音频文件: {audio_files}")
                else:
                    print(f"      ❌ 未找到音频文件引用")
                    
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()

def compare_original_vs_generated():
    """对比原始MCZ和生成的MCZ"""
    files_to_compare = [
        ("trainData/_song_4833.mcz", "原始训练数据"),
        ("generated_beatmaps/fixed_song_4833.mcz", "AI生成版本")
    ]
    
    print(f"\n{'='*60}")
    print(f"🔄 对比原始文件 vs 生成文件")
    
    for mcz_path, description in files_to_compare:
        print(f"\n{'='*30} {description} {'='*30}")
        analyze_mcz_audio_files(mcz_path)

def check_audio_extraction_process():
    """检查音频提取过程"""
    print(f"\n{'='*60}")
    print(f"🔍 检查音频提取过程")
    
    mcz_path = "trainData/_song_4833.mcz"
    
    try:
        with zipfile.ZipFile(mcz_path, 'r') as mcz:
            all_files = mcz.namelist()
            audio_files = [f for f in all_files if f.endswith(('.ogg', '.mp3', '.wav'))]
            
            print(f"📁 原始MCZ文件结构:")
            for file in all_files:
                print(f"   {file}")
            
            print(f"\n🎵 检测到的音频文件:")
            for audio_file in audio_files:
                print(f"   {audio_file}")
                
                # 模拟提取过程
                temp_dir = tempfile.mkdtemp()
                print(f"   提取到临时目录: {temp_dir}")
                
                mcz.extract(audio_file, temp_dir)
                extracted_path = os.path.join(temp_dir, audio_file)
                print(f"   提取后路径: {extracted_path}")
                
                if os.path.exists(extracted_path):
                    file_size = os.path.getsize(extracted_path)
                    print(f"   ✅ 提取成功，文件大小: {file_size} 字节")
                    
                    # 检查音频
                    try:
                        y, sr = librosa.load(extracted_path, sr=None)
                        duration = len(y) / sr
                        print(f"   🎵 音频时长: {duration:.2f} 秒")
                        
                        if duration < 60:
                            print(f"   ⚠️  警告：音频时长异常短! 预期应该是127秒")
                    except Exception as e:
                        print(f"   ❌ 音频读取失败: {e}")
                else:
                    print(f"   ❌ 提取失败")
                
                # 清理
                try:
                    if os.path.exists(extracted_path):
                        os.remove(extracted_path)
                    os.rmdir(temp_dir)
                except:
                    pass
                    
    except Exception as e:
        print(f"❌ 检查失败: {e}")

def main():
    compare_original_vs_generated()
    check_audio_extraction_process()

if __name__ == "__main__":
    main()
