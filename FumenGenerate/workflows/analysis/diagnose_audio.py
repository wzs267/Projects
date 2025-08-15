#!/usr/bin/env python3
"""
import sys
import os
# 修复工作区重组后的导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)


诊断音频特征提取问题
"""

import os
import sys
import tempfile
import shutil
import zipfile
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.mcz_parser import MCZParser
from core.four_k_extractor import FourKBeatmapExtractor
from core.audio_beatmap_analyzer import AudioBeatmapAnalyzer

def diagnose_audio_extraction(mcz_file: str):
    """诊断音频特征提取问题"""
    print(f"\n🔍 诊断文件: {mcz_file}")
    
    traindata_dir = "trainData"
    mcz_path = os.path.join(traindata_dir, mcz_file)
    
    parser = MCZParser()
    extractor = FourKBeatmapExtractor()
    analyzer = AudioBeatmapAnalyzer()
    
    temp_dir = None
    try:
        # 步骤1: 创建临时目录并解压
        temp_dir = tempfile.mkdtemp(prefix="mcz_debug_")
        print(f"  📁 临时目录: {temp_dir}")
        
        with zipfile.ZipFile(mcz_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        print("  ✅ MCZ文件解压成功")
        
        # 步骤2: 解析MCZ
        song_data = parser.parse_mcz_file(mcz_path)
        if not song_data:
            print("  ❌ MCZ解析失败")
            return False
        print("  ✅ MCZ解析成功")
        
        # 步骤3: 提取4K谱面
        beatmaps_4k = extractor.extract_4k_beatmap(song_data)
        if not beatmaps_4k:
            print("  ❌ 4K谱面提取失败")
            return False
        print(f"  ✅ 找到 {len(beatmaps_4k)} 个4K谱面")
        
        # 步骤4: 查找音频文件
        audio_files = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.lower().endswith(('.ogg', '.mp3', '.wav')):
                    audio_files.append(os.path.join(root, file))
        
        if not audio_files:
            print("  ❌ 未找到音频文件")
            return False
        
        audio_file = audio_files[0]
        print(f"  ✅ 找到音频文件: {os.path.basename(audio_file)}")
        
        # 步骤5: 测试音频特征提取
        print(f"  🎵 测试音频特征提取...")
        try:
            audio_features = analyzer.extract_audio_features(audio_file)
            if audio_features:
                print(f"  ✅ 音频特征提取成功")
                print(f"      采样率: {audio_features.sr}")
                print(f"      时长: {audio_features.duration:.2f}秒")
                print(f"      时间帧数: {len(audio_features.time_frames)}")
                
                # 步骤6: 测试谱面转换
                print(f"  🎯 测试谱面事件转换...")
                beatmap = beatmaps_4k[0]
                
                # 将FourKBeatmap转换为字典格式
                beatmap_dict = {
                    'notes': [
                        {
                            'beat': note.beat,  # 保持beat格式
                            'column': note.column,
                            'type': 'normal',  # 默认类型
                            'end_time': note.endbeat if hasattr(note, 'endbeat') and note.endbeat else None
                        } for note in beatmap.notes
                    ],
                    'timing_points': [
                        {
                            'beat': tp.beat,  # 保持beat格式
                            'bpm': tp.bpm,
                            'time_signature': [4, 4]  # 默认4/4拍
                        } for tp in beatmap.timing_points
                    ]
                }
                
                beatmap_events = analyzer.extract_beatmap_events(beatmap_dict)
                if beatmap_events:
                    print(f"  ✅ 谱面事件转换成功，事件数: {len(beatmap_events)}")
                    
                    # 步骤7: 测试对齐
                    print(f"  🔗 测试音频谱面对齐...")
                    aligned_data = analyzer.align_audio_beatmap(audio_features, beatmap_events)
                    if aligned_data and len(aligned_data.audio_features) > 100:
                        print(f"  ✅ 对齐成功!")
                        print(f"      对齐后特征数: {len(aligned_data.audio_features)}")
                        print(f"      对齐后事件数: {len(aligned_data.beatmap_events)}")
                        return True
                    else:
                        print(f"  ❌ 对齐失败或样本数不足")
                        print(f"      aligned_data: {aligned_data}")
                        if aligned_data:
                            print(f"      特征数: {len(aligned_data.audio_features)}")
                        return False
                else:
                    print(f"  ❌ 谱面事件转换失败")
                    return False
            else:
                print(f"  ❌ 音频特征提取返回空值")
                return False
        except Exception as e:
            print(f"  ❌ 音频特征提取异常: {e}")
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"  ❌ 总体异常: {e}")
        traceback.print_exc()
        return False
        
    finally:
        # 清理临时目录
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

def main():
    """主函数"""
    print("🔍 音频特征提取诊断工具")
    print("=" * 50)
    
    # 测试前3个文件
    traindata_dir = "trainData"
    mcz_files = [f for f in os.listdir(traindata_dir) if f.endswith('.mcz')][:3]
    
    success_count = 0
    
    for mcz_file in mcz_files:
        if diagnose_audio_extraction(mcz_file):
            success_count += 1
    
    print(f"\n📊 诊断结果: {success_count}/{len(mcz_files)} 文件处理成功")
    
    if success_count == 0:
        print("❌ 所有文件音频特征提取都失败")
        print("💡 建议检查librosa安装或音频文件格式")
    else:
        print(f"✅ {success_count} 个文件音频特征提取成功")

if __name__ == "__main__":
    main()
