#!/usr/bin/env python3
"""
验证修复后的MCZ格式
"""

import zipfile
import json
import os

def check_fixed_mcz(mcz_path):
    """检查修复后的MCZ文件格式"""
    print(f"🔍 验证修复后的MCZ文件: {mcz_path}")
    
    if not os.path.exists(mcz_path):
        print(f"❌ 文件不存在: {mcz_path}")
        return
    
    try:
        with zipfile.ZipFile(mcz_path, 'r') as mcz:
            file_list = mcz.namelist()
            print(f"📁 文件列表 ({len(file_list)}个文件):")
            for file in file_list:
                print(f"   📄 {file}")
            
            # 查找MC文件
            mc_files = [f for f in file_list if f.endswith('.mc')]
            print(f"\n🎼 MC文件 ({len(mc_files)}个):")
            for mc_file in mc_files:
                print(f"   📄 {mc_file}")
                
                # 读取MC文件内容
                with mcz.open(mc_file, 'r') as f:
                    mc_content = f.read().decode('utf-8')
                    mc_data = json.loads(mc_content)
                    
                    print(f"\n📋 MC文件结构分析:")
                    print(f"   🏷️  顶级字段: {list(mc_data.keys())}")
                    
                    # 检查meta字段
                    if 'meta' in mc_data:
                        meta = mc_data['meta']
                        print(f"   📊 meta字段: {list(meta.keys())}")
                        print(f"      版本: {meta.get('version', 'N/A')}")
                        print(f"      模式: {meta.get('mode', 'N/A')}")
                        if 'mode_ext' in meta:
                            print(f"      键数: {meta['mode_ext'].get('column', 'N/A')}")
                        if 'song' in meta:
                            song = meta['song']
                            print(f"      歌曲: {song.get('title', 'N/A')} - {song.get('artist', 'N/A')}")
                    
                    # 检查时间信息
                    if 'time' in mc_data:
                        time_info = mc_data['time']
                        print(f"   ⏰ time字段: {len(time_info)}个时间点")
                        if time_info:
                            first_time = time_info[0]
                            print(f"      第一个时间点: {first_time}")
                    
                    # 检查notes
                    if 'note' in mc_data:
                        notes = mc_data['note']
                        print(f"   🎵 note字段: {len(notes)}个音符")
                        if notes:
                            # 显示前几个notes的格式
                            print(f"      前3个音符格式:")
                            for i, note in enumerate(notes[:3]):
                                print(f"        Note {i+1}: {note}")
                            
                            # 统计按键分布
                            column_counts = {}
                            for note in notes:
                                col = note.get('column', -1)
                                column_counts[col] = column_counts.get(col, 0) + 1
                            print(f"      按键分布: {column_counts}")
                    
                    # 检查extra字段
                    if 'extra' in mc_data:
                        extra = mc_data['extra']
                        print(f"   ⚙️  extra字段: {list(extra.keys())}")
            
            # 查找音频文件
            audio_files = [f for f in file_list if f.endswith(('.ogg', '.mp3', '.wav'))]
            print(f"\n🎵 音频文件 ({len(audio_files)}个):")
            for audio_file in audio_files:
                print(f"   🎶 {audio_file}")
                
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()

def compare_with_standard(fixed_mcz, standard_mcz):
    """对比修复后的文件与标准文件"""
    print(f"\n🔄 对比修复文件与标准文件")
    print(f"   修复文件: {fixed_mcz}")
    print(f"   标准文件: {standard_mcz}")
    
    try:
        # 读取两个文件的MC数据
        def read_mc_data(mcz_path):
            with zipfile.ZipFile(mcz_path, 'r') as mcz:
                mc_files = [f for f in mcz.namelist() if f.endswith('.mc')]
                if mc_files:
                    with mcz.open(mc_files[0], 'r') as f:
                        return json.loads(f.read().decode('utf-8'))
            return None
        
        fixed_data = read_mc_data(fixed_mcz)
        standard_data = read_mc_data(standard_mcz)
        
        if fixed_data and standard_data:
            print(f"\n📊 结构对比:")
            print(f"   修复文件顶级字段: {list(fixed_data.keys())}")
            print(f"   标准文件顶级字段: {list(standard_data.keys())}")
            
            # 对比meta字段
            if 'meta' in fixed_data and 'meta' in standard_data:
                fixed_meta = fixed_data['meta']
                standard_meta = standard_data['meta']
                print(f"\n🏷️  meta字段对比:")
                print(f"   修复文件meta: {list(fixed_meta.keys())}")
                print(f"   标准文件meta: {list(standard_meta.keys())}")
                
                # 关键字段对比
                key_fields = ['version', 'mode', 'mode_ext']
                for field in key_fields:
                    fixed_val = fixed_meta.get(field)
                    standard_val = standard_meta.get(field)
                    print(f"   {field}: 修复[{fixed_val}] vs 标准[{standard_val}]")
            
            # 对比notes格式
            if 'note' in fixed_data and 'note' in standard_data:
                fixed_notes = fixed_data['note']
                standard_notes = standard_data['note']
                print(f"\n🎵 Notes格式对比:")
                print(f"   修复文件notes数量: {len(fixed_notes)}")
                print(f"   标准文件notes数量: {len(standard_notes)}")
                
                if fixed_notes and standard_notes:
                    print(f"   修复文件第一个note: {fixed_notes[0]}")
                    print(f"   标准文件第一个note: {standard_notes[0]}")
                    
                    # 检查字段一致性
                    fixed_note_fields = set(fixed_notes[0].keys()) if fixed_notes else set()
                    standard_note_fields = set(standard_notes[0].keys()) if standard_notes else set()
                    print(f"   修复文件note字段: {fixed_note_fields}")
                    print(f"   标准文件note字段: {standard_note_fields}")
                    
                    if fixed_note_fields == standard_note_fields:
                        print(f"   ✅ Note字段格式一致")
                    else:
                        print(f"   ⚠️  Note字段格式差异:")
                        print(f"      缺少字段: {standard_note_fields - fixed_note_fields}")
                        print(f"      多余字段: {fixed_note_fields - standard_note_fields}")
        
    except Exception as e:
        print(f"❌ 对比失败: {e}")

def main():
    # 验证修复后的文件
    fixed_mcz = "generated_beatmaps/fixed_song_4833.mcz"
    check_fixed_mcz(fixed_mcz)
    
    # 与标准文件对比
    standard_mcz = "trainData/_song_1203.mcz"
    if os.path.exists(standard_mcz):
        compare_with_standard(fixed_mcz, standard_mcz)
    else:
        print(f"\n⚠️  标准文件不存在: {standard_mcz}")

if __name__ == "__main__":
    main()
