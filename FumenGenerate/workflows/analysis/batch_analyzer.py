#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
import sys
import os
# 修复工作区重组后的导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)


批量分析MCZ训练数据

该脚本用于批量分析trainData目录下的所有MCZ文件，
提取音乐特征、谱面特征和难度信息，为训练做准备。
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
from core.mcz_parser import MCZParser, MCZFile
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class TrainingDataAnalyzer:
    """训练数据分析器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.parser = MCZParser()
        self.analysis_results = []
    
    def analyze_all_mcz_files(self) -> List[Dict[str, Any]]:
        """分析所有MCZ文件"""
        mcz_files = [f for f in os.listdir(self.data_dir) if f.endswith('.mcz')]
        
        print(f"发现 {len(mcz_files)} 个MCZ文件，开始分析...")
        
        for mcz_file in tqdm(mcz_files, desc="分析MCZ文件"):
            mcz_path = os.path.join(self.data_dir, mcz_file)
            try:
                mcz_data = self.parser.parse_mcz_file(mcz_path)
                analysis = self._analyze_single_mcz(mcz_data)
                self.analysis_results.extend(analysis)
            except Exception as e:
                print(f"分析失败 {mcz_file}: {e}")
        
        return self.analysis_results
    
    def _analyze_single_mcz(self, mcz_data: MCZFile) -> List[Dict[str, Any]]:
        """分析单个MCZ文件"""
        results = []
        
        # 基本信息
        file_info = {
            'file_name': os.path.basename(mcz_data.file_path),
            'audio_files_count': len(mcz_data.audio_files),
            'image_files_count': len(mcz_data.image_files),
            'mc_beatmaps_count': len(mcz_data.mc_beatmaps),
            'tja_beatmaps_count': len(mcz_data.tja_beatmaps)
        }
        
        # 分析MC谱面
        for i, beatmap in enumerate(mcz_data.mc_beatmaps):
            result = file_info.copy()
            result.update({
                'beatmap_type': 'MC',
                'beatmap_index': i,
                'song_title': beatmap.metadata.song_info.title,
                'artist': beatmap.metadata.song_info.artist,
                'song_id': beatmap.metadata.song_info.song_id,
                'creator': beatmap.metadata.creator,
                'difficulty_version': beatmap.metadata.version,
                'game_mode': beatmap.metadata.mode,
                'note_count': len(beatmap.notes),
                'timing_points_count': len(beatmap.timing_points),
                'has_long_notes': any(note.endbeat is not None for note in beatmap.notes),
                'max_column': max((note.column for note in beatmap.notes), default=0),
                'min_column': min((note.column for note in beatmap.notes), default=0)
            })
            
            # 计算谱面统计信息
            if beatmap.notes:
                result.update(self._calculate_beatmap_stats(beatmap))
            
            # 提取BPM信息
            if beatmap.timing_points:
                bpms = [tp.bpm for tp in beatmap.timing_points]
                result.update({
                    'initial_bpm': bpms[0] if bpms else 0,
                    'max_bpm': max(bpms),
                    'min_bpm': min(bpms),
                    'avg_bpm': sum(bpms) / len(bpms),
                    'bpm_changes': len(bpms) - 1
                })
            
            results.append(result)
        
        # 分析TJA谱面
        for i, beatmap in enumerate(mcz_data.tja_beatmaps):
            result = file_info.copy()
            result.update({
                'beatmap_type': 'TJA',
                'beatmap_index': i,
                'song_title': beatmap.title,
                'artist': beatmap.artist,
                'song_id': None,  # TJA通常没有song_id
                'creator': beatmap.author,
                'difficulty_version': f"Course {beatmap.course}",
                'game_mode': None,  # TJA是太鼓达人模式
                'course_level': beatmap.course,
                'star_level': beatmap.level,
                'bpm': beatmap.bpm,
                'offset': beatmap.offset,
                'wave_file': beatmap.wave_file,
                'cover_file': beatmap.cover_file
            })
            
            results.append(result)
        
        return results
    
    def _calculate_beatmap_stats(self, beatmap) -> Dict[str, Any]:
        """计算谱面统计信息"""
        notes = beatmap.notes
        if not notes:
            return {}
        
        # 计算音符密度
        beats = [note.beat for note in notes]
        beat_positions = [beat[0] + beat[1]/beat[2] for beat in beats]  # 转换为小数形式的节拍位置
        
        if len(beat_positions) >= 2:
            total_duration = max(beat_positions) - min(beat_positions)
            note_density = len(notes) / total_duration if total_duration > 0 else 0
        else:
            note_density = 0
        
        # 计算列使用分布
        column_counts = {}
        for note in notes:
            col = note.column
            column_counts[col] = column_counts.get(col, 0) + 1
        
        # 计算长押音符比例
        long_notes_count = sum(1 for note in notes if note.endbeat is not None)
        long_notes_ratio = long_notes_count / len(notes) if notes else 0
        
        return {
            'note_density': note_density,
            'long_notes_count': long_notes_count,
            'long_notes_ratio': long_notes_ratio,
            'used_columns_count': len(column_counts),
            'column_distribution': json.dumps(column_counts),
            'first_note_beat': min(beat_positions) if beat_positions else 0,
            'last_note_beat': max(beat_positions) if beat_positions else 0,
            'beatmap_duration': max(beat_positions) - min(beat_positions) if len(beat_positions) >= 2 else 0
        }
    
    def save_analysis_results(self, output_file: str = "mcz_analysis_results.csv"):
        """保存分析结果到CSV文件"""
        if not self.analysis_results:
            print("没有分析结果可保存")
            return
        
        df = pd.DataFrame(self.analysis_results)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"分析结果已保存到: {output_file}")
        return df
    
    def generate_summary_report(self, df: pd.DataFrame = None):
        """生成分析总结报告"""
        if df is None:
            if not self.analysis_results:
                print("没有分析结果可生成报告")
                return
            df = pd.DataFrame(self.analysis_results)
        
        print("=== MCZ训练数据分析总结报告 ===")
        print(f"总文件数: {df['file_name'].nunique()}")
        print(f"总谱面数: {len(df)}")
        print(f"MC谱面数: {len(df[df['beatmap_type'] == 'MC'])}")
        print(f"TJA谱面数: {len(df[df['beatmap_type'] == 'TJA'])}")
        
        # MC谱面分析
        mc_df = df[df['beatmap_type'] == 'MC']
        if not mc_df.empty:
            print("\n=== MC谱面分析 ===")
            print(f"平均音符数: {mc_df['note_count'].mean():.1f}")
            print(f"音符数范围: {mc_df['note_count'].min()} - {mc_df['note_count'].max()}")
            print(f"难度版本分布:")
            for version, count in mc_df['difficulty_version'].value_counts().head(10).items():
                print(f"  {version}: {count}")
            
            print(f"\n游戏模式分布:")
            for mode, count in mc_df['game_mode'].value_counts().items():
                print(f"  模式 {mode}: {count}")
            
            if 'initial_bpm' in mc_df.columns:
                print(f"\nBPM统计:")
                print(f"  平均初始BPM: {mc_df['initial_bpm'].mean():.1f}")
                print(f"  BPM范围: {mc_df['min_bpm'].min():.1f} - {mc_df['max_bpm'].max():.1f}")
        
        # TJA谱面分析
        tja_df = df[df['beatmap_type'] == 'TJA']
        if not tja_df.empty:
            print("\n=== TJA谱面分析 ===")
            print(f"难度等级分布:")
            for course, count in tja_df['course_level'].value_counts().items():
                print(f"  Course {course}: {count}")
            
            print(f"星级范围: {tja_df['star_level'].min()} - {tja_df['star_level'].max()}")
            print(f"平均BPM: {tja_df['bpm'].mean():.1f}")
        
        # 歌曲统计
        print(f"\n=== 歌曲统计 ===")
        print(f"独特歌曲数: {df['song_title'].nunique()}")
        print(f"独特艺术家数: {df['artist'].nunique()}")
        
        top_artists = df['artist'].value_counts().head(5)
        print(f"音乐最多的艺术家:")
        for artist, count in top_artists.items():
            print(f"  {artist}: {count}")
    
    def create_visualizations(self, df: pd.DataFrame = None, save_plots: bool = True):
        """创建可视化图表"""
        if df is None:
            if not self.analysis_results:
                print("没有分析结果可可视化")
                return
            df = pd.DataFrame(self.analysis_results)
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('MCZ训练数据分析可视化', fontsize=16)
        
        # 1. 谱面类型分布
        df['beatmap_type'].value_counts().plot(kind='pie', ax=axes[0,0], autopct='%1.1f%%')
        axes[0,0].set_title('谱面类型分布')
        
        # 2. MC谱面音符数分布
        mc_df = df[df['beatmap_type'] == 'MC']
        if not mc_df.empty and 'note_count' in mc_df.columns:
            mc_df['note_count'].hist(bins=30, ax=axes[0,1])
            axes[0,1].set_title('MC谱面音符数分布')
            axes[0,1].set_xlabel('音符数量')
            axes[0,1].set_ylabel('频次')
        
        # 3. 游戏模式分布
        if not mc_df.empty and 'game_mode' in mc_df.columns:
            mc_df['game_mode'].value_counts().plot(kind='bar', ax=axes[0,2])
            axes[0,2].set_title('游戏模式分布')
            axes[0,2].set_xlabel('游戏模式')
            axes[0,2].set_ylabel('谱面数量')
        
        # 4. TJA难度等级分布
        tja_df = df[df['beatmap_type'] == 'TJA']
        if not tja_df.empty and 'course_level' in tja_df.columns:
            tja_df['course_level'].value_counts().sort_index().plot(kind='bar', ax=axes[1,0])
            axes[1,0].set_title('TJA难度等级分布')
            axes[1,0].set_xlabel('Course等级')
            axes[1,0].set_ylabel('谱面数量')
        
        # 5. BPM分布
        if not mc_df.empty and 'initial_bpm' in mc_df.columns:
            mc_df['initial_bpm'].hist(bins=30, ax=axes[1,1])
            axes[1,1].set_title('BPM分布')
            axes[1,1].set_xlabel('BPM')
            axes[1,1].set_ylabel('频次')
        
        # 6. 艺术家作品数量TOP10
        top_artists = df['artist'].value_counts().head(10)
        if len(top_artists) > 0:
            top_artists.plot(kind='barh', ax=axes[1,2])
            axes[1,2].set_title('艺术家作品数量TOP10')
            axes[1,2].set_xlabel('作品数量')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('mcz_analysis_visualization.png', dpi=300, bbox_inches='tight')
            print("可视化图表已保存为: mcz_analysis_visualization.png")
        
        plt.show()


def main():
    """主函数"""
    data_dir = r"d:\Projects\FumenGenerate\trainData"
    
    # 创建分析器
    analyzer = TrainingDataAnalyzer(data_dir)
    
    # 批量分析MCZ文件
    print("开始批量分析MCZ文件...")
    results = analyzer.analyze_all_mcz_files()
    
    # 保存结果
    df = analyzer.save_analysis_results()
    
    # 生成报告
    analyzer.generate_summary_report(df)
    
    # 创建可视化图表
    try:
        analyzer.create_visualizations(df)
    except Exception as e:
        print(f"创建可视化图表时出错: {e}")
        print("可能需要安装matplotlib和seaborn: pip install matplotlib seaborn")


if __name__ == "__main__":
    main()
