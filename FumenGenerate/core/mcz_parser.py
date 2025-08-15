#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCZ 文件解析器

MCZ 文件是一个包含音乐游戏谱面数据的压缩包，包含：
1. 音频文件 (.ogg)
2. 图片文件 (.jpg)
3. 谱面文件 (.mc - JSON格式，.tja - 太鼓达人格式)

该脚本用于解析和提取 MCZ 文件中的所有信息。
"""

import os
import json
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class SongInfo:
    """歌曲基本信息"""
    title: str
    artist: str
    song_id: int
    

@dataclass
class BeatmapMetadata:
    """谱面元数据"""
    creator: str
    version: str  # 难度版本 (Easy, Normal, Hard, Expert等)
    mode: int     # 游戏模式
    background: str
    song_info: SongInfo


@dataclass
class Note:
    """音符信息"""
    beat: List[int]      # 节拍位置 [小节, 拍子分子, 拍子分母]
    column: int          # 列位置
    endbeat: Optional[List[int]] = None  # 长按音符的结束节拍


@dataclass
class TimingPoint:
    """时间轴变化点"""
    beat: List[int]      # 节拍位置
    bpm: float           # BPM值


@dataclass
class MCBeatmap:
    """MC格式谱面数据"""
    metadata: BeatmapMetadata
    notes: List[Note]
    timing_points: List[TimingPoint]
    raw_data: Dict[str, Any]


@dataclass
class TJABeatmap:
    """TJA格式谱面数据"""
    title: str
    artist: str
    wave_file: str
    cover_file: str
    author: str
    course: int      # 难度等级
    level: int       # 星级
    bpm: float
    offset: float
    raw_content: str


@dataclass
class MCZFile:
    """MCZ文件完整数据结构"""
    audio_files: List[str]      # 音频文件列表
    image_files: List[str]      # 图片文件列表
    mc_beatmaps: List[MCBeatmap]    # MC格式谱面
    tja_beatmaps: List[TJABeatmap]  # TJA格式谱面
    file_path: str              # 原文件路径


class MCZParser:
    """MCZ文件解析器"""
    
    def __init__(self):
        self.temp_dir = None
    
    def parse_mcz_file(self, mcz_path: str) -> MCZFile:
        """
        解析MCZ文件
        
        Args:
            mcz_path: MCZ文件路径
            
        Returns:
            MCZFile: 解析后的数据结构
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            self.temp_dir = temp_dir
            
            # 解压MCZ文件
            with zipfile.ZipFile(mcz_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # 获取解压后的文件列表
            extracted_files = self._get_extracted_files(temp_dir)
            
            # 解析各类文件
            audio_files = self._filter_files(extracted_files, ['.ogg', '.mp3', '.wav'])
            image_files = self._filter_files(extracted_files, ['.jpg', '.png', '.jpeg'])
            mc_files = self._filter_files(extracted_files, ['.mc'])
            tja_files = self._filter_files(extracted_files, ['.tja'])
            
            # 解析MC格式谱面
            mc_beatmaps = []
            for mc_file in mc_files:
                try:
                    beatmap = self._parse_mc_file(mc_file)
                    if beatmap:
                        mc_beatmaps.append(beatmap)
                except Exception as e:
                    print(f"解析MC文件失败 {mc_file}: {e}")
            
            # 解析TJA格式谱面
            tja_beatmaps = []
            for tja_file in tja_files:
                try:
                    beatmap = self._parse_tja_file(tja_file)
                    if beatmap:
                        tja_beatmaps.append(beatmap)
                except Exception as e:
                    print(f"解析TJA文件失败 {tja_file}: {e}")
            
            return MCZFile(
                audio_files=audio_files,
                image_files=image_files,
                mc_beatmaps=mc_beatmaps,
                tja_beatmaps=tja_beatmaps,
                file_path=mcz_path
            )
    
    def _get_extracted_files(self, base_dir: str) -> List[str]:
        """获取解压后的所有文件路径"""
        files = []
        for root, dirs, filenames in os.walk(base_dir):
            for filename in filenames:
                files.append(os.path.join(root, filename))
        return files
    
    def _filter_files(self, files: List[str], extensions: List[str]) -> List[str]:
        """根据扩展名过滤文件"""
        return [f for f in files if any(f.lower().endswith(ext) for ext in extensions)]
    
    def _parse_mc_file(self, mc_file_path: str) -> Optional[MCBeatmap]:
        """解析MC格式谱面文件"""
        try:
            with open(mc_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 解析元数据
            meta = data.get('meta', {})
            song_info = SongInfo(
                title=meta.get('song', {}).get('title', ''),
                artist=meta.get('song', {}).get('artist', ''),
                song_id=meta.get('song', {}).get('id', 0)
            )
            
            metadata = BeatmapMetadata(
                creator=meta.get('creator', ''),
                version=meta.get('version', ''),
                mode=meta.get('mode', 0),
                background=meta.get('background', ''),
                song_info=song_info
            )
            
            # 解析音符
            notes = []
            for note_data in data.get('note', []):
                if 'beat' in note_data and 'column' in note_data:
                    note = Note(
                        beat=note_data['beat'],
                        column=note_data['column'],
                        endbeat=note_data.get('endbeat')
                    )
                    notes.append(note)
            
            # 解析时间轴
            timing_points = []
            for time_data in data.get('time', []):
                if 'beat' in time_data and 'bpm' in time_data:
                    timing_point = TimingPoint(
                        beat=time_data['beat'],
                        bpm=time_data['bpm']
                    )
                    timing_points.append(timing_point)
            
            return MCBeatmap(
                metadata=metadata,
                notes=notes,
                timing_points=timing_points,
                raw_data=data
            )
            
        except Exception as e:
            print(f"解析MC文件出错: {e}")
            return None
    
    def _parse_tja_file(self, tja_file_path: str) -> Optional[TJABeatmap]:
        """解析TJA格式谱面文件"""
        try:
            with open(tja_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取头部信息
            title = self._extract_tja_field(content, 'TITLE')
            artist = self._extract_tja_field(content, 'ARTIST')
            wave_file = self._extract_tja_field(content, 'WAVE')
            cover_file = self._extract_tja_field(content, 'COVER')
            author = self._extract_tja_field(content, 'AUTHOR')
            
            course = int(self._extract_tja_field(content, 'COURSE') or '0')
            level = int(self._extract_tja_field(content, 'LEVEL') or '0')
            bpm = float(self._extract_tja_field(content, 'BPM') or '0')
            offset = float(self._extract_tja_field(content, 'OFFSET') or '0')
            
            return TJABeatmap(
                title=title,
                artist=artist,
                wave_file=wave_file,
                cover_file=cover_file,
                author=author,
                course=course,
                level=level,
                bpm=bpm,
                offset=offset,
                raw_content=content
            )
            
        except Exception as e:
            print(f"解析TJA文件出错: {e}")
            return None
    
    def _extract_tja_field(self, content: str, field_name: str) -> str:
        """从TJA内容中提取字段值"""
        pattern = rf'^{field_name}:(.*)$'
        match = re.search(pattern, content, re.MULTILINE)
        return match.group(1).strip() if match else ''
    
    def print_mcz_info(self, mcz_data: MCZFile):
        """打印MCZ文件信息"""
        print(f"=== MCZ文件分析: {os.path.basename(mcz_data.file_path)} ===")
        print(f"音频文件数量: {len(mcz_data.audio_files)}")
        print(f"图片文件数量: {len(mcz_data.image_files)}")
        print(f"MC谱面数量: {len(mcz_data.mc_beatmaps)}")
        print(f"TJA谱面数量: {len(mcz_data.tja_beatmaps)}")
        
        if mcz_data.audio_files:
            print("\n音频文件:")
            for audio in mcz_data.audio_files:
                print(f"  - {os.path.basename(audio)}")
        
        if mcz_data.image_files:
            print("\n图片文件:")
            for image in mcz_data.image_files:
                print(f"  - {os.path.basename(image)}")
        
        if mcz_data.mc_beatmaps:
            print("\nMC谱面:")
            for i, beatmap in enumerate(mcz_data.mc_beatmaps):
                print(f"  {i+1}. {beatmap.metadata.song_info.title}")
                print(f"     艺术家: {beatmap.metadata.song_info.artist}")
                print(f"     制谱者: {beatmap.metadata.creator}")
                print(f"     难度: {beatmap.metadata.version}")
                print(f"     音符数量: {len(beatmap.notes)}")
                print(f"     时间轴变化点: {len(beatmap.timing_points)}")
        
        if mcz_data.tja_beatmaps:
            print("\nTJA谱面:")
            for i, beatmap in enumerate(mcz_data.tja_beatmaps):
                print(f"  {i+1}. {beatmap.title}")
                print(f"     艺术家: {beatmap.artist}")
                print(f"     制谱者: {beatmap.author}")
                print(f"     难度等级: {beatmap.course} (星级: {beatmap.level})")
                print(f"     BPM: {beatmap.bpm}")
                print(f"     偏移: {beatmap.offset}")


def main():
    """主函数 - 示例用法"""
    parser = MCZParser()
    
    # 示例：解析单个MCZ文件
    mcz_path = r"d:\Projects\FumenGenerate\trainData\_song_10088.mcz"
    
    if os.path.exists(mcz_path):
        print("正在解析MCZ文件...")
        mcz_data = parser.parse_mcz_file(mcz_path)
        parser.print_mcz_info(mcz_data)
    else:
        print(f"文件不存在: {mcz_path}")


if __name__ == "__main__":
    main()
