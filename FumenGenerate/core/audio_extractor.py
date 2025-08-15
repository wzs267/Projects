#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频文件提取器

从MCZ文件中提取音频文件，为音频-谱面对齐分析做准备
"""

import os
import zipfile
import tempfile
import shutil
from typing import List, Tuple, Optional


class AudioExtractor:
    """音频文件提取器"""
    
    def __init__(self, output_dir: str = "extracted_audio"):
        """
        初始化提取器
        
        Args:
            output_dir: 音频文件输出目录
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def extract_audio_from_mcz(self, mcz_path: str) -> List[str]:
        """
        从MCZ文件中提取音频文件
        
        Args:
            mcz_path: MCZ文件路径
            
        Returns:
            List[str]: 提取的音频文件路径列表
        """
        extracted_files = []
        mcz_name = os.path.splitext(os.path.basename(mcz_path))[0]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 解压MCZ文件
            with zipfile.ZipFile(mcz_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # 查找音频文件
            audio_extensions = ['.ogg', '.mp3', '.wav', '.flac']
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in audio_extensions):
                        src_path = os.path.join(root, file)
                        
                        # 创建唯一的文件名
                        file_name, file_ext = os.path.splitext(file)
                        unique_name = f"{mcz_name}_{file_name}{file_ext}"
                        dst_path = os.path.join(self.output_dir, unique_name)
                        
                        # 复制文件
                        shutil.copy2(src_path, dst_path)
                        extracted_files.append(dst_path)
                        print(f"提取音频文件: {unique_name}")
        
        return extracted_files
    
    def extract_audio_from_directory(self, mcz_dir: str, max_files: Optional[int] = None) -> List[Tuple[str, List[str]]]:
        """
        从目录中的所有MCZ文件提取音频
        
        Args:
            mcz_dir: MCZ文件目录
            max_files: 最大处理文件数（None表示处理所有）
            
        Returns:
            List[Tuple[str, List[str]]]: (MCZ文件名, 音频文件列表) 的列表
        """
        mcz_files = [f for f in os.listdir(mcz_dir) if f.endswith('.mcz')]
        if max_files:
            mcz_files = mcz_files[:max_files]
        
        results = []
        
        for mcz_file in mcz_files:
            mcz_path = os.path.join(mcz_dir, mcz_file)
            print(f"\n处理: {mcz_file}")
            
            try:
                audio_files = self.extract_audio_from_mcz(mcz_path)
                results.append((mcz_file, audio_files))
            except Exception as e:
                print(f"提取失败 {mcz_file}: {e}")
        
        return results


def main():
    """主函数 - 提取音频文件"""
    mcz_dir = r"d:\Projects\FumenGenerate\trainData"
    
    # 创建音频提取器
    extractor = AudioExtractor()
    
    # 只提取前5个MCZ文件的音频（测试用）
    print("开始提取音频文件...")
    results = extractor.extract_audio_from_directory(mcz_dir, max_files=5)
    
    print(f"\n=== 提取完成 ===")
    total_audio_files = 0
    for mcz_file, audio_files in results:
        print(f"{mcz_file}: {len(audio_files)} 个音频文件")
        total_audio_files += len(audio_files)
    
    print(f"总计提取了 {total_audio_files} 个音频文件到 {extractor.output_dir} 目录")


if __name__ == "__main__":
    main()
