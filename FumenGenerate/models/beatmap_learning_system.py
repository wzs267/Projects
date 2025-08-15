#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的音频-谱面学习分析器

实现音游核心学习逻辑：
1. 音频特征提取（分贝变化、节拍检测）
2. 谱面事件对齐
3. 音频-谱面关联性学习
4. 难度参数影响分析
"""

import os
import json
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
from core.audio_beatmap_analyzer import AudioBeatmapAnalyzer, AlignedData
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class BeatmapLearningSystem:
    """谱面学习系统"""
    
    def __init__(self):
        self.analyzer = AudioBeatmapAnalyzer(time_resolution=0.05)  # 50ms分辨率
        self.note_placement_model = None
        self.column_selection_model = None
        self.long_note_model = None
        self.training_data = []
    
    def process_single_beatmap(self, mcz_name: str, beatmap_data: Dict[str, Any], 
                             audio_file: str) -> Optional[AlignedData]:
        """
        处理单个谱面的音频-谱面对齐
        
        Args:
            mcz_name: MCZ文件名
            beatmap_data: 谱面数据
            audio_file: 音频文件路径
            
        Returns:
            AlignedData: 对齐后的数据
        """
        if not os.path.exists(audio_file):
            print(f"音频文件不存在: {audio_file}")
            return None
        
        try:
            print(f"分析: {beatmap_data['song_title']} - {beatmap_data['difficulty_version']}")
            
            # 提取音频特征
            audio_features = self.analyzer.extract_audio_features(audio_file)
            
            # 提取谱面事件
            beatmap_events = self.analyzer.extract_beatmap_events(beatmap_data)
            
            # 构建难度参数
            difficulty_params = {
                'note_count': beatmap_data.get('note_count', 0),
                'note_density': beatmap_data.get('note_density', 0),
                'long_notes_ratio': beatmap_data.get('long_notes_ratio', 0),
                'avg_bpm': beatmap_data.get('avg_bpm', 120),
                'duration': beatmap_data.get('duration', 0)
            }
            
            # 对齐数据
            aligned_data = self.analyzer.align_audio_beatmap(
                audio_features, beatmap_events, difficulty_params
            )
            
            return aligned_data
            
        except Exception as e:
            print(f"处理失败: {e}")
            return None
    
    def collect_training_data(self, beatmaps_json: str, audio_dir: str) -> List[AlignedData]:
        """
        收集训练数据
        
        Args:
            beatmaps_json: 谱面JSON文件路径
            audio_dir: 音频文件目录
            
        Returns:
            List[AlignedData]: 训练数据列表
        """
        # 加载谱面数据
        with open(beatmaps_json, 'r', encoding='utf-8') as f:
            beatmaps_data = json.load(f)
        
        aligned_datasets = []
        
        for beatmap in beatmaps_data:
            # 查找对应的音频文件
            mcz_name = beatmap.get('source_mcz_file', '').replace('.mcz', '')
            audio_files = beatmap.get('audio_files', [])
            
            # 尝试找到匹配的音频文件
            matched_audio = None
            for audio_name in audio_files:
                # 尝试不同的命名模式
                possible_names = [
                    f"{mcz_name}_{audio_name}",
                    f"{mcz_name}_{os.path.splitext(audio_name)[0]}.ogg"
                ]
                
                for possible_name in possible_names:
                    audio_path = os.path.join(audio_dir, possible_name)
                    if os.path.exists(audio_path):
                        matched_audio = audio_path
                        break
                
                if matched_audio:
                    break
            
            if matched_audio:
                aligned_data = self.process_single_beatmap(mcz_name, beatmap, matched_audio)
                if aligned_data:
                    aligned_datasets.append(aligned_data)
                    print(f"✓ 成功处理: {beatmap['song_title']}")
            else:
                print(f"✗ 未找到音频: {beatmap['song_title']} ({audio_files})")
        
        self.training_data = aligned_datasets
        print(f"\n收集到 {len(aligned_datasets)} 个有效的训练样本")
        return aligned_datasets
    
    def prepare_machine_learning_data(self, aligned_datasets: List[AlignedData]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        为机器学习准备数据
        
        Args:
            aligned_datasets: 对齐后的数据集
            
        Returns:
            (X, y_note_placement, y_column_selection): 特征矩阵和标签
        """
        X_all = []  # 音频特征
        y_note_placement = []  # 是否放置音符 (0/1)
        y_column_selection = []  # 选择哪个轨道 (0-3 或 -1表示无音符)
        y_long_note = []  # 是否是长条音符 (0/1)
        
        for aligned_data in aligned_datasets:
            audio_features = aligned_data.audio_features
            beatmap_events = aligned_data.beatmap_events
            difficulty = aligned_data.difficulty_params
            
            # 为每个时间步准备特征
            for i in range(len(audio_features)):
                # 构建特征向量
                feature_vector = list(audio_features[i])  # 音频特征
                
                # 添加难度参数作为上下文特征
                feature_vector.extend([
                    difficulty.get('note_density', 0),
                    difficulty.get('long_notes_ratio', 0),
                    difficulty.get('avg_bpm', 120) / 200.0,  # 标准化BPM
                ])
                
                X_all.append(feature_vector)
                
                # 标签：是否有音符
                has_note = np.sum(beatmap_events[i, :4]) > 0
                y_note_placement.append(1 if has_note else 0)
                
                # 标签：选择的轨道（如果有多个，选择第一个）
                active_columns = np.where(beatmap_events[i, :4] > 0)[0]
                if len(active_columns) > 0:
                    y_column_selection.append(active_columns[0])
                else:
                    y_column_selection.append(-1)  # 无音符
                
                # 标签：是否是长条音符
                is_long_note = beatmap_events[i, 5] > 0  # long_start事件
                y_long_note.append(1 if is_long_note else 0)
        
        return (np.array(X_all), 
                np.array(y_note_placement), 
                np.array(y_column_selection),
                np.array(y_long_note))
    
    def train_models(self, X: np.ndarray, y_note: np.ndarray, 
                    y_column: np.ndarray, y_long: np.ndarray):
        """
        训练机器学习模型
        
        Args:
            X: 特征矩阵
            y_note: 音符放置标签
            y_column: 轨道选择标签
            y_long: 长条音符标签
        """
        print("训练机器学习模型...")
        
        # 分割数据
        X_train, X_test, y_note_train, y_note_test = train_test_split(
            X, y_note, test_size=0.2, random_state=42
        )
        _, _, y_col_train, y_col_test = train_test_split(
            X, y_column, test_size=0.2, random_state=42
        )
        _, _, y_long_train, y_long_test = train_test_split(
            X, y_long, test_size=0.2, random_state=42
        )
        
        # 1. 音符放置模型（分类：是否放置音符）
        print("训练音符放置模型...")
        self.note_placement_model = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced'
        )
        self.note_placement_model.fit(X_train, y_note_train)
        
        note_pred = self.note_placement_model.predict(X_test)
        print("音符放置模型性能:")
        print(classification_report(y_note_test, note_pred))
        
        # 2. 轨道选择模型（只对有音符的位置训练）
        print("训练轨道选择模型...")
        has_note_mask = y_col_train >= 0
        if np.sum(has_note_mask) > 0:
            self.column_selection_model = RandomForestClassifier(
                n_estimators=100, random_state=42
            )
            self.column_selection_model.fit(
                X_train[has_note_mask], y_col_train[has_note_mask]
            )
            
            # 测试
            test_has_note_mask = y_col_test >= 0
            if np.sum(test_has_note_mask) > 0:
                col_pred = self.column_selection_model.predict(X_test[test_has_note_mask])
                print("轨道选择模型性能:")
                print(classification_report(y_col_test[test_has_note_mask], col_pred))
        
        # 3. 长条音符模型
        print("训练长条音符模型...")
        self.long_note_model = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced'
        )
        self.long_note_model.fit(X_train, y_long_train)
        
        long_pred = self.long_note_model.predict(X_test)
        print("长条音符模型性能:")
        print(classification_report(y_long_test, long_pred))
        
        # 特征重要性分析
        self.analyze_feature_importance(X)
    
    def analyze_feature_importance(self, X: np.ndarray):
        """分析特征重要性"""
        if self.note_placement_model is None:
            return
        
        # 特征名称
        feature_names = [
            'RMS能量(dB)', '频谱质心', '过零率', '音符起始强度',
            'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5',
            '色度平均', '节拍邻近', '起始邻近',
            '音符密度', '长条比例', 'BPM标准化'
        ]
        
        # 获取特征重要性
        importance = self.note_placement_model.feature_importances_
        
        # 创建重要性图表
        plt.figure(figsize=(12, 8))
        indices = np.argsort(importance)[::-1]
        
        plt.title('音符放置模型 - 特征重要性')
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n特征重要性排序:")
        for i, idx in enumerate(indices[:10]):
            print(f"{i+1:2d}. {feature_names[idx]}: {importance[idx]:.4f}")
    
    def generate_beatmap_analysis(self, audio_file: str, difficulty_level: str = "Normal") -> Dict[str, Any]:
        """
        基于训练的模型分析音频并生成谱面建议
        
        Args:
            audio_file: 音频文件路径
            difficulty_level: 难度等级
            
        Returns:
            谱面分析结果
        """
        if not all([self.note_placement_model, self.column_selection_model, self.long_note_model]):
            print("模型尚未训练完成")
            return {}
        
        print(f"分析音频文件: {os.path.basename(audio_file)}")
        
        # 提取音频特征
        audio_features = self.analyzer.extract_audio_features(audio_file)
        
        # 设置难度参数
        difficulty_mapping = {
            "Easy": {"note_density": 0.5, "long_notes_ratio": 0.1, "bpm_factor": 1.0},
            "Normal": {"note_density": 1.0, "long_notes_ratio": 0.2, "bpm_factor": 1.0},
            "Hard": {"note_density": 1.5, "long_notes_ratio": 0.3, "bpm_factor": 1.0},
            "Expert": {"note_density": 2.0, "long_notes_ratio": 0.4, "bpm_factor": 1.0}
        }
        
        difficulty_params = difficulty_mapping.get(difficulty_level, difficulty_mapping["Normal"])
        
        # 准备特征矩阵
        X_predict = []
        for i in range(len(audio_features.time_frames)):
            # 确保索引不越界
            if i >= len(audio_features.rms_energy):
                break
                
            feature_vector = [
                float(audio_features.rms_energy[i]),
                float(audio_features.spectral_centroid[i]),
                float(audio_features.zero_crossing_rate[i]),
                float(np.interp(audio_features.time_frames[i], audio_features.time_frames, 
                               audio_features.onset_strength)),
            ]
            
            # 添加MFCC特征（确保固定长度）
            for j in range(5):  # 固定5个MFCC特征
                if j < audio_features.mfcc.shape[0] and i < audio_features.mfcc.shape[1]:
                    feature_vector.append(float(audio_features.mfcc[j, i]))
                else:
                    feature_vector.append(0.0)  # 填充零值
            
            # 添加色度特征
            if i < audio_features.chroma.shape[1]:
                chroma_mean = float(np.mean(audio_features.chroma[:, i]))
                feature_vector.append(chroma_mean)
            else:
                feature_vector.append(0.0)
            
            # 节拍和起始邻近性
            t = float(audio_features.time_frames[i])
            beat_proximity = min([abs(t - float(bt)) for bt in audio_features.beat_times], default=float('inf'))
            is_near_beat = 1.0 if beat_proximity < 0.1 else 0.0
            feature_vector.append(float(is_near_beat))
            
            onset_proximity = min([abs(t - float(ot)) for ot in audio_features.onset_times], default=float('inf'))
            is_near_onset = 1.0 if onset_proximity < 0.1 else 0.0
            feature_vector.append(float(is_near_onset))
            
            # 难度参数
            feature_vector.extend([
                float(difficulty_params['note_density']),
                float(difficulty_params['long_notes_ratio']),
                float(audio_features.tempo / 200.0)
            ])
            
            X_predict.append(feature_vector)
        
        X_predict = np.array(X_predict)
        
        # 预测
        note_probabilities = self.note_placement_model.predict_proba(X_predict)[:, 1]  # 放置音符的概率
        column_predictions = self.column_selection_model.predict(X_predict)
        long_note_probabilities = self.long_note_model.predict_proba(X_predict)[:, 1]
        
        # 生成建议的谱面事件
        suggested_events = []
        for i, t in enumerate(audio_features.time_frames):
            if note_probabilities[i] > 0.5:  # 超过50%概率放置音符
                event_type = 'long_start' if long_note_probabilities[i] > 0.3 else 'note'
                suggested_events.append({
                    'time': t,
                    'type': event_type,
                    'column': column_predictions[i],
                    'confidence': note_probabilities[i]
                })
        
        return {
            'audio_duration': audio_features.duration,
            'detected_tempo': audio_features.tempo,
            'suggested_events': suggested_events,
            'difficulty_level': difficulty_level,
            'analysis_params': difficulty_params
        }
    
    def visualize_learning_results(self, aligned_datasets: List[AlignedData]):
        """可视化学习结果"""
        if not aligned_datasets:
            return
        
        # 分析所有数据集的统计信息
        all_correlations = []
        all_difficulties = []
        
        for aligned_data in aligned_datasets:
            analysis = self.analyzer.analyze_beatmap_patterns(aligned_data)
            all_correlations.append(analysis['correlations'])
            all_difficulties.append(aligned_data.difficulty_params)
        
        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('音频-谱面学习分析结果', fontsize=16)
        
        # 1. RMS能量与音符的相关性
        rms_corrs = [corr['rms_energy_note'] for corr in all_correlations if not np.isnan(corr['rms_energy_note'])]
        axes[0, 0].hist(rms_corrs, bins=10, alpha=0.7)
        axes[0, 0].set_title('RMS能量-音符相关性分布')
        axes[0, 0].set_xlabel('相关系数')
        axes[0, 0].set_ylabel('频次')
        
        # 2. 音符密度与难度的关系
        note_densities = [diff['note_density'] for diff in all_difficulties]
        avg_bpms = [diff['avg_bpm'] for diff in all_difficulties]
        axes[0, 1].scatter(avg_bpms, note_densities, alpha=0.6)
        axes[0, 1].set_title('BPM vs 音符密度')
        axes[0, 1].set_xlabel('平均BPM')
        axes[0, 1].set_ylabel('音符密度')
        
        # 3. 长条音符比例分布
        long_note_ratios = [diff['long_notes_ratio'] for diff in all_difficulties]
        axes[1, 0].hist(long_note_ratios, bins=10, alpha=0.7)
        axes[1, 0].set_title('长条音符比例分布')
        axes[1, 0].set_xlabel('长条音符比例')
        axes[1, 0].set_ylabel('频次')
        
        # 4. 音符数量与时长的关系
        note_counts = [diff['note_count'] for diff in all_difficulties]
        durations = [diff['duration'] for diff in all_difficulties]
        axes[1, 1].scatter(durations, note_counts, alpha=0.6)
        axes[1, 1].set_title('时长 vs 音符数量')
        axes[1, 1].set_xlabel('时长(节拍)')
        axes[1, 1].set_ylabel('音符数量')
        
        plt.tight_layout()
        plt.savefig('learning_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """主函数 - 完整的学习流程演示"""
    print("=== 音游谱面生成 - 机器学习分析系统 ===")
    
    # 创建学习系统
    learning_system = BeatmapLearningSystem()
    
    # 收集训练数据
    beatmaps_json = "test_4k_beatmaps.json"
    audio_dir = "extracted_audio"
    
    print("第一步：收集训练数据...")
    aligned_datasets = learning_system.collect_training_data(beatmaps_json, audio_dir)
    
    if len(aligned_datasets) == 0:
        print("没有有效的训练数据，请检查音频文件")
        return
    
    # 准备机器学习数据
    print("\n第二步：准备机器学习数据...")
    X, y_note, y_column, y_long = learning_system.prepare_machine_learning_data(aligned_datasets)
    print(f"特征矩阵形状: {X.shape}")
    print(f"音符标签分布: {np.bincount(y_note)}")
    
    # 训练模型
    print("\n第三步：训练机器学习模型...")
    learning_system.train_models(X, y_note, y_column, y_long)
    
    # 可视化学习结果
    print("\n第四步：可视化学习结果...")
    learning_system.visualize_learning_results(aligned_datasets)
    
    # 演示谱面生成
    print("\n第五步：演示谱面生成...")
    if aligned_datasets:
        # 使用第一个音频文件进行演示
        audio_files = os.listdir(audio_dir)
        if audio_files:
            test_audio = os.path.join(audio_dir, audio_files[0])
            
            for difficulty in ["Easy", "Normal", "Hard"]:
                print(f"\n分析难度: {difficulty}")
                result = learning_system.generate_beatmap_analysis(test_audio, difficulty)
                
                if result:
                    print(f"  检测到的BPM: {result['detected_tempo']:.1f}")
                    print(f"  建议的音符数量: {len(result['suggested_events'])}")
                    print(f"  音频时长: {result['audio_duration']:.1f}秒")
    
    print("\n=== 学习分析完成 ===")
    print("系统已学会:")
    print("1. 根据音频RMS能量变化识别击打时机")
    print("2. 根据音频持续特征识别长条放置")
    print("3. 根据难度参数调节音符密度")
    print("4. 多轨道音符分配策略")


if __name__ == "__main__":
    main()
