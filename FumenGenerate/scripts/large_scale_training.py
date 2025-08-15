#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大规模深度学习训练脚本
利用700+个MCZ文件进行完整的深度学习训练
"""

import os
import sys
import time
import gc
from scripts.deep_learning_beatmap_system import DeepBeatmapLearningSystem

def large_scale_training():
    """大规模深度学习训练"""
    print("🎮 大规模深度学习音游谱面生成训练")
    print("=" * 60)
    print("📂 准备处理700+个MCZ文件...")
    
    # 创建高性能系统实例
    system = DeepBeatmapLearningSystem(
        sequence_length=64,    # 较长的序列以捕获音乐模式
        batch_size=64,         # 较大的批次以提高训练效率
        learning_rate=0.0005   # 较小的学习率以稳定训练
    )
    
    start_time = time.time()
    
    try:
        # 阶段1: 数据加载和预处理
        print("\n🔍 阶段1: 加载大规模数据集...")
        print("⚠️  这可能需要较长时间，请耐心等待...")
        
        # 检查trainData目录
        traindata_dir = 'trainData'
        if not os.path.exists(traindata_dir):
            print(f"❌ 找不到训练数据目录: {traindata_dir}")
            return
        
        mcz_files = [f for f in os.listdir(traindata_dir) if f.endswith('.mcz')]
        print(f"📊 发现 {len(mcz_files)} 个MCZ文件")
        
        if len(mcz_files) == 0:
            print("❌ 没有找到MCZ文件")
            return
        
        # 分批处理以避免内存溢出
        batch_size = 50  # 每次处理50个MCZ文件
        total_audio_features = []
        total_beatmap_labels = []
        
        for batch_start in range(0, min(200, len(mcz_files)), batch_size):  # 先处理200个文件
            batch_end = min(batch_start + batch_size, len(mcz_files))
            batch_files = mcz_files[batch_start:batch_end]
            
            print(f"\n📦 处理批次 [{batch_start//batch_size + 1}]: 文件 {batch_start+1}-{batch_end}")
            
            try:
                # 临时处理这一批文件
                temp_system = DeepBeatmapLearningSystem(sequence_length=32, batch_size=16)
                audio_features, beatmap_labels = temp_system.load_batch_dataset(
                    traindata_dir, batch_files
                )
                
                if audio_features is not None and beatmap_labels is not None:
                    total_audio_features.append(audio_features)
                    total_beatmap_labels.append(beatmap_labels)
                    print(f"✅ 批次完成: {audio_features.shape[0]:,} 个样本")
                else:
                    print("⚠️  批次处理失败")
                
                # 清理内存
                del temp_system, audio_features, beatmap_labels
                gc.collect()
                
            except Exception as e:
                print(f"❌ 批次处理出错: {e}")
                continue
        
        # 合并所有数据
        if not total_audio_features:
            print("❌ 没有成功处理任何数据")
            return
        
        print("\n🔗 合并所有批次数据...")
        import numpy as np
        final_audio_features = np.vstack(total_audio_features)
        final_beatmap_labels = np.vstack(total_beatmap_labels)
        
        print(f"📊 最终数据集大小: {final_audio_features.shape[0]:,} 个样本")
        print(f"🎵 特征维度: {final_audio_features.shape[1]}")
        print(f"🎮 标签维度: {final_beatmap_labels.shape[1]}")
        
        # 清理临时数据
        del total_audio_features, total_beatmap_labels
        gc.collect()
        
        # 阶段2: 模型创建和训练准备
        print("\n🏗️  阶段2: 创建深度学习模型...")
        system.create_model(input_dim=final_audio_features.shape[1])
        
        print("\n🔧 阶段3: 准备训练数据...")
        train_loader, val_loader = system.prepare_training_data(
            final_audio_features, final_beatmap_labels, train_ratio=0.85
        )
        
        # 清理大数组
        del final_audio_features, final_beatmap_labels
        gc.collect()
        
        # 阶段4: 深度学习训练
        print("\n🚀 阶段4: 开始大规模深度学习训练...")
        print("📋 训练配置:")
        print(f"   • 序列长度: {system.sequence_length}")
        print(f"   • 批次大小: {system.batch_size}")
        print(f"   • 学习率: {system.learning_rate}")
        print(f"   • 模型参数: {sum(p.numel() for p in system.model.parameters()):,}")
        print(f"   • 训练批次: {len(train_loader)}")
        print(f"   • 验证批次: {len(val_loader)}")
        
        # 开始训练
        system.train(
            train_loader, val_loader,
            num_epochs=100,  # 100轮训练
            save_path='large_scale_beatmap_model.pth'
        )
        
        # 阶段5: 结果分析和保存
        print("\n📊 阶段5: 分析训练结果...")
        system.plot_training_history()
        
        # 保存训练历史
        import json
        history_data = {
            'training_config': {
                'sequence_length': system.sequence_length,
                'batch_size': system.batch_size,
                'learning_rate': system.learning_rate,
                'model_parameters': sum(p.numel() for p in system.model.parameters()),
                'training_batches': len(train_loader),
                'validation_batches': len(val_loader)
            },
            'training_history': system.training_history,
            'final_performance': {
                'final_train_loss': system.training_history['train_loss'][-1],
                'final_val_loss': system.training_history['val_loss'][-1],
                'final_note_accuracy': system.training_history['note_accuracy'][-1],
                'final_event_accuracy': system.training_history['event_accuracy'][-1],
                'best_val_loss': min(system.training_history['val_loss'])
            }
        }
        
        with open('large_scale_training_results.json', 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        
        # 计算训练时间
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        print("\n🎉 大规模深度学习训练完成！")
        print("=" * 60)
        print("📈 最终结果:")
        print(f"   ⏱️  总训练时间: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"   📊 最终验证损失: {system.training_history['val_loss'][-1]:.4f}")
        print(f"   🎯 最佳验证损失: {min(system.training_history['val_loss']):.4f}")
        print(f"   🎵 音符准确率: {system.training_history['note_accuracy'][-1]:.3f}")
        print(f"   🎼 事件准确率: {system.training_history['event_accuracy'][-1]:.3f}")
        print("\n💾 保存文件:")
        print("   • large_scale_beatmap_model.pth - 最佳模型")
        print("   • large_scale_training_results.json - 训练历史")
        print("   • deep_learning_training_history.png - 训练图表")
        
        return system
        
    except Exception as e:
        print(f"\n❌ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_batch_loading_method():
    """为DeepBeatmapLearningSystem添加批量加载方法"""
    
    # 先读取原文件内容
    with open('deep_learning_beatmap_system.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 添加批量加载方法
    batch_method = '''
    def load_batch_dataset(self, traindata_dir: str, mcz_files: List[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        分批加载数据集以避免内存溢出
        
        Args:
            traindata_dir: 训练数据目录
            mcz_files: MCZ文件列表
            
        Returns:
            (audio_features, beatmap_labels): 音频特征和谱面标签
        """
        print(f"🔄 批量处理 {len(mcz_files)} 个MCZ文件...")
        
        # 导入必要的模块
        try:
            from core.mcz_parser import MCZParser
            from core.four_k_extractor import FourKBeatmapExtractor
            from core.audio_beatmap_analyzer import AudioBeatmapAnalyzer
        except ImportError as e:
            print(f"❌ 导入模块失败: {e}")
            return None, None
        
        parser = MCZParser()
        extractor = FourKBeatmapExtractor()
        analyzer = AudioBeatmapAnalyzer(time_resolution=0.05)
        
        all_audio_features = []
        all_beatmap_labels = []
        processed_count = 0
        
        for i, mcz_file in enumerate(mcz_files):
            try:
                mcz_path = os.path.join(traindata_dir, mcz_file)
                if not os.path.exists(mcz_path):
                    continue
                
                print(f"   📁 [{i+1}/{len(mcz_files)}]: {mcz_file[:50]}...")
                
                # 解析MCZ文件
                song_data = parser.parse_mcz_file(mcz_path)
                if not song_data:
                    continue
                
                # 提取4K谱面
                beatmaps_4k = extractor.extract_4k_beatmaps(song_data)
                if not beatmaps_4k:
                    continue
                
                # 提取音频文件
                temp_audio_dir = f"temp_audio_{i}"
                os.makedirs(temp_audio_dir, exist_ok=True)
                
                try:
                    extracted_audio = parser.extract_audio_files(mcz_path, temp_audio_dir)
                    if not extracted_audio:
                        continue
                    
                    # 处理第一个4K谱面和第一个音频文件
                    beatmap = beatmaps_4k[0]
                    audio_file = extracted_audio[0]
                    
                    # 分析音频和谱面
                    aligned_data = analyzer.align_audio_and_beatmap(
                        audio_file, beatmap, {}
                    )
                    
                    if aligned_data and len(aligned_data.audio_features) > self.sequence_length:
                        all_audio_features.append(aligned_data.audio_features)
                        all_beatmap_labels.append(aligned_data.beatmap_events)
                        processed_count += 1
                        
                        # 限制每批次的样本数量
                        if processed_count >= 20:  # 每批次最多20个谱面
                            break
                
                finally:
                    # 清理临时文件
                    import shutil
                    if os.path.exists(temp_audio_dir):
                        shutil.rmtree(temp_audio_dir)
                        
            except Exception as e:
                print(f"     ⚠️ 处理失败: {e}")
                continue
        
        if processed_count == 0:
            print("❌ 该批次没有成功处理任何数据")
            return None, None
        
        # 合并数据
        try:
            audio_features = np.vstack(all_audio_features)
            beatmap_labels = np.vstack(all_beatmap_labels)
            print(f"✅ 批次完成: {processed_count} 个谱面, {audio_features.shape[0]:,} 个样本")
            return audio_features, beatmap_labels
        except Exception as e:
            print(f"❌ 数据合并失败: {e}")
            return None, None
'''
    
    # 找到类定义的结束位置并插入新方法
    class_end_pos = content.rfind('def plot_training_history(self):')
    if class_end_pos != -1:
        insert_pos = content.rfind('\n', 0, class_end_pos)
        new_content = content[:insert_pos] + batch_method + content[insert_pos:]
        
        # 写回文件
        with open('deep_learning_beatmap_system.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ 已添加批量加载方法到深度学习系统")
    else:
        print("❌ 无法找到插入位置")


if __name__ == "__main__":
    print("🔧 准备大规模训练环境...")
    
    # 添加批量加载方法
    create_batch_loading_method()
    
    # 开始大规模训练
    trained_system = large_scale_training()
    
    if trained_system:
        print("\n🎊 大规模深度学习训练成功完成！")
        print("🚀 系统现在可以基于700+谱面的学习结果生成高质量谱面！")
    else:
        print("\n💥 训练过程中遇到问题，请检查日志信息")
