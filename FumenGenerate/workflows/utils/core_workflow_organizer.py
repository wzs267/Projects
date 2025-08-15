#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心工作流程整理器
将完整的预处理、训练、生成脚本分离出来，避免与分析脚本混杂
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict

class CoreWorkflowOrganizer:
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        
        # 核心工作流程分类
        self.core_workflows = {
            'preprocessing': {
                'description': '数据预处理流程',
                'files': [
                    'batch_mcz_preprocessor.py',
                    'data_processor.py',
                ]
            },
            'training': {
                'description': '完整训练流程',
                'files': [
                    'enhanced_weighted_fusion_training_3_7.py',  # 最新的权重融合训练
                    'large_scale_real_training.py',              # 大规模真实数据训练
                    'weighted_fusion_large_scale_training_2_8.py',  # 权重融合大规模训练
                    'large_scale_train_with_preprocessed.py',    # 预处理数据训练
                    'large_scale_training.py',                   # 基础大规模训练
                    'large_scale_optimized_training.py',         # 优化的大规模训练
                ]
            },
            'generation': {
                'description': '谱面生成和演示',
                'files': [
                    'deep_beatmap_generator.py',                 # 深度学习谱面生成器
                    'final_demo.py',                            # 最终演示程序
                    'improved_precise_generator.py',             # 改进的精确生成器
                    'high_density_beatmap_generator.py',         # 高密度谱面生成
                    'precise_beatmap_generator.py',              # 精确谱面生成器
                ]
            },
            'main_entry': {
                'description': '主程序入口',
                'files': [
                    'main.py',                                  # 主程序
                ]
            }
        }
        
        # 需要保留但归档的分析脚本
        self.analysis_scripts = [
            'analyze_meta_structure.py',
            'analyze_note_timing.py', 
            'analyze_note_types.py',
            'analyze_original_mcz.py',
            'audio_feature_analysis.py',
            'batch_analyzer.py',
            'check_audio_control.py',
            'check_generated_mcz.py',
            'check_generated_notes.py',
            'check_offset_issue.py',
            'compare_mcz_formats.py',
            'compare_note_density.py',
            'correct_beat_analysis.py',
            'correct_time_calc.py',
            'debug_audio_extraction.py',
            'debug_features.py',
            'debug_mcz_processing.py',
            'debug_mcz_structure.py',
            'debug_note_generation.py',
            'deep_analysis.py',
            'detailed_feature_analysis.py',
            'detailed_note_check.py',
            'diagnose_audio.py',
            'environment_summary.py',
            'validate_fixed_mcz.py',
            'verify_high_density.py',
            'final_verification.py',
        ]
        
        # 测试和演示脚本
        self.test_demo_scripts = [
            'quick_demo.py',
            'quick_test.py',
            'quick_verify_mcz.py',
            'simple_feature_demo.py',
            'parameter_combination_demo.py',
            'test_4k_extractor.py',
            'test_deep_learning.py',
            'test_imports.py',
            'test_single_preprocess.py',
        ]
        
        # 工具和修复脚本
        self.utility_scripts = [
            'fix_imports.py',
            'workspace_organizer.py',
            'generate_beatmap_for_new_song.py',
        ]
    
    def create_workflow_directories(self):
        """创建工作流程目录结构"""
        print("📁 创建核心工作流程目录结构...")
        
        # 创建主要目录
        workflows_dir = self.workspace_path / 'workflows'
        workflows_dir.mkdir(exist_ok=True)
        
        # 创建各个工作流程子目录
        for workflow_name, workflow_info in self.core_workflows.items():
            workflow_dir = workflows_dir / workflow_name
            workflow_dir.mkdir(exist_ok=True)
            print(f"   📂 {workflow_name}/ - {workflow_info['description']}")
        
        # 创建分析脚本目录
        analysis_dir = workflows_dir / 'analysis'
        analysis_dir.mkdir(exist_ok=True)
        print(f"   📂 analysis/ - 数据分析和调试脚本")
        
        # 创建测试演示目录
        testing_dir = workflows_dir / 'testing'
        testing_dir.mkdir(exist_ok=True)
        print(f"   📂 testing/ - 测试和演示脚本")
        
        # 创建工具目录
        utils_dir = workflows_dir / 'utils'
        utils_dir.mkdir(exist_ok=True)
        print(f"   📂 utils/ - 工具和修复脚本")
        
        return workflows_dir
    
    def move_core_workflows(self, workflows_dir: Path, dry_run: bool = True):
        """移动核心工作流程脚本"""
        print(f"\n🚀 {'模拟' if dry_run else '执行'}移动核心工作流程脚本...")
        
        moved_files = []
        
        # 移动核心工作流程文件
        for workflow_name, workflow_info in self.core_workflows.items():
            workflow_dir = workflows_dir / workflow_name
            
            print(f"\n📋 {workflow_name} - {workflow_info['description']}")
            for filename in workflow_info['files']:
                src_path = self.workspace_path / filename
                dst_path = workflow_dir / filename
                
                if src_path.exists():
                    action = f"移动: {filename} -> workflows/{workflow_name}/"
                    print(f"   • {action}")
                    moved_files.append(action)
                    
                    if not dry_run:
                        shutil.move(str(src_path), str(dst_path))
                else:
                    print(f"   ⚠️ 文件不存在: {filename}")
        
        return moved_files
    
    def move_analysis_scripts(self, workflows_dir: Path, dry_run: bool = True):
        """移动分析脚本"""
        print(f"\n📊 {'模拟' if dry_run else '执行'}移动分析脚本...")
        
        analysis_dir = workflows_dir / 'analysis'
        moved_files = []
        
        for filename in self.analysis_scripts:
            src_path = self.workspace_path / filename
            dst_path = analysis_dir / filename
            
            if src_path.exists():
                action = f"移动: {filename} -> workflows/analysis/"
                print(f"   • {action}")
                moved_files.append(action)
                
                if not dry_run:
                    shutil.move(str(src_path), str(dst_path))
        
        return moved_files
    
    def move_test_demo_scripts(self, workflows_dir: Path, dry_run: bool = True):
        """移动测试和演示脚本"""
        print(f"\n🧪 {'模拟' if dry_run else '执行'}移动测试演示脚本...")
        
        testing_dir = workflows_dir / 'testing'
        moved_files = []
        
        for filename in self.test_demo_scripts:
            src_path = self.workspace_path / filename
            dst_path = testing_dir / filename
            
            if src_path.exists():
                action = f"移动: {filename} -> workflows/testing/"
                print(f"   • {action}")
                moved_files.append(action)
                
                if not dry_run:
                    shutil.move(str(src_path), str(dst_path))
        
        return moved_files
    
    def move_utility_scripts(self, workflows_dir: Path, dry_run: bool = True):
        """移动工具脚本"""
        print(f"\n🔧 {'模拟' if dry_run else '执行'}移动工具脚本...")
        
        utils_dir = workflows_dir / 'utils'
        moved_files = []
        
        for filename in self.utility_scripts:
            src_path = self.workspace_path / filename
            dst_path = utils_dir / filename
            
            if src_path.exists():
                action = f"移动: {filename} -> workflows/utils/"
                print(f"   • {action}")
                moved_files.append(action)
                
                if not dry_run:
                    shutil.move(str(src_path), str(dst_path))
        
        return moved_files
    
    def create_workflow_guide(self, workflows_dir: Path):
        """创建工作流程使用指南"""
        print(f"\n📖 创建工作流程使用指南...")
        
        guide_content = """# 🎮 FumenGenerate 核心工作流程指南

## 📁 目录结构

### 1. workflows/preprocessing/ - 数据预处理
- `batch_mcz_preprocessor.py` - 批量MCZ文件预处理
- `data_processor.py` - 核心数据处理器

### 2. workflows/training/ - 模型训练
- ⭐ `enhanced_weighted_fusion_training_3_7.py` - **推荐**: 最新权重融合训练 (RF:NN=3:7)
- `large_scale_real_training.py` - 大规模真实数据训练
- `weighted_fusion_large_scale_training_2_8.py` - 权重融合大规模训练 (RF:NN=2:8)
- `large_scale_train_with_preprocessed.py` - 使用预处理数据的大规模训练
- `large_scale_training.py` - 基础大规模深度学习训练
- `large_scale_optimized_training.py` - 优化版大规模训练

### 3. workflows/generation/ - 谱面生成
- ⭐ `deep_beatmap_generator.py` - **推荐**: 深度学习谱面生成器
- ⭐ `final_demo.py` - **推荐**: 完整系统演示
- `improved_precise_generator.py` - 改进的精确生成器
- `high_density_beatmap_generator.py` - 高密度谱面生成
- `precise_beatmap_generator.py` - 精确谱面生成器

### 4. workflows/main_entry/ - 主程序
- `main.py` - 系统主入口

### 5. workflows/analysis/ - 分析调试
包含各种数据分析、结构分析、调试脚本

### 6. workflows/testing/ - 测试演示
包含快速测试、演示脚本

### 7. workflows/utils/ - 工具脚本
包含修复工具、项目整理工具

## 🎯 推荐工作流程

### 标准训练流程
```bash
# 1. 训练最新权重融合模型
cd workflows/training
python enhanced_weighted_fusion_training_3_7.py

# 2. 生成谱面
cd ../generation  
python deep_beatmap_generator.py

# 3. 完整演示
python final_demo.py
```

### 大规模训练流程
```bash
# 1. 预处理数据（可选）
cd workflows/preprocessing
python batch_mcz_preprocessor.py

# 2. 大规模真实数据训练
cd ../training
python large_scale_real_training.py

# 3. 生成和验证
cd ../generation
python final_demo.py
```

### 快速测试流程  
```bash
# 快速测试和演示
cd workflows/testing
python quick_demo.py
python quick_test.py
```

## 📊 模型对比

| 训练脚本 | 模型类型 | 权重比例 | 推荐场景 |
|----------|----------|----------|----------|
| enhanced_weighted_fusion_training_3_7.py | 权重融合 | RF:NN=3:7 | **最新推荐** |
| weighted_fusion_large_scale_training_2_8.py | 权重融合 | RF:NN=2:8 | 神经网络主导 |
| large_scale_real_training.py | 混合模型 | 固定架构 | 大规模真实数据 |
| large_scale_training.py | 深度学习 | 纯神经网络 | 传统深度学习 |

## 🎵 生成器对比

| 生成器脚本 | 特点 | 推荐场景 |
|------------|------|----------|
| deep_beatmap_generator.py | 深度学习，高质量 | **主要推荐** |
| final_demo.py | 完整演示，易用 | **演示推荐** |
| improved_precise_generator.py | 精确控制 | 高精度需求 |
| high_density_beatmap_generator.py | 高密度谱面 | 困难模式 |

## ⚡ 快速开始

1. **首次使用**: 运行 `workflows/training/enhanced_weighted_fusion_training_3_7.py` 训练模型
2. **生成谱面**: 运行 `workflows/generation/final_demo.py` 生成和测试谱面
3. **自定义**: 根据需要调整各训练脚本的参数

## 🔧 故障排除

- 如遇到导入错误，运行 `workflows/utils/fix_imports.py`
- 需要分析数据时，使用 `workflows/analysis/` 中的脚本
- 快速测试使用 `workflows/testing/` 中的脚本

## 📝 更新记录

- **enhanced_weighted_fusion_training_3_7.py**: 最新版本，支持完整算法架构，3:7权重融合
- **权重融合技术**: RF分支提供决策支持(30%)，NN分支负责序列学习(70%)
- **模型架构**: d_model=256, heads=8, layers=6，与完整版本对齐
"""
        
        guide_path = workflows_dir / 'WORKFLOW_GUIDE.md'
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        print(f"✅ 工作流程指南已创建: {guide_path}")
    
    def create_quick_launchers(self, workflows_dir: Path):
        """创建快速启动脚本"""
        print(f"\n🚀 创建快速启动脚本...")
        
        # 训练启动器
        train_launcher = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
快速训练启动器
\"\"\"

import os
import sys

def main():
    print("🎮 FumenGenerate 快速训练启动器")
    print("=" * 40)
    print("1. enhanced_weighted_fusion_training_3_7.py - 最新权重融合训练 ⭐推荐")
    print("2. large_scale_real_training.py - 大规模真实数据训练") 
    print("3. weighted_fusion_large_scale_training_2_8.py - 2:8权重融合训练")
    print("4. large_scale_training.py - 基础深度学习训练")
    
    choice = input("\\n请选择训练方案 (1-4): ").strip()
    
    training_scripts = {
        '1': 'enhanced_weighted_fusion_training_3_7.py',
        '2': 'large_scale_real_training.py', 
        '3': 'weighted_fusion_large_scale_training_2_8.py',
        '4': 'large_scale_training.py'
    }
    
    if choice in training_scripts:
        script = training_scripts[choice]
        print(f"\\n🚀 启动训练: {script}")
        os.system(f"python {script}")
    else:
        print("❌ 无效选择")

if __name__ == "__main__":
    main()
"""
        
        # 生成启动器
        gen_launcher = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
快速生成启动器
\"\"\"

import os
import sys

def main():
    print("🎵 FumenGenerate 快速生成启动器")
    print("=" * 40)
    print("1. final_demo.py - 完整系统演示 ⭐推荐")
    print("2. deep_beatmap_generator.py - 深度学习生成器")
    print("3. improved_precise_generator.py - 精确生成器")
    print("4. high_density_beatmap_generator.py - 高密度生成器")
    
    choice = input("\\n请选择生成方案 (1-4): ").strip()
    
    generation_scripts = {
        '1': 'final_demo.py',
        '2': 'deep_beatmap_generator.py',
        '3': 'improved_precise_generator.py', 
        '4': 'high_density_beatmap_generator.py'
    }
    
    if choice in generation_scripts:
        script = generation_scripts[choice]
        print(f"\\n🎵 启动生成: {script}")
        os.system(f"python {script}")
    else:
        print("❌ 无效选择")

if __name__ == "__main__":
    main()
"""
        
        # 保存启动器
        train_launcher_path = workflows_dir / 'quick_train.py'
        with open(train_launcher_path, 'w', encoding='utf-8') as f:
            f.write(train_launcher)
        
        gen_launcher_path = workflows_dir / 'quick_generate.py' 
        with open(gen_launcher_path, 'w', encoding='utf-8') as f:
            f.write(gen_launcher)
        
        print(f"✅ 快速启动脚本已创建:")
        print(f"   • {train_launcher_path}")
        print(f"   • {gen_launcher_path}")

def main():
    """主函数"""
    workspace_path = r"D:\Projects\FumenGenerate"
    
    print("🔧 FumenGenerate 核心工作流程整理器")
    print("=" * 50)
    
    organizer = CoreWorkflowOrganizer(workspace_path)
    
    # 1. 创建目录结构
    workflows_dir = organizer.create_workflow_directories()
    
    # 2. 模拟移动操作
    print(f"\n📋 模拟移动操作预览:")
    organizer.move_core_workflows(workflows_dir, dry_run=True)
    organizer.move_analysis_scripts(workflows_dir, dry_run=True)
    organizer.move_test_demo_scripts(workflows_dir, dry_run=True)
    organizer.move_utility_scripts(workflows_dir, dry_run=True)
    
    # 3. 询问是否执行
    print(f"\\n💡 这是模拟运行，查看操作预览")
    confirm = input("是否执行实际移动操作? (y/N): ").strip().lower()
    
    if confirm == 'y':
        print(f"\\n🚀 执行实际移动操作...")
        organizer.move_core_workflows(workflows_dir, dry_run=False)
        organizer.move_analysis_scripts(workflows_dir, dry_run=False)
        organizer.move_test_demo_scripts(workflows_dir, dry_run=False) 
        organizer.move_utility_scripts(workflows_dir, dry_run=False)
        
        # 4. 创建指南和启动器
        organizer.create_workflow_guide(workflows_dir)
        organizer.create_quick_launchers(workflows_dir)
        
        print(f"\\n🎉 核心工作流程整理完成！")
        print(f"📁 新的目录结构:")
        print(f"   workflows/preprocessing/ - 数据预处理")
        print(f"   workflows/training/ - 模型训练") 
        print(f"   workflows/generation/ - 谱面生成")
        print(f"   workflows/analysis/ - 分析调试")
        print(f"   workflows/testing/ - 测试演示")
        print(f"   workflows/utils/ - 工具脚本")
        print(f"\\n📖 查看以下文件了解详情:")
        print(f"   • workflows/WORKFLOW_GUIDE.md - 详细使用指南")
        print(f"   • workflows/quick_train.py - 快速训练启动器")
        print(f"   • workflows/quick_generate.py - 快速生成启动器")
    else:
        print(f"\\n❌ 取消移动操作")

if __name__ == "__main__":
    main()
