#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工作区整理器
识别重要文件，清理重复文档，整理项目结构
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Set
import hashlib

class WorkspaceOrganizer:
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.important_files = set()
        self.duplicate_files = {}
        self.backup_files = []
        self.documentation_files = []
        self.analysis_report = {}
        
    def analyze_workspace(self):
        """分析工作区结构"""
        print("🔍 分析工作区结构...")
        
        # 重要的核心文件模式
        core_patterns = {
            # 核心训练脚本
            'enhanced_weighted_fusion_training_3_7.py': '最新增强权重融合训练脚本',
            'large_scale_real_training.py': '大规模真实数据训练',
            'weighted_fusion_large_scale_training_2_8.py': '权重融合大规模训练',
            
            # 核心模型和组件
            'models/weighted_fusion_model.py': '权重融合模型定义',
            'models/deep_learning_beatmap_system.py': '深度学习系统',
            'models/hybrid_beatmap_system.py': '混合模型系统',
            
            # 核心解析器
            'core/mcz_parser.py': 'MCZ文件解析器',
            'core/four_k_extractor.py': '4K谱面提取器',
            'core/audio_beatmap_analyzer.py': '音频谱面分析器',
            
            # 主程序和生成器
            'main.py': '主程序入口',
            'deep_beatmap_generator.py': '深度学习谱面生成器',
            'final_demo.py': '最终演示程序',
            
            # 重要配置和文档
            'README.md': '项目说明文档',
        }
        
        # 已训练的重要模型
        important_models = {
            'enhanced_weighted_fusion_model_3_7.pth': '增强权重融合模型 (3:7)',
            'best_weighted_fusion_model.pth': '最佳权重融合模型',
        }
        
        # 备份文件模式
        backup_patterns = [
            '_backup_',
            '.bak',
            '_old',
            '_copy'
        ]
        
        # 重复文档模式
        duplicate_doc_patterns = [
            '工作原理详解.md',
            '可视化演示.py',
            '原理简化演示.py'
        ]
        
        all_files = []
        for root, dirs, files in os.walk(self.workspace_path):
            for file in files:
                file_path = Path(root) / file
                relative_path = file_path.relative_to(self.workspace_path)
                all_files.append(relative_path)
        
        # 分类文件
        for file_path in all_files:
            file_str = str(file_path)
            
            # 检查是否是核心文件
            if file_str in core_patterns:
                self.important_files.add((file_path, core_patterns[file_str]))
            elif file_str in important_models:
                self.important_files.add((file_path, important_models[file_str]))
            
            # 检查备份文件
            if any(pattern in file_str for pattern in backup_patterns):
                self.backup_files.append(file_path)
            
            # 检查重复文档
            if any(pattern in file_str for pattern in duplicate_doc_patterns):
                self.documentation_files.append(file_path)
        
        # 查找重复文件（基于内容哈希）
        self._find_duplicates(all_files)
        
        print(f"✅ 分析完成:")
        print(f"   📁 总文件数: {len(all_files)}")
        print(f"   ⭐ 重要文件: {len(self.important_files)}")
        print(f"   📄 备份文件: {len(self.backup_files)}")
        print(f"   📝 重复文档: {len(self.documentation_files)}")
        print(f"   🔄 重复文件组: {len(self.duplicate_files)}")
    
    def _find_duplicates(self, all_files: List[Path]):
        """查找重复文件"""
        file_hashes = {}
        
        for file_path in all_files:
            full_path = self.workspace_path / file_path
            if full_path.is_file() and full_path.suffix in ['.py', '.md']:
                try:
                    content_hash = self._get_file_hash(full_path)
                    if content_hash in file_hashes:
                        # 发现重复文件
                        if content_hash not in self.duplicate_files:
                            self.duplicate_files[content_hash] = []
                        self.duplicate_files[content_hash].append(file_path)
                    else:
                        file_hashes[content_hash] = file_path
                except Exception as e:
                    print(f"⚠️ 无法读取文件 {file_path}: {e}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """计算文件内容哈希"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            content = f.read()
            hasher.update(content)
        return hasher.hexdigest()
    
    def create_cleanup_plan(self):
        """创建清理计划"""
        print("\n📋 创建清理计划...")
        
        cleanup_plan = {
            'keep_files': [],
            'backup_to_archive': [],
            'duplicate_docs_to_remove': [],
            'old_training_scripts_to_archive': [],
            'generated_data_to_clean': []
        }
        
        # 保留重要文件
        for file_path, description in self.important_files:
            cleanup_plan['keep_files'].append({
                'path': str(file_path),
                'description': description
            })
        
        # 备份文件归档
        for file_path in self.backup_files:
            cleanup_plan['backup_to_archive'].append(str(file_path))
        
        # 重复文档清理
        doc_groups = self._group_similar_docs()
        for group_name, files in doc_groups.items():
            if len(files) > 1:
                # 保留最新的，其他的标记删除
                sorted_files = sorted(files, key=lambda x: os.path.getmtime(self.workspace_path / x), reverse=True)
                for file_to_remove in sorted_files[1:]:
                    cleanup_plan['duplicate_docs_to_remove'].append(str(file_to_remove))
        
        # 旧训练脚本归档
        old_training_patterns = [
            'quick_weighted_fusion_training_2_8.py',
            'working_train.py',
            'quick_hybrid_train.py',
            'large_scale_train_with_preprocessed_backup_*.py'
        ]
        
        all_files = list(self.workspace_path.glob('*.py'))
        for file_path in all_files:
            relative_path = file_path.relative_to(self.workspace_path)
            if any(pattern.replace('*', '') in str(relative_path) for pattern in old_training_patterns):
                if relative_path not in [f[0] for f in self.important_files]:
                    cleanup_plan['old_training_scripts_to_archive'].append(str(relative_path))
        
        # 生成数据清理
        generated_patterns = [
            'temp_mcz_analysis/',
            'extracted_audio/',
            '__pycache__/',
            '*.png',
            '*.csv',
            'test_*.json'
        ]
        
        for pattern in generated_patterns:
            matches = list(self.workspace_path.glob(pattern))
            for match in matches:
                relative_path = match.relative_to(self.workspace_path)
                cleanup_plan['generated_data_to_clean'].append(str(relative_path))
        
        self.cleanup_plan = cleanup_plan
        return cleanup_plan
    
    def _group_similar_docs(self) -> Dict[str, List[Path]]:
        """将相似的文档分组"""
        groups = {
            'neural_network_docs': [],
            'random_forest_docs': [],
            'fusion_docs': [],
            'principle_docs': []
        }
        
        for file_path in self.documentation_files:
            file_str = str(file_path)
            if '神经网络' in file_str:
                groups['neural_network_docs'].append(file_path)
            elif '随机森林' in file_str:
                groups['random_forest_docs'].append(file_path)
            elif '权重融合' in file_str or '融合' in file_str:
                groups['fusion_docs'].append(file_path)
            elif '原理' in file_str or '详解' in file_str:
                groups['principle_docs'].append(file_path)
        
        return groups
    
    def execute_cleanup(self, dry_run: bool = True):
        """执行清理操作"""
        if not hasattr(self, 'cleanup_plan'):
            print("❌ 请先调用 create_cleanup_plan()")
            return
        
        print(f"\n🚀 {'模拟' if dry_run else '执行'}清理操作...")
        
        # 创建归档目录
        archive_dir = self.workspace_path / 'archived'
        if not dry_run:
            archive_dir.mkdir(exist_ok=True)
            (archive_dir / 'backups').mkdir(exist_ok=True)
            (archive_dir / 'old_scripts').mkdir(exist_ok=True)
            (archive_dir / 'generated_data').mkdir(exist_ok=True)
        
        actions = []
        
        # 归档备份文件
        for backup_file in self.cleanup_plan['backup_to_archive']:
            src = self.workspace_path / backup_file
            dst = archive_dir / 'backups' / backup_file
            action = f"归档备份: {backup_file} -> archived/backups/"
            actions.append(action)
            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
        
        # 删除重复文档
        for doc_file in self.cleanup_plan['duplicate_docs_to_remove']:
            action = f"删除重复文档: {doc_file}"
            actions.append(action)
            if not dry_run:
                (self.workspace_path / doc_file).unlink()
        
        # 归档旧训练脚本
        for old_script in self.cleanup_plan['old_training_scripts_to_archive']:
            src = self.workspace_path / old_script
            dst = archive_dir / 'old_scripts' / old_script
            action = f"归档旧脚本: {old_script} -> archived/old_scripts/"
            actions.append(action)
            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
        
        # 清理生成数据
        for generated_item in self.cleanup_plan['generated_data_to_clean']:
            path = self.workspace_path / generated_item
            if path.exists():
                if path.is_dir():
                    action = f"归档目录: {generated_item} -> archived/generated_data/"
                    if not dry_run:
                        dst = archive_dir / 'generated_data' / generated_item
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(path), str(dst))
                else:
                    action = f"删除生成文件: {generated_item}"
                    if not dry_run:
                        path.unlink()
                actions.append(action)
        
        print(f"📊 清理操作总结:")
        for action in actions:
            print(f"   • {action}")
        
        if dry_run:
            print(f"\n💡 这是模拟运行，没有实际修改文件")
            print(f"   要执行实际清理，请调用 execute_cleanup(dry_run=False)")
    
    def create_project_structure_doc(self):
        """创建项目结构文档"""
        print("\n📖 创建项目结构文档...")
        
        doc_content = """# FumenGenerate 项目结构

## 📁 核心文件

### 🚀 主要训练脚本
"""
        
        # 按类别组织重要文件
        categories = {
            '训练脚本': [],
            '核心模型': [],
            '解析器': [],
            '生成器': [],
            '配置文档': []
        }
        
        for file_path, description in self.important_files:
            file_str = str(file_path)
            if 'training' in file_str:
                categories['训练脚本'].append((file_path, description))
            elif file_str.startswith('models/'):
                categories['核心模型'].append((file_path, description))
            elif file_str.startswith('core/'):
                categories['解析器'].append((file_path, description))
            elif 'generator' in file_str or 'demo' in file_str:
                categories['生成器'].append((file_path, description))
            else:
                categories['配置文档'].append((file_path, description))
        
        for category, files in categories.items():
            if files:
                doc_content += f"\n### {category}\n"
                for file_path, description in files:
                    doc_content += f"- `{file_path}` - {description}\n"
        
        doc_content += """

## 🗂️ 目录结构

- `core/` - 核心功能模块
  - `mcz_parser.py` - MCZ文件解析
  - `four_k_extractor.py` - 4K谱面提取
  - `audio_beatmap_analyzer.py` - 音频分析
  
- `models/` - 机器学习模型
  - `weighted_fusion_model.py` - 权重融合模型
  - `deep_learning_beatmap_system.py` - 深度学习系统
  - `hybrid_beatmap_system.py` - 混合模型
  
- `trainData/` - 训练数据 (MCZ文件)
- `preprocessed_data/` - 预处理数据
- `generated_beatmaps/` - 生成的谱面
- `archived/` - 归档文件

## 🎯 推荐使用流程

1. **训练模型**: 使用 `enhanced_weighted_fusion_training_3_7.py`
2. **生成谱面**: 使用 `deep_beatmap_generator.py`
3. **演示效果**: 使用 `final_demo.py`

## 📝 版本说明

- **enhanced_weighted_fusion_training_3_7.py**: 最新版本，3:7权重比例，完整算法架构
- **权重融合**: RF分支30% + NN分支70%，自动学习最优权重
- **模型架构**: d_model=256, heads=8, layers=6，与完整版本对齐

"""
        
        doc_path = self.workspace_path / 'PROJECT_STRUCTURE.md'
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(doc_content)
        
        print(f"✅ 项目结构文档已创建: {doc_path}")
    
    def generate_report(self):
        """生成清理报告"""
        report = {
            'workspace_path': str(self.workspace_path),
            'analysis_date': '2025-08-09',
            'important_files': [
                {'path': str(path), 'description': desc} 
                for path, desc in self.important_files
            ],
            'cleanup_plan': self.cleanup_plan if hasattr(self, 'cleanup_plan') else {},
            'recommendations': [
                "保留enhanced_weighted_fusion_training_3_7.py作为主训练脚本",
                "归档所有备份文件到archived/目录",
                "删除重复的文档文件，保留最新版本",
                "清理生成的临时数据和缓存",
                "定期备份重要的训练好的模型文件"
            ]
        }
        
        report_path = self.workspace_path / 'workspace_cleanup_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📊 清理报告已生成: {report_path}")
        return report

def main():
    """主函数"""
    workspace_path = r"D:\Projects\FumenGenerate"
    
    print("🔧 FumenGenerate 工作区整理器")
    print("=" * 50)
    
    organizer = WorkspaceOrganizer(workspace_path)
    
    # 分析工作区
    organizer.analyze_workspace()
    
    # 创建清理计划
    organizer.create_cleanup_plan()
    
    # 模拟清理（先不实际执行）
    organizer.execute_cleanup(dry_run=True)
    
    # 创建项目结构文档
    organizer.create_project_structure_doc()
    
    # 生成报告
    organizer.generate_report()
    
    print("\n🎉 工作区分析完成！")
    print("📋 查看以下文件了解详情:")
    print("   • PROJECT_STRUCTURE.md - 项目结构说明")
    print("   • workspace_cleanup_report.json - 详细清理报告")
    print("\n💡 要执行实际清理，请运行:")
    print("   organizer.execute_cleanup(dry_run=False)")

if __name__ == "__main__":
    main()
