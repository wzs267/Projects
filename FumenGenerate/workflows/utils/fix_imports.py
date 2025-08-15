#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
import sys
import os
# 修复工作区重组后的导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)


修复所有Python文件的导入问题
整理工作区后的导入路径修复工具
"""

import os
import re
from typing import List, Dict

class ImportFixer:
    """导入修复器"""
    
    def __init__(self):
        self.base_dir = "d:/Projects/FumenGenerate"
        
        # 定义文件的新位置映射
        self.file_locations = {
            # 核心模块在 core/ 目录
            'mcz_parser': 'core.mcz_parser',
            'four_k_extractor': 'core.four_k_extractor', 
            'audio_beatmap_analyzer': 'core.audio_beatmap_analyzer',
            'audio_extractor': 'core.audio_extractor',
            'data_processor': 'core.data_processor',
            
            # 学习系统在根目录（需要移动到scripts）
            'beatmap_learning_system': 'scripts.beatmap_learning_system',
            'deep_learning_beatmap_system': 'scripts.deep_learning_beatmap_system',
            'hybrid_beatmap_system': 'scripts.hybrid_beatmap_system',
        }
        
        # 需要修复的文件列表
        self.files_to_fix = [
            'large_scale_real_training.py',
            'scripts/quick_hybrid_train.py', 
            'scripts/test_deep_learning.py',
            'scripts/final_demo.py',
            'scripts/large_scale_training.py',
            'batch_analyzer.py',
            'quick_demo.py',
            'quick_test.py',
        ]
    
    def find_missing_files(self):
        """查找缺失的文件"""
        print("🔍 查找需要移动的文件...")
        
        # 查找学习系统文件
        learning_files = [
            'beatmap_learning_system.py',
            'deep_learning_beatmap_system.py', 
            'hybrid_beatmap_system.py'
        ]
        
        for file in learning_files:
            if os.path.exists(file):
                print(f"📁 发现文件: {file} -> 需要移动到 scripts/")
            else:
                print(f"❌ 文件不存在: {file}")
    
    def move_files_to_scripts(self):
        """移动学习系统文件到scripts目录"""
        os.makedirs('scripts', exist_ok=True)
        
        learning_files = [
            'beatmap_learning_system.py',
            'deep_learning_beatmap_system.py', 
            'hybrid_beatmap_system.py'
        ]
        
        for file in learning_files:
            if os.path.exists(file):
                import shutil
                shutil.move(file, f'scripts/{file}')
                print(f"📦 移动: {file} -> scripts/{file}")
    
    def fix_imports_in_file(self, file_path: str):
        """修复单个文件的导入"""
        if not os.path.exists(file_path):
            print(f"⚠️  文件不存在: {file_path}")
            return
        
        print(f"🔧 修复导入: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 修复各种导入模式
            for old_import, new_import in self.file_locations.items():
                # 修复 from xxx import 
                pattern1 = f"from {old_import} import"
                replacement1 = f"from {new_import} import"
                content = re.sub(pattern1, replacement1, content)
                
                # 修复 import xxx
                pattern2 = f"import {old_import}"
                replacement2 = f"import {new_import}"
                content = re.sub(pattern2, replacement2, content)
                
                # 修复 from xxx.yyy import
                pattern3 = f"from {old_import}\\."
                replacement3 = f"from {new_import}."
                content = re.sub(pattern3, replacement3, content)
            
            # 如果内容有变化，写回文件
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"  ✅ 修复完成")
            else:
                print(f"  ℹ️  无需修复")
                
        except Exception as e:
            print(f"  ❌ 修复失败: {e}")
    
    def fix_all_imports(self):
        """修复所有文件的导入"""
        print("🚀 开始修复所有导入问题...")
        
        # 首先移动文件
        self.move_files_to_scripts()
        
        # 修复每个文件的导入
        for file_path in self.files_to_fix:
            self.fix_imports_in_file(file_path)
        
        print("✅ 导入修复完成！")
    
    def create_working_training_script(self):
        """创建一个能正常工作的训练脚本"""
        script_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工作目录修复后的快速训练脚本
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 添加路径以便导入
sys.path.append('.')
sys.path.append('core')
sys.path.append('scripts')

def load_existing_data():
    """加载现有的训练数据"""
    try:
        from scripts.beatmap_learning_system import BeatmapLearningSystem
        
        print("📥 加载现有训练数据...")
        system = BeatmapLearningSystem()
        aligned_datasets = system.collect_training_data('test_4k_beatmaps.json', 'extracted_audio')
        
        if not aligned_datasets:
            print("❌ 无法加载数据")
            return None, None
        
        X, y_note, y_column, y_long = system.prepare_machine_learning_data(aligned_datasets)
        return X, y_note
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None, None

class QuickHybridNet(nn.Module):
    """快速混合网络"""
    
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x).squeeze()

def quick_train():
    """快速训练"""
    print("🚀 快速混合训练")
    print("=" * 40)
    
    # 加载数据
    X, y = load_existing_data()
    if X is None:
        print("❌ 无法获取训练数据")
        return
    
    print(f"✅ 数据加载完成: {len(X):,} 样本")
    
    # 训练RF
    print("🌲 训练随机森林...")
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X, y)
    rf_probs = rf.predict_proba(X)[:, 1]
    
    # 构建增强特征
    enhanced_X = np.column_stack([X, rf_probs, np.gradient(rf_probs)])
    
    # 标准化
    scaler = StandardScaler()
    enhanced_X = scaler.fit_transform(enhanced_X)
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        enhanced_X, y, test_size=0.2, random_state=42
    )
    
    # 训练神经网络
    print("🧠 训练神经网络...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = QuickHybridNet(enhanced_X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # 转换数据
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # 训练
    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                test_pred = (test_outputs > 0.5).float()
                accuracy = (test_pred == y_test_tensor).float().mean()
                print(f"Epoch {epoch}: 损失={loss:.4f}, 准确率={accuracy:.3f}")
    
    # 最终测试
    model.eval()
    with torch.no_grad():
        final_outputs = model(X_test_tensor)
        final_pred = (final_outputs > 0.5).float()
        final_accuracy = (final_pred == y_test_tensor).float().mean()
        
        rf_acc = rf.score(X_test[:, :X.shape[1]], y_test)
    
    print(f"\\n🎉 训练完成！")
    print(f"🌲 随机森林准确率: {rf_acc:.3f}")
    print(f"🧠 混合模型准确率: {final_accuracy:.3f}")
    print(f"📈 提升: {final_accuracy - rf_acc:.3f}")
    
    # 保存模型
    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': final_accuracy.item()
    }, 'models/working_hybrid_model.pth')
    
    import pickle
    with open('models/working_components.pkl', 'wb') as f:
        pickle.dump({'rf': rf, 'scaler': scaler}, f)
    
    print("💾 模型已保存到 models/")

if __name__ == "__main__":
    quick_train()
'''
        
        with open('working_train.py', 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        print("📝 创建了 working_train.py - 修复导入问题的训练脚本")

def main():
    """主函数"""
    print("🔧 Python工作区导入修复工具")
    print("=" * 50)
    
    fixer = ImportFixer()
    
    # 查找文件
    fixer.find_missing_files()
    
    # 修复导入
    fixer.fix_all_imports()
    
    # 创建工作脚本
    fixer.create_working_training_script()
    
    print("\\n✅ 修复完成！现在可以运行:")
    print("   python working_train.py")

if __name__ == "__main__":
    main()
