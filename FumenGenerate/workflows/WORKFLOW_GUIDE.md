# 🎮 FumenGenerate 核心工作流程指南

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
