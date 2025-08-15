# 🎵 FumenGenerate - AI音游谱面生成系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个基于深度学习的音乐游戏谱面自动生成系统，专门用于生成4轨道(4K)节奏游戏谱面。

## ✨ 核心功能

- 🤖 **AI谱面生成**：基于Transformer + 随机森林的混合架构
- 🎵 **音频特征提取**：MFCC、频谱质心、RMS能量等12维音频特征
- 🎮 **标准格式支持**：生成标准MC格式(.mcz)谱面文件
- ⚡ **快速训练**：支持混合精度训练，3-10倍加速
- 🎯 **智能难度调节**：自动调整音符密度和难度等级

## 🏗️ 项目架构

```
FumenGenerate/
├── 📂 core/                    # 核心模块
│   ├── mcz_parser.py          # MCZ文件解析器
│   ├── audio_extractor.py     # 音频提取器
│   └── four_k_extractor.py    # 4K谱面提取器
├── 📂 models/                  # AI模型
│   ├── improved_sequence_transformer.py  # 主力模型
│   └── deep_learning_beatmap_system.py   # 训练系统
├── 📂 workflows/               # 工作流程
│   ├── 🔧 training/           # 模型训练
│   ├── 🎮 generation/         # 谱面生成
│   └── 📊 preprocessing/      # 数据预处理
└── 📂 trainData/              # 训练数据(.mcz文件)
```

## 🚀 快速开始

### 1. 环境安装
```bash
pip install torch librosa numpy pandas scikit-learn
```

### 2. 快速生成谱面
```bash
# 使用预训练模型生成谱面
python workflows/quick_generate.py

# 或直接使用高密度生成器
python workflows/generation/high_density_beatmap_generator.py
```

### 3. 模型训练
```bash
# 快速训练（推荐）
python workflows/quick_train.py

# 完整训练
python workflows/training/improved_sequence_fusion_training_3_7.py
```

## 🎯 模型架构

### ImprovedWeightedFusionTransformer
- **输入维度**：12维音频特征
- **序列长度**：64步 (3.2秒音频历史)
- **架构**：Transformer(6层) + RandomForest(32棵树)
- **融合权重**：NN(70%) + RF(30%)
- **参数量**：5.29M

### 训练配置
```python
model = ImprovedWeightedFusionTransformer(
    input_dim=12,           # 12维音频特征
    d_model=256,            # Transformer隐藏维度
    num_heads=8,            # 8头自注意力
    num_layers=6,           # 6层深度
    sequence_length=64,     # 64步序列
    batch_size=64,          # 批次大小
    learning_rate=0.001     # 学习率
)
```

## 📊 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 训练数据 | 2,839个谱面 | 来自758个MCZ文件 |
| 音符总数 | 240万+ | 平衡分布在4个轨道 |
| 生成速度 | ~3秒/谱面 | 90秒音频生成时间 |
| 模型大小 | 20MB | 5.29M参数 |
| 训练时间 | 30分钟 | 50个epoch（快速配置） |

## 🎮 生成样例

```python
# 生成示例
generator = FixedBeatmapGenerator()
generator.generate_beatmap(
    input_mcz="trainData/_song_10088.mcz",
    output_mcz="generated_beatmaps/ai_generated.mcz",
    difficulty=15,
    target_keys=4
)
```

**输出结果**：
- 📁 `ai_generated.mcz` - 标准MC格式谱面
- 🎵 包含原音频文件
- 🎯 AI生成的4K谱面数据
- 📊 平均密度：3-5个音符/秒

## 🔧 高级配置

### 降低音符密度
```python
# 在 high_density_beatmap_generator.py 中调整
threshold = 0.002           # 提高阈值减少音符
density_multiplier = 0.5    # 降低密度倍数
subdivisions = 6           # 减少节拍细分
```

### 加速训练
```python
# 使用混合精度训练
use_amp = True
batch_size = 64        # 增大批次
sequence_length = 32   # 减少序列长度
```

## 📈 最新更新

- ✅ **v1.0** - 修复beat计算错误，重新预处理数据
- ✅ **v1.1** - 优化模型架构，提升生成质量
- ✅ **v1.2** - 添加密度控制，支持难度调节
- ✅ **v1.3** - 集成混合精度训练，3倍加速

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

*🎵 让AI为你的音乐创作完美的谱面！*
