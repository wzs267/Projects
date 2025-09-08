# 细胞异常检测使用指南

## 项目概述
基于DAE_RED_TF2设计思路，针对你的32x32x3 RGB细胞tiles实现的无监督异常检测系统。

## 核心特点
- **无监督学习**：不需要预先标注异常样本
- **去噪自编码器**：学习正常细胞的特征表示
- **重建误差**：通过重建误差识别异常细胞
- **批处理**：支持大规模数据集（100万张tiles）
- **伪影过滤**：自动过滤染色伪影和图像质量问题

## 快速开始

### 1. 训练模型
```bash
# 使用10,000个样本训练模型（推荐用于快速测试）
python cell_anomaly_detection.py --do_train --tiles_dir tiles_output --sample_size 10000 --epochs 50

# 使用更多样本训练（更好的效果）
python cell_anomaly_detection.py --do_train --tiles_dir tiles_output --sample_size 50000 --epochs 100
```

### 2. 异常检测
```bash
# 对所有tiles进行异常检测
python cell_anomaly_detection.py --do_detect --tiles_dir tiles_output --model_path cell_dae_model --output_dir anomaly_results
```

### 3. 一键完整流程
```bash
# 训练+检测
python cell_anomaly_detection.py --do_train --do_detect --tiles_dir tiles_output --sample_size 20000 --epochs 50
```

## 参数说明

### 主要参数
- `--tiles_dir`: tiles目录路径（默认：tiles_output）
- `--do_train`: 是否训练模型
- `--do_detect`: 是否进行异常检测
- `--model_path`: 模型保存/加载路径（默认：cell_dae_model）
- `--output_dir`: 结果输出目录（默认：anomaly_results）

### 训练参数
- `--sample_size`: 训练样本数量（默认：10000）
- `--epochs`: 训练轮数（默认：50）
- `--latent_dim`: 潜在空间维度（默认：64）

## 输出结果

### 训练阶段输出
- `cell_dae_model/cell_dae_model.h5`: 训练好的模型
- `anomaly_results/training_history.npy`: 训练历史
- `anomaly_results/training_curves.png`: 训练曲线图

### 检测阶段输出
- `anomaly_results/anomaly_results.npy`: 检测结果数据
- `anomaly_results/top_anomalies.txt`: 异常样本排行榜
- `anomaly_results/top_anomalies_visualization.png`: TOP异常样本可视化
- `anomaly_results/error_distribution.png`: 异常分数分布图

## 结果解读

### 异常分数
- **低分数（<0.1）**: 正常细胞，重建效果好
- **中等分数（0.1-0.2）**: 可能异常，需要进一步确认
- **高分数（>0.2）**: 高度可疑异常细胞

### 筛选策略
1. **统计筛选**: 选择95%或99%分位数以上的样本
2. **人工确认**: 对TOP异常样本进行生物学验证
3. **批量分析**: 结合细胞形态学特征分析

## 技术细节

### 模型架构
- **编码器**: 3层卷积+池化，降维到64维潜在空间
- **解码器**: 3层转置卷积+上采样，重建32x32x3图像
- **训练策略**: 去噪训练（添加10%高斯噪声）

### 数据处理
- **批处理**: 1000张图像/批，内存友好
- **归一化**: 像素值缩放到[0,1]范围
- **质量过滤**: 自动过滤过亮、过暗、颜色单一的图像

### 性能优化
- **GPU加速**: 自动检测并使用GPU
- **早停机制**: 防止过拟合
- **学习率调度**: 自适应学习率调整

## 进阶使用

### 自定义过滤器
修改`filter_artifacts`函数，添加针对你的数据的特定过滤逻辑：
```python
def custom_filter(errors, images, file_paths):
    # 添加你的过滤逻辑
    # 例如：基于细胞大小、形状、颜色特征的过滤
    return filtered_errors, filtered_images, filtered_paths
```

### 超参数调优
- **latent_dim**: 32-128，更大的值可以捕获更多细节
- **noise_factor**: 0.05-0.2，控制去噪训练强度
- **learning_rate**: 1e-5到1e-3，影响收敛速度

### 大规模部署
对于100万张tiles的完整检测：
1. 使用更多训练样本（50k-100k）
2. 增加训练轮数（100-200 epochs）
3. 考虑使用分布式训练
4. 实施增量检测策略

## 故障排除

### 常见问题
1. **内存不足**: 减少batch_size或sample_size
2. **训练不收敛**: 增加epochs或调整learning_rate
3. **异常检测效果差**: 增加训练样本数量或调整模型架构

### 性能监控
- 观察训练损失曲线，确保收敛
- 检查重建图像质量
- 分析异常分数分布的合理性

## 联系与支持
如有问题或需要定制化开发，请联系开发团队。
