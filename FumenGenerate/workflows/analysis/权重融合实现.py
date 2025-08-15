#!/usr/bin/env python3
"""
权重融合版混合模型实现

将原来的特征拼接改为真正的权重融合：
- 老师傅(随机森林) 权重: α 
- 学生(神经网络) 权重: β
- 最终预测 = α × RF预测 + β × NN预测

支持动态调整权重比例，实现真正的师徒协作调优
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class WeightedFusionHybridModel(nn.Module):
    """权重融合版混合模型"""
    
    def __init__(self, audio_features_dim=15, rf_features_dim=15, 
                 rf_weight=0.4, nn_weight=0.6):
        super(WeightedFusionHybridModel, self).__init__()
        
        # 权重融合参数（可学习）
        self.rf_weight = nn.Parameter(torch.tensor(rf_weight, dtype=torch.float32))
        self.nn_weight = nn.Parameter(torch.tensor(nn_weight, dtype=torch.float32))
        
        print(f"🤝 创建权重融合混合模型:")
        print(f"   🌲 初始RF权重: {rf_weight}")
        print(f"   🧠 初始NN权重: {nn_weight}")
        
        # 随机森林分支 - 模拟RF的决策过程
        self.rf_branch = nn.Sequential(
            nn.Linear(rf_features_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # 输出0-1概率
        )
        
        # 神经网络分支 - 深度特征学习
        self.nn_branch = nn.Sequential(
            nn.Linear(audio_features_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 输出0-1概率
        )
        
    def forward(self, audio_features, rf_features, return_individual=False):
        """
        权重融合前向传播
        
        Args:
            audio_features: 音频特征 [batch_size, audio_features_dim]
            rf_features: RF特征 [batch_size, rf_features_dim]
            return_individual: 是否返回各分支的单独预测
            
        Returns:
            如果return_individual=True: (fused_prob, rf_prob, nn_prob, weights)
            否则: fused_prob
        """
        # 确保权重归一化
        total_weight = self.rf_weight + self.nn_weight
        normalized_rf_weight = self.rf_weight / total_weight
        normalized_nn_weight = self.nn_weight / total_weight
        
        # 两个分支分别输出概率
        rf_prob = self.rf_branch(rf_features)      # [batch_size, 1]
        nn_prob = self.nn_branch(audio_features)   # [batch_size, 1]
        
        # 权重融合 - 这是关键！
        fused_prob = normalized_rf_weight * rf_prob + normalized_nn_weight * nn_prob
        
        if return_individual:
            return fused_prob, rf_prob, nn_prob, (normalized_rf_weight, normalized_nn_weight)
        else:
            return fused_prob
    
    def set_fusion_weights(self, rf_weight: float, nn_weight: float):
        """手动设置融合权重"""
        with torch.no_grad():
            self.rf_weight.data = torch.tensor(rf_weight, dtype=torch.float32)
            self.nn_weight.data = torch.tensor(nn_weight, dtype=torch.float32)
        
        print(f"🔧 手动设置融合权重:")
        print(f"   🌲 RF权重: {rf_weight:.3f}")
        print(f"   🧠 NN权重: {nn_weight:.3f}")
    
    def get_current_weights(self):
        """获取当前的融合权重"""
        total = self.rf_weight + self.nn_weight
        rf_w = self.rf_weight / total
        nn_w = self.nn_weight / total
        return rf_w.item(), nn_w.item()
    
    def freeze_weights(self, freeze=True):
        """冻结或解冻权重参数"""
        self.rf_weight.requires_grad = not freeze
        self.nn_weight.requires_grad = not freeze
        
        status = "冻结" if freeze else "解冻"
        print(f"🔒 权重参数已{status}")

class WeightFusionTrainer:
    """权重融合模型训练器"""
    
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'rf_weights': [],
            'nn_weights': [],
            'rf_accuracy': [],
            'nn_accuracy': [],
            'fusion_accuracy': []
        }
    
    def train_epoch(self, train_loader, verbose=True):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        correct_predictions = 0
        
        for batch_idx, (audio_features, rf_features, labels) in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            # 前向传播
            fused_prob, rf_prob, nn_prob, weights = self.model(
                audio_features, rf_features, return_individual=True
            )
            
            # 计算损失
            loss = self.criterion(fused_prob.squeeze(), labels.float())
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            predictions = (fused_prob.squeeze() > 0.5).float()
            correct_predictions += (predictions == labels.float()).sum().item()
            total_samples += labels.size(0)
            
            if verbose and batch_idx % 10 == 0:
                rf_w, nn_w = weights
                print(f"   批次 {batch_idx}: Loss={loss.item():.4f}, "
                      f"RF权重={rf_w:.3f}, NN权重={nn_w:.3f}")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def evaluate(self, test_loader):
        """评估模型"""
        self.model.eval()
        total_samples = 0
        fusion_correct = 0
        rf_correct = 0
        nn_correct = 0
        
        all_fusion_probs = []
        all_rf_probs = []
        all_nn_probs = []
        all_labels = []
        
        with torch.no_grad():
            for audio_features, rf_features, labels in test_loader:
                fused_prob, rf_prob, nn_prob, weights = self.model(
                    audio_features, rf_features, return_individual=True
                )
                
                # 预测
                fusion_pred = (fused_prob.squeeze() > 0.5).float()
                rf_pred = (rf_prob.squeeze() > 0.5).float()
                nn_pred = (nn_prob.squeeze() > 0.5).float()
                
                # 统计正确率
                fusion_correct += (fusion_pred == labels.float()).sum().item()
                rf_correct += (rf_pred == labels.float()).sum().item()
                nn_correct += (nn_pred == labels.float()).sum().item()
                total_samples += labels.size(0)
                
                # 保存概率用于分析
                all_fusion_probs.extend(fused_prob.squeeze().cpu().numpy())
                all_rf_probs.extend(rf_prob.squeeze().cpu().numpy())
                all_nn_probs.extend(nn_prob.squeeze().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        fusion_acc = fusion_correct / total_samples
        rf_acc = rf_correct / total_samples
        nn_acc = nn_correct / total_samples
        
        return {
            'fusion_accuracy': fusion_acc,
            'rf_accuracy': rf_acc,
            'nn_accuracy': nn_acc,
            'fusion_probs': all_fusion_probs,
            'rf_probs': all_rf_probs,
            'nn_probs': all_nn_probs,
            'labels': all_labels,
            'current_weights': self.model.get_current_weights()
        }
    
    def train(self, train_loader, test_loader, num_epochs=50):
        """完整训练过程"""
        print(f"🚀 开始权重融合模型训练 ({num_epochs} epochs)")
        print("=" * 60)
        
        best_accuracy = 0
        
        for epoch in range(num_epochs):
            print(f"\n📊 Epoch {epoch+1}/{num_epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, verbose=(epoch % 10 == 0))
            
            # 评估
            eval_results = self.evaluate(test_loader)
            
            # 记录历史
            self.training_history['loss'].append(train_loss)
            self.training_history['accuracy'].append(train_acc)
            self.training_history['rf_weights'].append(eval_results['current_weights'][0])
            self.training_history['nn_weights'].append(eval_results['current_weights'][1])
            self.training_history['rf_accuracy'].append(eval_results['rf_accuracy'])
            self.training_history['nn_accuracy'].append(eval_results['nn_accuracy'])
            self.training_history['fusion_accuracy'].append(eval_results['fusion_accuracy'])
            
            # 打印进度
            rf_w, nn_w = eval_results['current_weights']
            print(f"   训练损失: {train_loss:.4f}")
            print(f"   融合准确率: {eval_results['fusion_accuracy']:.4f}")
            print(f"   RF准确率: {eval_results['rf_accuracy']:.4f}")
            print(f"   NN准确率: {eval_results['nn_accuracy']:.4f}")
            print(f"   当前权重: RF={rf_w:.3f}, NN={nn_w:.3f}")
            
            # 保存最佳模型
            if eval_results['fusion_accuracy'] > best_accuracy:
                best_accuracy = eval_results['fusion_accuracy']
                torch.save(self.model.state_dict(), 'best_weighted_fusion_model.pth')
                print(f"   ✅ 保存最佳模型 (准确率: {best_accuracy:.4f})")
        
        print(f"\n🎉 训练完成！最佳准确率: {best_accuracy:.4f}")
        return eval_results

def create_mock_dataset(num_samples=5000):
    """创建模拟数据集用于演示"""
    print(f"🎲 创建模拟数据集 ({num_samples} 样本)")
    
    # 模拟音频特征 (15维)
    audio_features = np.random.randn(num_samples, 15).astype(np.float32)
    
    # 模拟RF特征 (15维，与音频特征相关但有噪音)
    rf_features = audio_features + 0.1 * np.random.randn(num_samples, 15)
    rf_features = rf_features.astype(np.float32)
    
    # 创建有意义的标签（基于特征的某种组合）
    # 模拟音符检测：RMS能量高 + 频谱质心合适 = 有音符
    rms_energy = audio_features[:, 0]  # 假设第一维是RMS能量
    spectral_centroid = audio_features[:, 1]  # 假设第二维是频谱质心
    
    # 创建标签：能量高且质心在合理范围内
    labels = ((rms_energy > 0.5) & (np.abs(spectral_centroid) < 1.0)).astype(np.float32)
    
    # 添加一些噪音让问题更真实
    noise_indices = np.random.choice(num_samples, size=int(0.1 * num_samples), replace=False)
    labels[noise_indices] = 1 - labels[noise_indices]
    
    print(f"   ✅ 数据集创建完成")
    print(f"   📊 正样本比例: {np.mean(labels):.3f}")
    
    return audio_features, rf_features, labels

def create_data_loader(audio_features, rf_features, labels, batch_size=32, shuffle=True):
    """创建数据加载器"""
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(audio_features),
        torch.FloatTensor(rf_features), 
        torch.FloatTensor(labels)
    )
    
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

def plot_training_results(trainer):
    """绘制训练结果"""
    history = trainer.training_history
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('权重融合模型训练结果', fontsize=16)
    
    # 1. 损失和准确率
    axes[0, 0].plot(history['loss'], label='训练损失', color='red')
    ax_twin = axes[0, 0].twinx()
    ax_twin.plot(history['accuracy'], label='训练准确率', color='blue')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('损失', color='red')
    ax_twin.set_ylabel('准确率', color='blue')
    axes[0, 0].set_title('训练损失和准确率')
    
    # 2. 权重变化
    axes[0, 1].plot(history['rf_weights'], label='RF权重', marker='o')
    axes[0, 1].plot(history['nn_weights'], label='NN权重', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('权重值')
    axes[0, 1].set_title('融合权重动态变化')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. 各分支准确率对比
    axes[1, 0].plot(history['fusion_accuracy'], label='融合模型', linewidth=2)
    axes[1, 0].plot(history['rf_accuracy'], label='RF分支', linestyle='--')
    axes[1, 0].plot(history['nn_accuracy'], label='NN分支', linestyle=':')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('准确率')
    axes[1, 0].set_title('各模型准确率对比')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 4. 最终权重分布
    final_rf_weight = history['rf_weights'][-1]
    final_nn_weight = history['nn_weights'][-1]
    
    weights = [final_rf_weight, final_nn_weight]
    labels_pie = [f'随机森林\n{final_rf_weight:.3f}', f'神经网络\n{final_nn_weight:.3f}']
    colors = ['lightblue', 'lightcoral']
    
    axes[1, 1].pie(weights, labels=labels_pie, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('最终权重分布')
    
    plt.tight_layout()
    plt.savefig('weighted_fusion_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def weight_sensitivity_analysis(model, test_loader):
    """权重敏感性分析"""
    print("\n🔬 权重敏感性分析")
    print("=" * 40)
    
    # 测试不同权重组合
    weight_combinations = [
        (0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.5),
        (0.6, 0.4), (0.7, 0.3), (0.8, 0.2), (0.9, 0.1)
    ]
    
    results = []
    
    for rf_w, nn_w in weight_combinations:
        model.set_fusion_weights(rf_w, nn_w)
        
        # 评估
        model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for audio_features, rf_features, labels in test_loader:
                fused_prob = model(audio_features, rf_features)
                predictions = (fused_prob.squeeze() > 0.5).float()
                total_correct += (predictions == labels.float()).sum().item()
                total_samples += labels.size(0)
        
        accuracy = total_correct / total_samples
        results.append((rf_w, nn_w, accuracy))
        
        print(f"RF权重: {rf_w:.1f}, NN权重: {nn_w:.1f} → 准确率: {accuracy:.4f}")
    
    # 找出最佳权重组合
    best_result = max(results, key=lambda x: x[2])
    print(f"\n🏆 最佳权重组合:")
    print(f"   RF权重: {best_result[0]:.1f}")
    print(f"   NN权重: {best_result[1]:.1f}")
    print(f"   准确率: {best_result[2]:.4f}")
    
    return results, best_result

def main():
    """主演示函数"""
    print("🎮 权重融合混合模型完整实现")
    print("=" * 60)
    
    # 1. 创建模拟数据
    audio_features, rf_features, labels = create_mock_dataset(5000)
    
    # 2. 数据分割
    X_audio_train, X_audio_test, X_rf_train, X_rf_test, y_train, y_test = train_test_split(
        audio_features, rf_features, labels, test_size=0.2, random_state=42
    )
    
    # 3. 创建数据加载器
    train_loader = create_data_loader(X_audio_train, X_rf_train, y_train, batch_size=64)
    test_loader = create_data_loader(X_audio_test, X_rf_test, y_test, batch_size=64, shuffle=False)
    
    print(f"📊 数据集信息:")
    print(f"   训练集: {len(X_audio_train)} 样本")
    print(f"   测试集: {len(X_audio_test)} 样本")
    
    # 4. 创建权重融合模型
    print(f"\n🏗️ 创建权重融合模型...")
    model = WeightedFusionHybridModel(
        audio_features_dim=15,
        rf_features_dim=15,
        rf_weight=0.4,  # 初始老师傅权重
        nn_weight=0.6   # 初始学生权重
    )
    
    # 5. 创建训练器
    trainer = WeightFusionTrainer(model, learning_rate=0.001)
    
    # 6. 开始训练
    print(f"\n🚀 开始训练...")
    final_results = trainer.train(train_loader, test_loader, num_epochs=30)
    
    # 7. 绘制训练结果
    print(f"\n📊 绘制训练结果...")
    plot_training_results(trainer)
    
    # 8. 权重敏感性分析
    sensitivity_results, best_weights = weight_sensitivity_analysis(model, test_loader)
    
    # 9. 总结
    print(f"\n🎉 权重融合实现完成！")
    print("=" * 40)
    print(f"📈 最终结果:")
    print(f"   🤝 融合模型准确率: {final_results['fusion_accuracy']:.4f}")
    print(f"   🌲 RF分支准确率: {final_results['rf_accuracy']:.4f}")
    print(f"   🧠 NN分支准确率: {final_results['nn_accuracy']:.4f}")
    
    rf_w, nn_w = final_results['current_weights']
    print(f"   ⚖️ 学习到的权重: RF={rf_w:.3f}, NN={nn_w:.3f}")
    print(f"   🏆 最优权重组合: RF={best_weights[0]:.1f}, NN={best_weights[1]:.1f}")
    
    print(f"\n💾 保存的文件:")
    print(f"   • best_weighted_fusion_model.pth - 最佳模型")
    print(f"   • weighted_fusion_training_results.png - 训练图表")
    
    return model, trainer, final_results

if __name__ == "__main__":
    model, trainer, results = main()
