#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
权重融合模型 - 支持大规模训练的权重融合架构
基于Transformer和CNN的混合模型，实现RF:NN权重融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WeightedFusionTransformer(nn.Module):
    """权重融合Transformer模型 - 专为谱面生成优化"""
    
    def __init__(self, input_dim=15, d_model=256, num_heads=8, num_layers=6, 
                 dropout=0.1, rf_weight=0.2, nn_weight=0.8, 
                 learnable_weights=True):
        super().__init__()
        
        # 权重参数p
        if learnable_weights:
            self.rf_weight = nn.Parameter(torch.tensor(rf_weight, dtype=torch.float32))
            self.nn_weight = nn.Parameter(torch.tensor(nn_weight, dtype=torch.float32))
        else:
            self.register_buffer('rf_weight', torch.tensor(rf_weight, dtype=torch.float32))
            self.register_buffer('nn_weight', torch.tensor(nn_weight, dtype=torch.float32))
        
        self.learnable_weights = learnable_weights
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # RF分支 - 模拟传统决策树/随机森林逻辑
        self.rf_branch = RFBranch(d_model, dropout)
        
        # NN分支 - 深度Transformer网络
        self.nn_branch = NNBranch(d_model, num_heads, num_layers, dropout)
        
        # 输出层
        self.output_notes = nn.Linear(d_model, 4)     # 4轨道音符
        self.output_events = nn.Linear(d_model, 3)    # 3种事件类型
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # 输入投影
        x_projected = self.input_projection(x)  # [batch, d_model]
        
        # 权重归一化
        total_weight = self.rf_weight + self.nn_weight
        normalized_rf = self.rf_weight / (total_weight + 1e-8)
        normalized_nn = self.nn_weight / (total_weight + 1e-8)
        
        # RF分支处理
        rf_features = self.rf_branch(x_projected)
        
        # NN分支处理
        nn_features = self.nn_branch(x_projected)
        
        # 权重融合
        fused_features = normalized_rf * rf_features + normalized_nn * nn_features
        
        # 输出预测
        note_output = torch.sigmoid(self.output_notes(fused_features))
        event_output = torch.sigmoid(self.output_events(fused_features))
        
        # 返回分离的输出以匹配基础系统期望的格式
        return note_output, event_output
    
    def get_branch_predictions(self, x):
        """获取各分支的单独预测，用于分析"""
        with torch.no_grad():
            x_projected = self.input_projection(x)
            
            # 各分支特征
            rf_features = self.rf_branch(x_projected)
            nn_features = self.nn_branch(x_projected)
            
            # 各分支输出
            rf_notes = torch.sigmoid(self.output_notes(rf_features))
            rf_events = torch.sigmoid(self.output_events(rf_features))
            rf_pred = torch.cat([rf_notes, rf_events], dim=1)
            
            nn_notes = torch.sigmoid(self.output_notes(nn_features))
            nn_events = torch.sigmoid(self.output_events(nn_features))
            nn_pred = torch.cat([nn_notes, nn_events], dim=1)
            
            # 融合预测
            fused_note_pred, fused_event_pred = self.forward(x)
            fused_pred = torch.cat([fused_note_pred, fused_event_pred], dim=1)
            
            # 当前权重
            total_weight = self.rf_weight + self.nn_weight
            
            return {
                'rf_pred': rf_pred,
                'nn_pred': nn_pred,
                'fused_pred': fused_pred,
                'rf_weight': (self.rf_weight / total_weight).item(),
                'nn_weight': (self.nn_weight / total_weight).item(),
                'rf_features': rf_features,
                'nn_features': nn_features
            }
    
    def set_weights(self, rf_weight, nn_weight):
        """设置权重比例"""
        if self.learnable_weights:
            with torch.no_grad():
                self.rf_weight.data = torch.tensor(rf_weight, dtype=torch.float32)
                self.nn_weight.data = torch.tensor(nn_weight, dtype=torch.float32)
        else:
            self.rf_weight.data = torch.tensor(rf_weight, dtype=torch.float32)
            self.nn_weight.data = torch.tensor(nn_weight, dtype=torch.float32)
    
    def get_weights(self):
        """获取当前权重"""
        total = self.rf_weight + self.nn_weight
        return {
            'rf_weight': (self.rf_weight / total).item(),
            'nn_weight': (self.nn_weight / total).item(),
            'raw_rf': self.rf_weight.item(),
            'raw_nn': self.nn_weight.item()
        }

class RFBranch(nn.Module):
    """RF分支 - 模拟随机森林的决策逻辑"""
    
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        
        # 多个决策路径，模拟随机森林的多棵树
        self.decision_paths = nn.ModuleList([
            self._create_decision_path(d_model, dropout) for _ in range(8)
        ])
        
        # 路径融合
        self.path_fusion = nn.Sequential(
            nn.Linear(d_model * 8, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU()
        )
        
    def _create_decision_path(self, d_model, dropout):
        """创建单个决策路径（模拟一棵决策树）"""
        return nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),  # RF分支使用较少的dropout
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model)
        )
    
    def forward(self, x):
        # 多路径处理
        path_outputs = []
        for path in self.decision_paths:
            path_out = path(x)
            path_outputs.append(path_out)
        
        # 连接所有路径输出
        concatenated = torch.cat(path_outputs, dim=1)
        
        # 融合多路径特征
        fused = self.path_fusion(concatenated)
        
        return fused

class NNBranch(nn.Module):
    """NN分支 - 深度神经网络特征提取"""
    
    def __init__(self, d_model, num_heads=8, num_layers=6, dropout=0.1):
        super().__init__()
        
        # 多头自注意力层
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # 前馈网络层
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # 层归一化
        self.layer_norms1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.layer_norms2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        
        # 最终特征提取
        self.final_projection = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
    def forward(self, x):
        # x: [batch, d_model]
        # 扩展维度以适应多头注意力 [batch, 1, d_model]
        x = x.unsqueeze(1)
        
        # 多层Transformer处理
        for i, (attn, ffn, ln1, ln2) in enumerate(zip(
            self.attention_layers, self.ffn_layers, 
            self.layer_norms1, self.layer_norms2
        )):
            # 自注意力
            attn_out, _ = attn(x, x, x)
            x = ln1(x + attn_out)
            
            # 前馈网络
            ffn_out = ffn(x)
            x = ln2(x + ffn_out)
        
        # 移除序列维度 [batch, d_model]
        x = x.squeeze(1)
        
        # 最终特征投影
        x = self.final_projection(x)
        
        return x

class WeightedFusionCNN(nn.Module):
    """基于CNN的权重融合模型 - 适用于序列数据"""
    
    def __init__(self, input_dim=15, rf_weight=0.2, nn_weight=0.8, 
                 learnable_weights=True):
        super().__init__()
        
        # 权重参数
        if learnable_weights:
            self.rf_weight = nn.Parameter(torch.tensor(rf_weight, dtype=torch.float32))
            self.nn_weight = nn.Parameter(torch.tensor(nn_weight, dtype=torch.float32))
        else:
            self.register_buffer('rf_weight', torch.tensor(rf_weight, dtype=torch.float32))
            self.register_buffer('nn_weight', torch.tensor(nn_weight, dtype=torch.float32))
        
        # RF分支 - 模拟传统特征提取
        self.rf_branch = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7),
            nn.Sigmoid()
        )
        
        # NN分支 - 深度卷积网络
        self.nn_branch = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 如果输入是2D，需要转换为3D进行卷积
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [batch, features, 1]
            x = x.transpose(1, 2)  # [batch, 1, features]
            x = x.expand(-1, x.size(-1), -1)  # [batch, features, features]
        
        # 权重归一化
        total_weight = self.rf_weight + self.nn_weight
        normalized_rf = self.rf_weight / (total_weight + 1e-8)
        normalized_nn = self.nn_weight / (total_weight + 1e-8)
        
        # 两个分支的预测
        rf_pred = self.rf_branch(x)
        nn_pred = self.nn_branch(x)
        
        # 权重融合
        fused_pred = normalized_rf * rf_pred + normalized_nn * nn_pred
        
        return fused_pred
    
    def get_branch_predictions(self, x):
        """获取各分支预测"""
        with torch.no_grad():
            if x.dim() == 2:
                x = x.unsqueeze(-1).transpose(1, 2).expand(-1, x.size(-1), -1)
            
            rf_pred = self.rf_branch(x)
            nn_pred = self.nn_branch(x)
            fused_pred = self.forward(x)
            
            total_weight = self.rf_weight + self.nn_weight
            return {
                'rf_pred': rf_pred,
                'nn_pred': nn_pred,
                'fused_pred': fused_pred,
                'rf_weight': (self.rf_weight / total_weight).item(),
                'nn_weight': (self.nn_weight / total_weight).item()
            }

def create_weighted_fusion_model(model_type='transformer', **kwargs):
    """模型工厂函数"""
    if model_type == 'transformer':
        return WeightedFusionTransformer(**kwargs)
    elif model_type == 'cnn':
        return WeightedFusionCNN(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

# 导出主要类
__all__ = [
    'WeightedFusionTransformer',
    'WeightedFusionCNN', 
    'RFBranch',
    'NNBranch',
    'create_weighted_fusion_model'
]

if __name__ == "__main__":
    # 简单测试
    print("🧪 权重融合模型测试")
    
    # 测试Transformer模型
    model = WeightedFusionTransformer(
        input_dim=15, 
        rf_weight=0.2, 
        nn_weight=0.8
    )
    
    # 模拟输入
    batch_size = 32
    x = torch.randn(batch_size, 15)
    
    # 前向传播
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 分支分析
    branch_results = model.get_branch_predictions(x)
    print(f"RF权重: {branch_results['rf_weight']:.3f}")
    print(f"NN权重: {branch_results['nn_weight']:.3f}")
    
    print("✅ 权重融合模型测试通过！")
