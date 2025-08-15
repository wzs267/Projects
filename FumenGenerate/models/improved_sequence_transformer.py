"""
音游专用的64步序列Transformer设计
================================

基于音游的特殊需求，充分利用64个时间步的历史信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RhythmGameTransformer(nn.Module):
    """音游专用的序列Transformer"""
    
    def __init__(self, input_dim=15, d_model=256, num_heads=8, num_layers=6, 
                 dropout=0.1, max_seq_len=64):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 输入投影：将15维特征映射到256维
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码：让模型理解时间顺序
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # 音游专用的多层次注意力
        self.transformer_layers = nn.ModuleList([
            RhythmTransformerLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 时序信息压缩：从64步→1步决策
        self.temporal_compression = TemporalCompressionLayer(d_model)
        
        # 输出层
        self.output_notes = nn.Linear(d_model, 4)   # 4轨道音符
        self.output_events = nn.Linear(d_model, 3)  # 3种事件类型
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Args:
            x: [batch, 64, 15] - 64个时间步的音频特征
        Returns:
            note_output: [batch, 4] - 4轨道音符概率
            event_output: [batch, 3] - 3种事件概率
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 输入投影：每个时间步的特征 15→256
        x = self.input_projection(x)  # [batch, 64, 256]
        
        # 2. 位置编码：添加时间位置信息
        x = self.pos_encoding(x)      # [batch, 64, 256]
        
        # 3. 多层Transformer处理
        for layer in self.transformer_layers:
            x = layer(x)              # [batch, 64, 256]
        
        # 4. 时序压缩：64步→1个决策
        decision_features = self.temporal_compression(x)  # [batch, 256]
        
        # 5. 输出预测
        note_output = torch.sigmoid(self.output_notes(decision_features))
        event_output = torch.sigmoid(self.output_events(decision_features))
        
        return note_output, event_output

class PositionalEncoding(nn.Module):
    """位置编码：让模型知道每个时间步的位置"""
    
    def __init__(self, d_model, max_len=64):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        # 添加位置编码
        x = x + self.pe[:seq_len, :].transpose(0, 1)
        return x

class RhythmTransformerLayer(nn.Module):
    """音游专用的Transformer层"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        
        # 多头自注意力
        self.self_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        # 1. 自注意力 + 残差连接
        attn_out, attn_weights = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 2. 前馈网络 + 残差连接
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x

class TemporalCompressionLayer(nn.Module):
    """时序压缩层：将64步历史压缩为单个决策特征"""
    
    def __init__(self, d_model):
        super().__init__()
        
        # 多种压缩策略
        self.current_projection = nn.Linear(d_model, d_model // 2)
        self.history_compression = nn.Sequential(
            nn.Conv1d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, 64, d_model]
        Returns:
            compressed: [batch, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # 1. 当前时刻特征（最后一步）
        current_features = self.current_projection(x[:, -1, :])  # [batch, d_model//2]
        
        # 2. 历史信息压缩
        x_transposed = x.transpose(1, 2)  # [batch, d_model, 64]
        history_features = self.history_compression(x_transposed)  # [batch, d_model//2, 1]
        history_features = history_features.squeeze(-1)  # [batch, d_model//2]
        
        # 3. 融合当前+历史
        combined = torch.cat([current_features, history_features], dim=1)  # [batch, d_model]
        compressed = self.fusion(combined)
        
        return compressed

class ImprovedWeightedFusionTransformer(nn.Module):
    """改进的权重融合Transformer - 真正利用64步序列"""
    
    def __init__(self, input_dim=15, d_model=256, num_heads=8, num_layers=6, 
                 dropout=0.1, rf_weight=0.3, nn_weight=0.7, learnable_weights=True):
        super().__init__()
        
        # 权重参数
        if learnable_weights:
            self.rf_weight = nn.Parameter(torch.tensor(rf_weight, dtype=torch.float32))
            self.nn_weight = nn.Parameter(torch.tensor(nn_weight, dtype=torch.float32))
        else:
            self.register_buffer('rf_weight', torch.tensor(rf_weight, dtype=torch.float32))
            self.register_buffer('nn_weight', torch.tensor(nn_weight, dtype=torch.float32))
        
        # RF分支：增强的随机森林模拟
        self.rf_branch = EnhancedRFBranch(input_dim, d_model, dropout)
        
        # NN分支：完整的序列Transformer
        self.nn_branch = RhythmGameTransformer(input_dim, d_model, num_heads, num_layers, dropout)
        
    def forward(self, x):
        """
        Args:
            x: [batch, 64, 15] - 真实的64步序列
        """
        # 权重归一化
        total_weight = self.rf_weight + self.nn_weight
        normalized_rf = self.rf_weight / (total_weight + 1e-8)
        normalized_nn = self.nn_weight / (total_weight + 1e-8)
        
        # RF分支：简单序列处理
        rf_notes, rf_events = self.rf_branch(x)
        
        # NN分支：复杂序列处理
        nn_notes, nn_events = self.nn_branch(x)
        
        # 权重融合
        fused_notes = normalized_rf * rf_notes + normalized_nn * nn_notes
        fused_events = normalized_rf * rf_events + normalized_nn * nn_events
        
        return fused_notes, fused_events
    
    def get_weights(self):
        """获取当前权重"""
        total = self.rf_weight + self.nn_weight
        return {
            'rf_weight': (self.rf_weight / total).item(),
            'nn_weight': (self.nn_weight / total).item()
        }

class EnhancedRFBranch(nn.Module):
    """增强的RF分支 - 模拟真实随机森林的复杂决策逻辑"""
    
    def __init__(self, input_dim, d_model, dropout=0.1, num_trees=32, tree_depth=5):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_trees = num_trees
        
        # 特征子集选择器（模拟随机森林的特征随机采样）
        self.feature_selectors = nn.ModuleList([
            FeatureSelector(input_dim, tree_idx) for tree_idx in range(num_trees)
        ])
        
        # 多棵决策树（不同深度和结构）
        self.decision_trees = nn.ModuleList([
            DecisionTreeNetwork(input_dim, tree_depth, dropout, tree_idx, 
                              feature_ratio=0.7) 
            for tree_idx in range(num_trees)
        ])
        
        # 树的权重（模拟不同树的重要性）
        self.tree_weights = nn.Parameter(torch.ones(num_trees) / num_trees)
        
        # 时序特征提取（RF也需要理解时序）
        self.temporal_features = TemporalFeatureExtractor(input_dim, 64)
        
        # 集成学习融合层
        self.ensemble_fusion = nn.Sequential(
            nn.Linear(num_trees * 16 + 32, d_model),  # 树输出 + 时序特征
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU()
        )
        
        # 输出层
        self.output_notes = nn.Sequential(
            nn.Linear(d_model // 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),  # RF使用较少dropout
            nn.Linear(32, 4),
            nn.Sigmoid()
        )
        
        self.output_events = nn.Sequential(
            nn.Linear(d_model // 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, 64, input_dim] - 64步序列
        Returns:
            notes: [batch, 4] - 音符输出
            events: [batch, 3] - 事件输出
        """
        batch_size = x.size(0)
        
        # 1. 提取时序特征
        temporal_features = self.temporal_features(x)  # [batch, 32]
        
        # 2. 多棵决策树并行处理
        tree_outputs = []
        for tree_idx, (selector, tree) in enumerate(zip(self.feature_selectors, self.decision_trees)):
            # 特征选择
            selected_features = selector(x)  # [batch, 64, selected_dim]
            
            # 决策树处理
            tree_output = tree(selected_features)  # [batch, 16]
            tree_outputs.append(tree_output)
        
        # 3. 集成所有树的输出
        tree_features = torch.cat(tree_outputs, dim=1)  # [batch, num_trees * 16]
        
        # 4. 应用树权重（模拟投票机制）
        tree_weights = F.softmax(self.tree_weights, dim=0)  # 归一化权重
        weighted_trees = []
        for i in range(self.num_trees):
            start_idx = i * 16
            end_idx = (i + 1) * 16
            weighted_tree = tree_weights[i] * tree_features[:, start_idx:end_idx]
            weighted_trees.append(weighted_tree)
        
        weighted_tree_features = torch.cat(weighted_trees, dim=1)  # [batch, num_trees * 16]
        
        # 5. 融合树特征和时序特征
        combined_features = torch.cat([weighted_tree_features, temporal_features], dim=1)
        fused_features = self.ensemble_fusion(combined_features)  # [batch, d_model // 2]
        
        # 6. 输出预测
        notes = self.output_notes(fused_features)
        events = self.output_events(fused_features)
        
        return notes, events

class FeatureSelector(nn.Module):
    """特征选择器 - 模拟随机森林的特征随机采样"""
    
    def __init__(self, input_dim, tree_idx, feature_ratio=0.7):
        super().__init__()
        
        # 根据树索引生成确定性的特征选择
        torch.manual_seed(tree_idx * 42)  # 确保每棵树有不同但固定的特征选择
        
        num_features = max(int(input_dim * feature_ratio), 1)
        self.selected_features = torch.randperm(input_dim)[:num_features].sort()[0]
        self.register_buffer('feature_indices', self.selected_features)
        
        # 恢复随机种子
        torch.manual_seed(torch.initial_seed())
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim] or [batch, input_dim]
        Returns:
            selected_x: 选中的特征子集
        """
        if x.dim() == 3:
            return x[:, :, self.feature_indices]
        else:
            return x[:, self.feature_indices]

class DecisionTreeNetwork(nn.Module):
    """神经网络模拟的决策树"""
    
    def __init__(self, input_dim, depth=5, dropout=0.1, tree_idx=0, feature_ratio=0.7):
        super().__init__()
        
        self.depth = depth
        self.tree_idx = tree_idx
        
        # 计算实际使用的特征数量
        self.num_features = max(int(input_dim * feature_ratio), 1)
        
        # 根据树索引创建不同的架构
        width_factor = 1.0 + 0.1 * (tree_idx % 5)  # 不同树有不同宽度
        hidden_dim = int(64 * width_factor)
        
        # 决策路径网络（模拟决策树的分支逻辑）
        layers = []
        current_dim = self.num_features  # 使用实际特征数量
        
        for i in range(depth):
            next_dim = hidden_dim // (2 ** i) if i < depth - 1 else 16
            next_dim = max(next_dim, 8)  # 最小维度
            
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.ReLU() if i % 2 == 0 else nn.LeakyReLU(0.1),  # 交替激活函数
                nn.Dropout(dropout * 0.5) if i < depth - 1 else nn.Identity()
            ])
            current_dim = next_dim
        
        self.decision_path = nn.Sequential(*layers)
        
        # 叶子节点特征
        self.leaf_features = nn.Linear(16, 16)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, selected_features] or [batch, selected_features]
        Returns:
            tree_output: [batch, 16] - 单棵树的输出特征
        """
        # 如果是序列输入，取最后一个时间步（RF关注当前时刻）
        if x.dim() == 3:
            x = x[:, -1, :]  # [batch, selected_features]
        
        # 决策路径处理
        features = self.decision_path(x)  # [batch, 16]
        
        # 叶子节点处理
        output = self.leaf_features(features)  # [batch, 16]
        
        return output

class TemporalFeatureExtractor(nn.Module):
    """时序特征提取器 - RF分支的轻量级时序理解"""
    
    def __init__(self, input_dim, seq_len):
        super().__init__()
        
        # 多尺度卷积（不同时间窗口）
        self.conv_short = nn.Conv1d(input_dim, 8, kernel_size=3, padding=1)    # 短期
        self.conv_medium = nn.Conv1d(input_dim, 8, kernel_size=7, padding=3)   # 中期
        self.conv_long = nn.Conv1d(input_dim, 8, kernel_size=15, padding=7)    # 长期
        
        # 池化层
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(24, 32),  # 3个尺度 × 8维 = 24维
            nn.ReLU(),
            nn.Linear(32, 32)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            temporal_features: [batch, 32]
        """
        # 转换为卷积格式
        x = x.transpose(1, 2)  # [batch, input_dim, seq_len]
        
        # 多尺度特征提取
        short_features = self.adaptive_pool(F.relu(self.conv_short(x))).squeeze(-1)   # [batch, 8]
        medium_features = self.adaptive_pool(F.relu(self.conv_medium(x))).squeeze(-1) # [batch, 8]
        long_features = self.adaptive_pool(F.relu(self.conv_long(x))).squeeze(-1)     # [batch, 8]
        
        # 特征融合
        combined = torch.cat([short_features, medium_features, long_features], dim=1)  # [batch, 24]
        temporal_features = self.fusion(combined)  # [batch, 32]
        
        return temporal_features

def main():
    """测试新架构"""
    print("🎵 测试改进的64步序列Transformer")
    
    # 创建模型
    model = ImprovedWeightedFusionTransformer(
        input_dim=15,
        d_model=256, 
        num_heads=8,
        num_layers=6
    )
    
    # 测试输入
    batch_size = 32
    seq_len = 64
    input_dim = 15
    
    x = torch.randn(batch_size, seq_len, input_dim)
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    notes, events = model(x)
    print(f"音符输出: {notes.shape}")
    print(f"事件输出: {events.shape}")
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数数: {total_params:,}")
    
    # 权重信息
    weights = model.get_weights()
    print(f"权重比例: RF={weights['rf_weight']:.3f}, NN={weights['nn_weight']:.3f}")

if __name__ == "__main__":
    main()
