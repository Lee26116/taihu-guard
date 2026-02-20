"""
ST-GAT: 时空图注意力网络完整模型
Spatial-Temporal Graph Attention Network for Water Quality Prediction
"""

import torch
import torch.nn as nn

from model.temporal_encoder import TemporalEncoder
from model.spatial_gat import SpatialGATBlock


class STGAT(nn.Module):
    """
    时空图注意力网络

    架构:
    1. Temporal Encoder: 对每个节点的时间序列编码
    2. Spatial GAT: 2层图注意力网络，捕获空间依赖
    3. Prediction Heads:
       - 水质回归: 预测未来 predict_steps 天的 11 维水质参数
       - 蓝藻预警: 4 分类 (无/轻/中/重)

    输入: (batch, num_nodes, seq_len, feature_dim)
    """

    def __init__(self, input_dim=25, hidden_dim=64, num_heads=4,
                 num_wq_params=11, predict_steps=7, num_bloom_classes=4,
                 dropout=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_wq_params = num_wq_params
        self.predict_steps = predict_steps
        self.num_bloom_classes = num_bloom_classes

        # 1. 时间编码器 (per node)
        self.temporal_encoder = TemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=1,
            dropout=dropout
        )

        # 2. 空间 GAT (2层, 多头注意力)
        self.spatial_gat = SpatialGATBlock(
            in_features=hidden_dim,
            hidden_dim=hidden_dim,
            out_features=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # 3a. 水质回归头
        self.wq_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_wq_params * predict_steps)
        )

        # 3b. 蓝藻预警分类头
        self.bloom_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_bloom_classes)
        )

    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: (batch, num_nodes, seq_len, feature_dim)
            edge_index: (2, num_edges) — 图结构
            edge_weight: (num_edges,) — 边权重

        Returns:
            wq_pred: (batch, num_nodes, predict_steps, num_wq_params)
            bloom_pred: (batch, num_nodes, num_bloom_classes)
        """
        batch_size, num_nodes, seq_len, feat_dim = x.shape

        # 1. 时间编码: 对每个节点独立处理
        # Reshape: (B*N, T, F)
        x_flat = x.reshape(batch_size * num_nodes, seq_len, feat_dim)
        h_temporal = self.temporal_encoder(x_flat)  # (B*N, hidden_dim)

        # Reshape 回: (B, N, hidden_dim)
        h_temporal = h_temporal.reshape(batch_size, num_nodes, self.hidden_dim)

        # 2. 空间 GAT: 对每个 batch 样本独立处理
        h_spatial_list = []
        for b in range(batch_size):
            h_b = self.spatial_gat(h_temporal[b], edge_index, edge_weight)  # (N, hidden_dim)
            h_spatial_list.append(h_b)

        h_spatial = torch.stack(h_spatial_list)  # (B, N, hidden_dim)

        # 3a. 水质回归预测
        wq_flat = self.wq_head(h_spatial)  # (B, N, predict_steps * num_wq_params)
        wq_pred = wq_flat.reshape(batch_size, num_nodes, self.predict_steps, self.num_wq_params)

        # 3b. 蓝藻预警分类
        bloom_pred = self.bloom_head(h_spatial)  # (B, N, num_bloom_classes)

        return wq_pred, bloom_pred


class STGATLoss(nn.Module):
    """
    ST-GAT 复合损失函数

    L = λ1 × MSE(水质预测) + λ2 × CrossEntropy(蓝藻分类) + λ3 × 时间衰减损失

    时间衰减: 越远的预测步骤权重越低
    """

    def __init__(self, lambda1=1.0, lambda2=0.5, lambda3=0.1, predict_steps=7):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.mse = nn.MSELoss(reduction="none")
        self.ce = nn.CrossEntropyLoss()

        # 时间衰减权重: 指数衰减
        decay = torch.exp(-0.1 * torch.arange(predict_steps, dtype=torch.float32))
        self.register_buffer("time_weights", decay / decay.sum())

    def forward(self, wq_pred, wq_target, bloom_pred, bloom_target):
        """
        Args:
            wq_pred: (B, N, P, 11) 水质预测
            wq_target: (B, N, P, 11) 水质标签
            bloom_pred: (B, N, 4) 蓝藻分类 logits
            bloom_target: (B, N) 蓝藻标签 (0-3)
        """
        # 1. 水质 MSE 损失
        mse_loss = self.mse(wq_pred, wq_target)  # (B, N, P, 11)
        mse_loss = mse_loss.mean(dim=[0, 1, 3])  # (P,) — 按时间步
        mse_loss = (mse_loss * self.time_weights).sum()  # 标量

        # 2. 蓝藻分类损失
        bloom_pred_flat = bloom_pred.reshape(-1, bloom_pred.size(-1))  # (B*N, 4)
        bloom_target_flat = bloom_target.reshape(-1)  # (B*N,)
        ce_loss = self.ce(bloom_pred_flat, bloom_target_flat)

        # 3. 时间衰减惩罚（对远期预测给额外正则化）
        # 鼓励模型对近期预测更精确
        step_errors = self.mse(wq_pred, wq_target).mean(dim=[0, 1, 3])  # (P,)
        time_decay_loss = (step_errors * torch.arange(
            len(step_errors), device=step_errors.device, dtype=torch.float32
        )).mean()

        # 总损失
        total_loss = (self.lambda1 * mse_loss +
                      self.lambda2 * ce_loss +
                      self.lambda3 * time_decay_loss)

        return total_loss, {
            "mse_loss": mse_loss.item(),
            "ce_loss": ce_loss.item(),
            "time_decay_loss": time_decay_loss.item(),
            "total_loss": total_loss.item()
        }
