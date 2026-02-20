"""
V2 ST-GAT: 时空图注意力网络
Spatial-Temporal Graph Attention Network for Water Quality Prediction

V2 升级:
  - Transformer Encoder (替代 Conv1D+GRU)
  - hidden_dim 256, 8 heads, 4 GAT layers
  - 14 天预测窗口
  - 不确定性估计 (预测均值 + 方差)
  - ~800 万参数
"""

import torch
import torch.nn as nn

from model.temporal_encoder import TemporalTransformerEncoder
from model.spatial_gat import SpatialGATBlock


class STGAT(nn.Module):
    """
    V2 时空图注意力网络

    架构:
    1. Temporal Transformer: 多头自注意力时间编码
    2. Spatial GAT: 4层8头图注意力，捕获空间依赖
    3. Spatial-Temporal Fusion: 跨站点时空交互
    4. Prediction Heads:
       - 水质回归: 均值 + 方差（不确定性）
       - 蓝藻预警: 4 分类

    输入: (batch, num_nodes, seq_len, feature_dim)
    """

    def __init__(self, input_dim=35, hidden_dim=256, num_heads=8,
                 num_wq_params=11, predict_steps=14, num_bloom_classes=4,
                 temporal_layers=4, spatial_layers=4, ff_dim=512,
                 dropout=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_wq_params = num_wq_params
        self.predict_steps = predict_steps
        self.num_bloom_classes = num_bloom_classes

        # 1. Temporal Transformer (per node)
        self.temporal_encoder = TemporalTransformerEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=temporal_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout
        )

        # 2. Spatial GAT (4层, 多头注意力)
        self.spatial_gat = SpatialGATBlock(
            in_features=hidden_dim,
            hidden_dim=hidden_dim,
            out_features=hidden_dim,
            num_heads=num_heads,
            num_layers=spatial_layers,
            dropout=dropout
        )

        # 3. Spatial-Temporal Fusion (跨站点交互)
        self.st_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 4a. 水质回归头 — 预测均值
        self.wq_mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_wq_params * predict_steps)
        )

        # 4b. 水质回归头 — 预测不确定性 (log variance)
        self.wq_var_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_wq_params * predict_steps)
        )

        # 4c. 蓝藻预警分类头
        self.bloom_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, num_bloom_classes)
        )

    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: (batch, num_nodes, seq_len, feature_dim)
            edge_index: (2, num_edges)
            edge_weight: (num_edges,)

        Returns:
            wq_pred: (batch, num_nodes, predict_steps, num_wq_params) — 均值
            wq_log_var: (batch, num_nodes, predict_steps, num_wq_params) — 对数方差
            bloom_pred: (batch, num_nodes, num_bloom_classes) — 分类 logits
        """
        batch_size, num_nodes, seq_len, feat_dim = x.shape

        # 1. Temporal Transformer: 对每个节点独立编码
        x_flat = x.reshape(batch_size * num_nodes, seq_len, feat_dim)
        h_temporal = self.temporal_encoder(x_flat)  # (B*N, H)
        h_temporal = h_temporal.reshape(batch_size, num_nodes, self.hidden_dim)

        # 2. Spatial GAT: 批量化处理 (将 batch 维合并到 edge_index)
        h_flat = h_temporal.reshape(batch_size * num_nodes, self.hidden_dim)
        # 为每个 batch 复制 edge_index 并偏移节点编号
        offsets = torch.arange(batch_size, device=x.device).unsqueeze(1) * num_nodes  # (B, 1)
        batch_edge_index = (edge_index.unsqueeze(0) + offsets.unsqueeze(1)).reshape(2, -1)  # (2, B*E)
        batch_edge_weight = edge_weight.repeat(batch_size) if edge_weight is not None else None
        h_spatial = self.spatial_gat(h_flat, batch_edge_index, batch_edge_weight)
        h_spatial = h_spatial.reshape(batch_size, num_nodes, self.hidden_dim)  # (B, N, H)

        # 3. Spatial-Temporal Fusion
        h_fused = self.st_fusion(
            torch.cat([h_temporal, h_spatial], dim=-1)
        )  # (B, N, H)

        # 4a. 水质回归 — 均值
        wq_mean = self.wq_mean_head(h_fused)  # (B, N, P*11)
        wq_pred = wq_mean.reshape(batch_size, num_nodes, self.predict_steps, self.num_wq_params)

        # 4b. 水质回归 — 不确定性
        wq_lv = self.wq_var_head(h_fused)
        wq_log_var = wq_lv.reshape(batch_size, num_nodes, self.predict_steps, self.num_wq_params)

        # 4c. 蓝藻预警分类
        bloom_pred = self.bloom_head(h_fused)  # (B, N, 4)

        return wq_pred, wq_log_var, bloom_pred


class STGATLoss(nn.Module):
    """
    V2 复合损失函数

    L = λ1 × GaussianNLL(水质) + λ2 × FocalLoss(蓝藻) + λ3 × 时间衰减

    改进:
      - Gaussian NLL: 同时优化预测均值和不确定性
      - Focal Loss: 处理蓝藻类别不平衡
    """

    def __init__(self, lambda1=1.0, lambda2=0.5, lambda3=0.1,
                 predict_steps=14, focal_gamma=2.0):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.focal_gamma = focal_gamma

        # 时间衰减权重
        decay = torch.exp(-0.05 * torch.arange(predict_steps, dtype=torch.float32))
        self.register_buffer("time_weights", decay / decay.sum())

    def gaussian_nll_loss(self, pred_mean, pred_log_var, target):
        """
        高斯负对数似然损失
        同时优化预测均值和不确定性
        """
        var = torch.exp(pred_log_var).clamp(min=1e-6)
        nll = 0.5 * (pred_log_var + (target - pred_mean) ** 2 / var)
        return nll

    def focal_loss(self, logits, targets, gamma=2.0):
        """Focal Loss — 解决类别不平衡"""
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** gamma) * ce
        return focal.mean()

    def forward(self, wq_pred, wq_log_var, wq_target, bloom_pred, bloom_target):
        """
        Args:
            wq_pred: (B, N, P, 11) 水质预测均值
            wq_log_var: (B, N, P, 11) 水质预测对数方差
            wq_target: (B, N, P, 11) 水质标签
            bloom_pred: (B, N, 4) 蓝藻分类 logits
            bloom_target: (B, N) 蓝藻标签
        """
        # 1. Gaussian NLL 损失 (含不确定性)
        nll = self.gaussian_nll_loss(wq_pred, wq_log_var, wq_target)  # (B, N, P, 11)
        # 按时间步加权
        nll_per_step = nll.mean(dim=[0, 1, 3])  # (P,)
        nll_loss = (nll_per_step * self.time_weights).sum()

        # 2. Focal Loss (蓝藻分类)
        bloom_flat = bloom_pred.reshape(-1, bloom_pred.size(-1))
        bloom_target_flat = bloom_target.reshape(-1)
        focal = self.focal_loss(bloom_flat, bloom_target_flat, self.focal_gamma)

        # 3. 时间衰减正则
        mse_per_step = ((wq_pred - wq_target) ** 2).mean(dim=[0, 1, 3])
        time_decay = (mse_per_step * torch.arange(
            len(mse_per_step), device=mse_per_step.device, dtype=torch.float32
        )).mean()

        # 总损失
        total = self.lambda1 * nll_loss + self.lambda2 * focal + self.lambda3 * time_decay

        return total, {
            "nll_loss": nll_loss.item(),
            "focal_loss": focal.item(),
            "time_decay": time_decay.item(),
            "total_loss": total.item(),
            "mean_uncertainty": wq_log_var.exp().mean().item(),
        }
