"""
时间编码器 (Temporal Encoder)
Conv1D × 2 + GRU，对每个节点的时间序列提取时间模式
"""

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """
    时间序列编码器

    输入: (batch, seq_len, feature_dim) — 单个节点的时间序列
    输出: (batch, hidden_dim) — 时间特征表示

    结构:
      1D Conv (kernel=3) × 2 → GRU → 取最后隐状态
    """

    def __init__(self, input_dim=25, hidden_dim=64, num_layers=1, dropout=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 1D 卷积层 (提取局部时间模式)
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # GRU 层 (捕获长期时间依赖)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Layer Norm
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) 或 (batch*num_nodes, seq_len, input_dim)

        Returns:
            h: (batch, hidden_dim) — 时间编码特征
        """
        # Conv1D 输入需要 (batch, channels, seq_len)
        x_conv = x.permute(0, 2, 1)  # (B, F, T)

        # 两层 1D 卷积
        x_conv = self.relu(self.bn1(self.conv1(x_conv)))
        x_conv = self.dropout(x_conv)
        x_conv = self.relu(self.bn2(self.conv2(x_conv)))
        x_conv = self.dropout(x_conv)

        # 转回 (batch, seq_len, hidden_dim)
        x_conv = x_conv.permute(0, 2, 1)

        # GRU
        gru_out, h_n = self.gru(x_conv)
        # h_n: (num_layers, batch, hidden_dim)

        # 取最后一层最后时间步的隐状态
        h = h_n[-1]  # (batch, hidden_dim)

        # Layer Norm
        h = self.layer_norm(h)

        return h


class TemporalEncoderSeq(nn.Module):
    """
    返回完整序列的时间编码器版本
    用于需要逐步预测的场景
    """

    def __init__(self, input_dim=25, hidden_dim=64, num_layers=1, dropout=0.1):
        super().__init__()

        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, x):
        """
        返回完整 GRU 序列输出
        Input: (B, T, F)
        Output: (B, T, hidden_dim)
        """
        x_conv = x.permute(0, 2, 1)
        x_conv = self.relu(self.bn1(self.conv1(x_conv)))
        x_conv = self.dropout(x_conv)
        x_conv = self.relu(self.bn2(self.conv2(x_conv)))
        x_conv = self.dropout(x_conv)
        x_conv = x_conv.permute(0, 2, 1)

        gru_out, _ = self.gru(x_conv)
        return gru_out
