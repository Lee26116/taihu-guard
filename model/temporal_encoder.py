"""
V2 时间编码器 (Temporal Encoder)
Transformer Encoder 替代 Conv1D + GRU
多头自注意力 + 位置编码 + FFN
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalTransformerEncoder(nn.Module):
    """
    V2 时间序列编码器 — Transformer Encoder

    输入: (batch, seq_len, feature_dim) — 单个节点的时间序列
    输出: (batch, hidden_dim) — 时间特征表示

    结构:
      Input Projection → Positional Encoding → Transformer Encoder (N层) → Pooling
    """

    def __init__(self, input_dim=35, hidden_dim=256, num_layers=4,
                 num_heads=8, ff_dim=512, dropout=0.1, max_seq_len=200):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=max_seq_len, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN (更稳定的训练)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        # CLS token (类似 BERT，用专门 token 聚合序列信息)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            h: (batch, hidden_dim) — 聚合的时间特征
        """
        batch_size = x.size(0)

        # 输入投影
        x = self.input_proj(x)  # (B, T, hidden_dim)

        # 在序列头部拼接 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, H)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, H)

        # 位置编码
        x = self.pos_encoder(x)

        # Transformer Encoder
        x = self.transformer(x)  # (B, T+1, H)

        # 取 CLS token 的输出作为序列表示
        cls_out = x[:, 0, :]  # (B, H)

        # 输出投影
        h = self.output_proj(cls_out)  # (B, H)

        return h


class TemporalTransformerSeq(nn.Module):
    """
    返回完整序列输出的 Transformer 编码器
    用于需要逐步预测的场景
    """

    def __init__(self, input_dim=35, hidden_dim=256, num_layers=4,
                 num_heads=8, ff_dim=512, dropout=0.1, max_seq_len=200):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        """
        Input: (B, T, F)
        Output: (B, T, hidden_dim)
        """
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return x


# 保留 V1 兼容接口
class TemporalEncoder(TemporalTransformerEncoder):
    """V2 默认使用 Transformer，保留 V1 类名兼容"""
    pass
