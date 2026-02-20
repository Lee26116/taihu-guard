"""
空间图注意力层 (Spatial GAT)
基于 GAT (Graph Attention Network) 捕获站点间的空间依赖关系
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    单头图注意力层

    实现标准 GAT 注意力机制:
    e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
    α_ij = softmax_j(e_ij)
    h_i' = σ(Σ_j α_ij Wh_j)
    """

    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        # 线性变换
        self.W = nn.Linear(in_features, out_features, bias=False)
        # 注意力参数
        self.a = nn.Linear(2 * out_features, 1, bias=False)

        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.weight)

    def forward(self, h, edge_index, edge_weight=None):
        """
        Args:
            h: (num_nodes, in_features) 节点特征
            edge_index: (2, num_edges) 边索引 [src, dst]
            edge_weight: (num_edges,) 边权重（可选）

        Returns:
            h_prime: (num_nodes, out_features) 更新后的节点特征
        """
        num_nodes = h.size(0)

        # 线性变换
        Wh = self.W(h)  # (N, out_features)

        # 计算注意力系数
        src, dst = edge_index[0], edge_index[1]
        Wh_src = Wh[src]  # (E, out_features)
        Wh_dst = Wh[dst]  # (E, out_features)

        # 拼接 [Wh_i || Wh_j]
        edge_feats = torch.cat([Wh_src, Wh_dst], dim=-1)  # (E, 2*out_features)
        e = self.leaky_relu(self.a(edge_feats)).squeeze(-1)  # (E,)

        # 如果有边权重，乘上去
        if edge_weight is not None:
            e = e * edge_weight

        # Softmax 归一化（按目标节点）
        attention = self._sparse_softmax(e, dst, num_nodes)
        attention = self.dropout(attention)

        # 加权聚合
        h_prime = torch.zeros(num_nodes, self.out_features, device=h.device)
        h_prime.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.out_features),
                             attention.unsqueeze(-1) * Wh_src)

        return h_prime

    def _sparse_softmax(self, values, indices, num_nodes):
        """稀疏 softmax：对同一目标节点的边做 softmax"""
        # 数值稳定性
        max_vals = torch.zeros(num_nodes, device=values.device)
        max_vals.scatter_reduce_(0, indices, values, reduce="amax", include_self=True)
        values = values - max_vals[indices]

        exp_vals = torch.exp(values)
        sum_exp = torch.zeros(num_nodes, device=values.device)
        sum_exp.scatter_add_(0, indices, exp_vals)

        return exp_vals / (sum_exp[indices] + 1e-8)


class MultiHeadGAT(nn.Module):
    """
    多头图注意力层

    将多个 GAT 头的输出拼接（或取均值）
    """

    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1,
                 concat=True, residual=True):
        super().__init__()

        self.num_heads = num_heads
        self.concat = concat
        self.residual = residual

        if concat:
            assert out_features % num_heads == 0
            head_dim = out_features // num_heads
        else:
            head_dim = out_features

        self.heads = nn.ModuleList([
            GraphAttentionLayer(in_features, head_dim, dropout=dropout)
            for _ in range(num_heads)
        ])

        if residual:
            if in_features != out_features:
                self.residual_proj = nn.Linear(in_features, out_features)
            else:
                self.residual_proj = nn.Identity()

        self.layer_norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_weight=None):
        """
        Args:
            h: (N, in_features)
            edge_index: (2, E)
            edge_weight: (E,)

        Returns:
            (N, out_features)
        """
        head_outputs = [head(h, edge_index, edge_weight) for head in self.heads]

        if self.concat:
            out = torch.cat(head_outputs, dim=-1)  # (N, num_heads * head_dim)
        else:
            out = torch.mean(torch.stack(head_outputs), dim=0)  # (N, head_dim)

        out = F.elu(out)

        # 残差连接
        if self.residual:
            out = out + self.residual_proj(h)

        out = self.layer_norm(out)
        out = self.dropout(out)

        return out


class SpatialGATBlock(nn.Module):
    """
    V2 空间 GAT 模块: N 层 Multi-Head GAT (默认 4 层)
    支持可配置深度
    """

    def __init__(self, in_features=256, hidden_dim=256, out_features=256,
                 num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(num_layers):
            in_f = in_features if i == 0 else hidden_dim
            out_f = out_features if i == num_layers - 1 else hidden_dim

            self.layers.append(MultiHeadGAT(
                in_features=in_f,
                out_features=out_f,
                num_heads=num_heads,
                dropout=dropout,
                concat=True,
                residual=True
            ))

    def forward(self, h, edge_index, edge_weight=None):
        """
        N 层 GAT

        Input: (N, in_features)
        Output: (N, out_features)
        """
        for layer in self.layers:
            h = layer(h, edge_index, edge_weight)
        return h
