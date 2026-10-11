import torch
import torch.nn as nn


class GraphEncoder(nn.Module):
    """Mean-aggregation message passing over the supplier-consumer graph (S2V-style)."""

    def __init__(self, node_feature_size, embed_dim=64, T=3):
        super().__init__()
        self.T = T
        self.inp = nn.Linear(node_feature_size, embed_dim)
        self.msg = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(T)])
        self.upd = nn.ModuleList([nn.Linear(2 * embed_dim, embed_dim) for _ in range(T)])

    def forward(self, x, adj):
        # x: (B, n, F), adj: (n, n)
        h0 = self.inp(x).clamp(0)
        h = h0
        for t in range(self.T):
            m = self.msg[t](torch.matmul(adj, h)).clamp(0)
            h = self.upd[t](torch.cat((h0, m), dim=-1)).clamp(0)
        return h


class OptionNetwork(nn.Module):
    """Higher layer, action-in: Q^I(s, [t_r, B_r, b])."""

    def __init__(self, node_feature_size, embed_dim=64):
        super().__init__()
        self.enc = GraphEncoder(node_feature_size, embed_dim, T=3)
        self.opt = nn.Linear(3, embed_dim)
        self.head = nn.Sequential(nn.Linear(2 * embed_dim, embed_dim), nn.ReLU(),
                                  nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))

    def forward(self, x, option, adj):
        g = self.enc(x, adj).mean(dim=1)                    # graph embedding (B, d)
        o = self.opt(option).clamp(0)
        return self.head(torch.cat((g, o), dim=-1)).squeeze(-1)


class ActionNetwork(nn.Module):
    """Lower layer, action-out: Q^II(s, u) for every node u, plus the null action at index n."""

    def __init__(self, node_feature_size, embed_dim=64):
        super().__init__()
        self.enc = GraphEncoder(node_feature_size, embed_dim, T=3)
        self.node = nn.Sequential(nn.Linear(2 * embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))
        self.null = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))

    def forward(self, x, adj, mask=None):
        h = self.enc(x, adj)                                # (B, n, d)
        g = h.mean(dim=1, keepdim=True)
        q = self.node(torch.cat((h, g.expand_as(h)), dim=-1)).squeeze(-1)    # (B, n)
        q = torch.cat((q, self.null(g).squeeze(-1)), dim=1)                 # (B, n + 1)
        if mask is not None:
            q = q.masked_fill(mask, -1e9)
        return q
