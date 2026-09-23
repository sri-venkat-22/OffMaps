"""Dilated TCN + 1 GRU + 3 heads. ~180K params, the AI Speed & Vibration Filter.

Heads: (a) forward displacement over 1 s with log-variance (heteroscedastic ->
fusable), (b) lateral slip residual, (c) 4-class motion. Input (B, 9, 20).
"""
from __future__ import annotations
import torch
import torch.nn as nn
from model.features import C, WIN


class TCNBlock(nn.Module):
    def __init__(self, ch, k, d):
        super().__init__()
        pad = (k - 1) * d // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, k, padding=pad, dilation=d), nn.ReLU(), nn.BatchNorm1d(ch),
            nn.Conv1d(ch, ch, k, padding=pad, dilation=d), nn.ReLU(), nn.BatchNorm1d(ch))

    def forward(self, x): return x + self.net(x)   # residual


class SpeedNet(nn.Module):
    def __init__(self, ch=64, k=3):
        super().__init__()
        self.inp = nn.Conv1d(C, ch, 1)
        self.tcn = nn.Sequential(*[TCNBlock(ch, k, 2 ** i) for i in range(5)])
        self.gru = nn.GRU(ch, ch, batch_first=True)
        self.disp = nn.Linear(ch, 2)     # [mu_raw, logvar] displacement over 1 s
        self.slip = nn.Linear(ch, 1)
        self.cls = nn.Linear(ch, 4)

    def forward(self, x):                # x: (B, C, T)
        h = self.tcn(self.inp(x))        # (B, ch, T)
        _, hn = self.gru(h.transpose(1, 2))   # (1, B, ch)
        z = hn[-1]
        d = self.disp(z)
        mu = torch.nn.functional.softplus(d[:, 0])       # displacement >= 0
        logvar = torch.clamp(d[:, 1], -8, 6)
        return mu, logvar, self.slip(z).squeeze(-1), self.cls(z)


def count_params(m): return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    m = SpeedNet()
    x = torch.randn(4, C, WIN)
    mu, lv, slip, cls = m(x)
    assert mu.shape == (4,) and cls.shape == (4, 4)
    print(f"tcn ok: params={count_params(m):,} out mu{tuple(mu.shape)} cls{tuple(cls.shape)}")
