import torch
import torch.nn as nn


class TTLinear(nn.Module):
    def __init__(self, in_features=960, out_features=128, rank=16, in_channels = 1):
        super().__init__()
        self.in_features  = in_features   # 8 * 8 * 15
        self.out_features = out_features  # 4 * 4 * 8
        self.rank = rank

        # Форма входа:  (n1, n2, n3) = (8,  8,  15)
        # Форма выхода: (m1, m2, m3) = (4,  4,   8)
        # Ядра: G_k имеет форму (r_{k-1}, n_k, m_k, r_k)
        self.core1 = nn.Parameter(torch.empty(1,    8,  4, rank))
        self.core2 = nn.Parameter(torch.empty(rank, 8,  4, rank))
        self.core3 = nn.Parameter(torch.empty(rank, 15, 8, 1   ))
        self.bias  = nn.Parameter(torch.zeros(out_features))

        self._init_cores()

    def _init_cores(self):
        """Xavier-инициализация с учётом эффективного fan_in/fan_out"""
        for core in [self.core1, self.core2, self.core3]:
            fan = core.shape[1] * core.shape[2]
            std = (2.0 / fan) ** 0.5
            nn.init.normal_(core, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = x.reshape(batch, 8, 8, 15)  # (b, n1, n2, n3)

        # Шаг 1: свёртка по n1=8, core1: (1, n1=8, m1=4, r)
        h = torch.einsum('bijk, ilr -> blrjk', x, self.core1.squeeze(0))
        # h: (b, m1=4, r=rank, n2=8, n3=15)

        # Шаг 2: свёртка по n2=8, core2: (r, n2=8, m2=4, s)
        h = torch.einsum('blrjk, rjms -> blmsk', h, self.core2)
        # h: (b, m1=4, m2=4, s=rank, n3=15)

        # Шаг 3: свёртка по n3=15, core3: (s, n3=15, m3=8, 1)
        h = torch.einsum('blmsk, skn -> blmn', h, self.core3.squeeze(-1))
        # h: (b, m1=4, m2=4, m3=8)

        return h.reshape(batch, self.out_features) + self.bias

    def extra_repr(self):
        n_tt   = sum(p.numel() for p in [self.core1, self.core2, self.core3])
        n_full = self.in_features * self.out_features
        return (f'in={self.in_features}, out={self.out_features}, '
                f'rank={self.rank} | params: {n_tt} vs {n_full} full '
                f'(compression {n_full/n_tt:.1f}x)')