import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tt_linear import TTLinear


class GhostModule(nn.Module):
    def __init__(self, inp, oup, ratio=2, relu=True):
        super().__init__()
        init_channels = math.ceil(oup / ratio)
        new_channels  = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, 1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 3, 1, 1,
                      groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )
        self.oup = oup

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, stride=1):
        super().__init__()
        self.ghost1 = GhostModule(in_chs, mid_chs)
        self.dw = nn.Sequential(
            nn.Conv2d(mid_chs, mid_chs, 3, stride, 1,
                      groups=mid_chs, bias=False),
            nn.BatchNorm2d(mid_chs)
        ) if stride > 1 else nn.Identity()
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        needs_proj = (in_chs != out_chs) or (stride != 1)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, 1, stride, bias=False),
            nn.BatchNorm2d(out_chs)
        ) if needs_proj else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.ghost1(x)
        x = self.dw(x)
        x = self.ghost2(x)
        return x + residual


class TT_GhostNetV2_FER(nn.Module):
    def __init__(self, num_classes=7, dropout=0.3):
        super().__init__()

        # Stem: 1 в 16, карта остаётся 48x48
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Даунсэмплинг: 48 в 24 в 12 в 6
        self.stage1 = GhostBottleneck(16,  48,  24, stride=2)
        self.stage2 = GhostBottleneck(24,  72,  40, stride=2)
        self.stage3 = GhostBottleneck(40, 120,  80, stride=2)

        # Блоки на 6x6 — уменьшены mid_chs чтобы не жечь память
        self.blocks = nn.Sequential(
            GhostBottleneck(80,  160,  80, stride=1),  # было 240
            GhostBottleneck(80,  160,  80, stride=1),  # было 200
            GhostBottleneck(80,  240, 112, stride=1),  # было 480
            GhostBottleneck(112, 336, 112, stride=1),  # было 672
            GhostBottleneck(112, 480, 160, stride=1),  # было 960
        )

        # Head: 160 в 960, глобальный пул в вектор 960
        self.head = nn.Sequential(
            nn.Conv2d(160, 960, 1, bias=False),
            nn.BatchNorm2d(960),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        # TT-слой: 960 в 128 со сжатием ~11x
        self.tt_layer   = TTLinear(960, 128, rank=16)
        self.bn_tt      = nn.BatchNorm1d(128)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)              # [B,  16, 48, 48]
        x = self.stage1(x)            # [B,  24, 24, 24]
        x = self.stage2(x)            # [B,  40, 12, 12]
        x = self.stage3(x)            # [B,  80,  6,  6]
        x = self.blocks(x)            # [B, 160,  6,  6]
        x = self.head(x).flatten(1)   # [B, 960]
        x = self.tt_layer(x)          # [B, 128]
        x = self.bn_tt(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.classifier(x)        # [B,   7]
        return x