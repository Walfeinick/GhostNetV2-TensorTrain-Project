import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tt_linear import TTLinear


class GhostModule(nn.Module):
    """
    Генерирует «ghost»-признаки: сначала обычная свёртка 1×1 (дорогая),
    затем дешёвая групповая свёртка 3×3 для создания «призрачных» копий.
    """
    def __init__(self, in_channels, out_channels, expansion_ratio=2, use_relu=True):
        super().__init__()

        # Сколько каналов делаем «настоящей» свёрткой 1×1
        primary_channels = math.ceil(out_channels / expansion_ratio)

        # Остаток каналов получаем дешёвой групповой свёрткой
        ghost_channels = primary_channels * (expansion_ratio - 1)

        # Обычная точечная свёртка 1×1 — основная, дорогая часть
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, primary_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(primary_channels),
            nn.ReLU(inplace=True) if use_relu else nn.Identity()
        )

        # Групповая свёртка 3×3 — дешёвая, применяется к каждому каналу отдельно
        # (groups=primary_channels означает depthwise-свёртку)
        self.ghost_conv = nn.Sequential(
            nn.Conv2d(primary_channels, ghost_channels,
                      kernel_size=3, stride=1, padding=1,
                      groups=primary_channels, bias=False),
            nn.BatchNorm2d(ghost_channels),
            nn.ReLU(inplace=True) if use_relu else nn.Identity()
        )

        self.out_channels = out_channels

    def forward(self, x):
        primary_features = self.primary_conv(x)          # [B, primary_channels, H, W]
        ghost_features   = self.ghost_conv(primary_features)  # [B, ghost_channels,   H, W]

        # Склеиваем и обрезаем до нужного числа каналов (на случай нечётного out_channels)
        all_features = torch.cat([primary_features, ghost_features], dim=1)
        return all_features[:, :self.out_channels, :, :]


class GhostBottleneck(nn.Module):
    """
    Классический остаточный блок (ResNet-стиль), но все 1×1-свёртки заменены
    на GhostModule. Структура: расширение → (опциональный downsampling) → сжатие.
    """
    def __init__(self, in_channels, bottleneck_channels, out_channels, stride=1):
        super().__init__()

        # Шаг 1: расширяем число каналов до bottleneck (с ReLU)
        self.expand_ghost = GhostModule(in_channels, bottleneck_channels, use_relu=True)

        # Шаг 2: если stride > 1, уменьшаем пространственный размер (H и W)
        # depthwise-свёртка — дёшево, потому что groups=bottleneck_channels
        self.downsample_dw = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels,
                      kernel_size=3, stride=stride, padding=1,
                      groups=bottleneck_channels, bias=False),
            nn.BatchNorm2d(bottleneck_channels)
        ) if stride > 1 else nn.Identity()

        # Шаг 3: сжимаем обратно до out_channels (без ReLU — перед сложением с residual)
        self.project_ghost = GhostModule(bottleneck_channels, out_channels, use_relu=False)

        # Shortcut: если размерности входа и выхода не совпадают — выравниваем их
        dimensions_match = (in_channels == out_channels) and (stride == 1)
        self.shortcut = nn.Identity() if dimensions_match else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = self.shortcut(x)        # сохраняем «ветку пропуска»

        x = self.expand_ghost(x)           # расширяем каналы
        x = self.downsample_dw(x)          # (опционально) уменьшаем H×W
        x = self.project_ghost(x)          # сжимаем каналы обратно

        return x + residual                # складываем с пропущенной веткой


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