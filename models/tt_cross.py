import torch
import torch.nn as nn
import numpy as np


class TTCrossLinear(nn.Module):
    """
    Полносвязный слой с TT-параметризацией.
    Ядра инициализируются либо случайно, либо через TT-Cross из обученного Linear.

    Размерности:
        in_features  = 960 = 8 × 8 × 15   (n1, n2, n3)
        out_features = 128 = 4 × 4 × 8    (m1, m2, m3)
    """

    # Фиксированное разбиение размерностей
    IN_MODES  = (8, 8, 15)   # n1 × n2 × n3 = 960
    OUT_MODES = (4, 4, 8)    # m1 × m2 × m3 = 128

    def __init__(self, in_features: int = 960, out_features: int = 128, rank: int = 16):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.rank = rank

        r = rank
        # Три ядра TT: форма (r_{k-1}, n_k, m_k, r_k)
        self.core1 = nn.Parameter(torch.empty(1,    8,  4, r))
        self.core2 = nn.Parameter(torch.empty(r,    8,  4, r))
        self.core3 = nn.Parameter(torch.empty(r,   15,  8, 1))
        self.bias  = nn.Parameter(torch.zeros(out_features))

        self._init_random()

    # ─── Инициализация ─────────────────────────────────────────────────────────

    def _init_random(self):
        """Xavier-like инициализация (используется по умолчанию)."""
        for core in [self.core1, self.core2, self.core3]:
            fan = core.shape[1] * core.shape[2]
            std = (2.0 / fan) ** 0.5
            nn.init.normal_(core, mean=0.0, std=std)

    def init_from_linear(self, linear: nn.Linear, tol: float = 1e-3, max_rank: int = None):
        """
        Инициализация ядер через TT-Cross аппроксимацию весов обученного Linear.

        Алгоритм:
            1. Берём матрицу весов W (out × in) из linear
            2. Решейпим в тензор T (n1, n2, n3, m1, m2, m3)
            3. Применяем TT-SVD (HOSVD-based cross) последовательно по модам
            4. Ограничиваем ранг до self.rank
            5. Записываем результат в core1, core2, core3

        Примечание: TT-Cross в полном варианте требует адаптивного выбора опорных
        строк/столбцов (maxvol-алгоритм). Здесь используется TT-SVD как детерминированная
        альтернатива, дающая оптимальное разложение заданного ранга.
        """
        if max_rank is None:
            max_rank = self.rank

        W = linear.weight.data  # (out_features, in_features) = (128, 960)

        # Решейпим W в тензор по модам (n1, n2, n3, m1, m2, m3)
        n1, n2, n3 = self.IN_MODES   # 8, 8, 15
        m1, m2, m3 = self.OUT_MODES  # 4, 4, 8

        # W: (128, 960) → (m1, m2, m3, n1, n2, n3) → (n1, n2, n3, m1, m2, m3)
        T = W.reshape(m1, m2, m3, n1, n2, n3).permute(3, 4, 5, 0, 1, 2)
        # T: (8, 8, 15, 4, 4, 8)

        cores = _tt_svd(T, max_rank=max_rank)
        # cores[k] имеет форму (r_{k-1}, mode_k, r_k)
        # Нам нужна форма (r_{k-1}, n_k, m_k, r_k) — объединяем пары мод

        G1, G2, G3 = _reshape_cores(cores, n1, n2, n3, m1, m2, m3)

        # Обрезаем или дополняем до self.rank
        G1, G2, G3 = _fit_rank(G1, G2, G3, self.rank)

        with torch.no_grad():
            self.core1.copy_(G1)
            self.core2.copy_(G2)
            self.core3.copy_(G3)
            if linear.bias is not None:
                self.bias.copy_(linear.bias.data)

        print(f"TT-Cross инициализация завершена | ранг={self.rank} | "
              f"ошибка аппроксимации: {_approx_error(linear.weight.data, self):.6f}")

    # ─── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = x.reshape(batch, 8, 8, 15)

        # Последовательная контракция по каждому измерению входа
        h = torch.einsum('bijk, ilr -> blrjk', x, self.core1.squeeze(0))
        h = torch.einsum('blrjk, rjms -> blmsk', h, self.core2)
        h = torch.einsum('blmsk, skn -> blmn',   h, self.core3.squeeze(-1))

        return h.reshape(batch, self.out_features) + self.bias

    def extra_repr(self):
        n_tt   = sum(p.numel() for p in [self.core1, self.core2, self.core3])
        n_full = self.in_features * self.out_features
        return (f'in={self.in_features}, out={self.out_features}, '
                f'rank={self.rank} | params: {n_tt} vs {n_full} full '
                f'(compression {n_full/n_tt:.1f}x)')


# ─── TT-SVD ────────────────────────────────────────────────────────────────────

def _tt_svd(T: torch.Tensor, max_rank: int) -> list:
    """
    TT-SVD (Oseledets, 2011) — последовательное SVD-разложение по модам тензора.

    Для тензора T размерности (d1, d2, ..., dd):
        Шаг 1: разворачиваем T в матрицу (d1, d2*...*dd)
        Шаг 2: SVD → обрезаем до max_rank сингулярных значений
        Шаг 3: левая часть → ядро G_k, правая → остаток для следующего шага
        Повторяем для каждой моды

    Возвращает список ядер cores[k] формы (r_{k-1}, d_k, r_k).
    """
    shape = T.shape
    d     = len(shape)
    cores = []
    r_prev = 1
    C = T.reshape(r_prev, -1)  # текущий остаток

    for k in range(d - 1):
        n_k = shape[k]
        C   = C.reshape(r_prev * n_k, -1)

        # SVD
        U, S, Vh = torch.linalg.svd(C, full_matrices=False)

        # Обрезаем до max_rank
        r_k = min(max_rank, S.shape[0])
        U   = U[:, :r_k]
        S   = S[:r_k]
        Vh  = Vh[:r_k, :]

        # Ядро: (r_prev, n_k, r_k)
        G_k = U.reshape(r_prev, n_k, r_k)
        cores.append(G_k)

        # Остаток для следующего шага
        C     = torch.diag(S) @ Vh
        r_prev = r_k

    # Последнее ядро
    cores.append(C.reshape(r_prev, shape[-1], 1))

    return cores


def _reshape_cores(cores: list, n1, n2, n3, m1, m2, m3) -> tuple:
    """
    TT-SVD работает по отдельным модам (n1, n2, n3, m1, m2, m3).
    Нам нужны ядра по парам мод (n_k, m_k).

    Объединяем соседние ядра:
        G1  cores[0] × cores[3]  (мода n1 и мода m1)
        G2  cores[1] × cores[4]  (мода n2 и мода m2)
        G3  cores[2] × cores[5]  (мода n3 и мода m3)

    Результирующие ядра имеют форму (r_{k-1}, n_k, m_k, r_k).
    """
    # cores[0]: (1,  n1, r1)  cores[3]: (r3, m1, r4)
    # cores[1]: (r1, n2, r2)  cores[4]: (r4, m2, r5)
    # cores[2]: (r2, n3, r3)  cores[5]: (r5, m3, 1)

    # Контрактируем пары через промежуточные ранги
    # Для простоты объединяем мод-пары напрямую:

    def merge(Gn, Gm):
        # Gn: (r_l, n_k, r_mid)
        # Gm: (r_mid, m_k, r_r)
        # → (r_l, n_k, m_k, r_r)  через einsum по r_mid
        return torch.einsum('lnr, rms -> lnms', Gn, Gm)

    G1 = merge(cores[0], cores[3])  # (1,  n1, m1, r)
    G2 = merge(cores[1], cores[4])  # (r,  n2, m2, r)
    G3 = merge(cores[2], cores[5])  # (r,  n3, m3, 1)

    return G1, G2, G3


def _fit_rank(G1: torch.Tensor, G2: torch.Tensor, G3: torch.Tensor,
              target_rank: int) -> tuple:
    """
    Обрезает или дополняет нулями ранги ядер до target_rank.
    Это нужно когда SVD дал меньший ранг чем запрошено (маловероятно),
    или больший (обрезаем).
    """
    def clip(G, ax_left, ax_right):
        # Обрезаем по левому и правому ранговым измерениям
        slices = [slice(None)] * G.ndim
        if ax_left is not None:
            slices[ax_left] = slice(0, min(target_rank, G.shape[ax_left]))
        if ax_right is not None:
            slices[ax_right] = slice(0, min(target_rank, G.shape[ax_right]))
        return G[tuple(slices)].contiguous()

    # G1: (1, n1, m1, r) — обрезаем только правый ранг (dim=3)
    G1 = clip(G1, None, 3)
    # G2: (r, n2, m2, r) — обрезаем оба
    G2 = clip(G2, 0, 3)
    # G3: (r, n3, m3, 1) — обрезаем только левый ранг (dim=0)
    G3 = clip(G3, 0, None)

    # Дополняем нулями если ранг получился меньше target_rank
    def pad_to(G, ax_left, ax_right):
        shape = list(G.shape)
        if ax_left is not None and shape[ax_left] < target_rank:
            pad = torch.zeros(*shape[:ax_left], target_rank - shape[ax_left],
                              *shape[ax_left+1:])
            G = torch.cat([G, pad], dim=ax_left)
        if ax_right is not None and shape[ax_right] < target_rank:
            shape = list(G.shape)
            pad = torch.zeros(*shape[:ax_right], target_rank - shape[ax_right],
                              *shape[ax_right+1:])
            G = torch.cat([G, pad], dim=ax_right)
        return G

    G1 = pad_to(G1, None, 3)
    G2 = pad_to(G2, 0,    3)
    G3 = pad_to(G3, 0,    None)

    return G1, G2, G3


def _approx_error(W: torch.Tensor, layer: TTCrossLinear) -> float:
    """Относительная ошибка аппроксимации ||W - W_tt|| / ||W||."""
    with torch.no_grad():
        dummy = torch.eye(layer.in_features)
        W_tt  = layer(dummy) - layer.bias.unsqueeze(0)
        W_tt  = W_tt.T  # (out, in)
        err   = torch.norm(W - W_tt) / torch.norm(W)
    return err.item()


# ─── Функция конвертации ───────────────────────────────────────────────────────

def convert_linear_to_tt_cross(linear: nn.Linear, rank: int = 16,
                                tol: float = 1e-3) -> TTCrossLinear:
    """
    Конвертирует обученный nn.Linear в TTCrossLinear.

    Использование:
        model = GhostNetV2_Base(...)
        checkpoint = torch.load('best_model_base.pth')
        model.load_state_dict(checkpoint['model_state'])

        model.fc = convert_linear_to_tt_cross(model.fc, rank=16)
        # → дообучить несколько эпох
    """
    tt_layer = TTCrossLinear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        rank=rank
    )
    tt_layer.init_from_linear(linear, tol=tol)
    return tt_layer