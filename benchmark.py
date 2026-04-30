import os
import time
import torch
import numpy as np

from config import Config, ConfigExpW
from models import TT_GhostNetV2_FER, GhostNetV2_Base, TTCrossLinear
from models.basemodel import GhostNetV2_Base
from models.tt_cross import convert_linear_to_tt_cross


def benchmark_model(model, model_path, cfg, name="Model"):
    device = cfg.DEVICE
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")

    # 1. Параметры и сжатие
    total_params = sum(p.numel() for p in model.parameters())

    try:
        fc_params = sum(p.numel() for p in model.tt_layer.parameters())
        fc_name   = "TTLinear"
    except AttributeError:
        fc_params = sum(p.numel() for p in model.fc.parameters())
        fc_name   = "Linear"

    linear_full = 960 * 128 + 128
    compression = linear_full / fc_params

    print(f"\n[Параметры]")
    print(f"  Всего параметров:          {total_params:,}")
    print(f"  Параметры {fc_name}:       {fc_params:,}")
    print(f"  Параметры Linear(960,128): {linear_full:,}")
    print(f"  Сжатие FC-слоя:            {compression:.1f}x")

    # 2. Размер файла
    size_mb = None
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / 1e6
        print(f"\n[Размер модели]")
        print(f"  Файл .pth: {size_mb:.2f} MB")

    dummy_gpu = torch.randn(1, cfg.IN_CHANNELS, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE).to(device)
    dummy_cpu = torch.randn(1, cfg.IN_CHANNELS, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)

    # 3. Латентность на GPU
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            model(dummy_gpu)

    times_gpu = []
    with torch.no_grad():
        for _ in range(200):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(dummy_gpu)
            torch.cuda.synchronize()
            times_gpu.append(time.perf_counter() - start)

    print(f"\n[Латентность GPU (batch=1)]")
    print(f"  Среднее: {np.mean(times_gpu)*1000:.2f} ms")
    print(f"  Медиана: {np.median(times_gpu)*1000:.2f} ms")
    print(f"  Мин:     {np.min(times_gpu)*1000:.2f} ms")
    print(f"  FPS:     {1/np.mean(times_gpu):.1f}")

    # 4. Латентность на CPU
    model_cpu = model.cpu()
    with torch.no_grad():
        for _ in range(10):
            model_cpu(dummy_cpu)

    times_cpu = []
    with torch.no_grad():
        for _ in range(100):
            start = time.perf_counter()
            model_cpu(dummy_cpu)
            times_cpu.append(time.perf_counter() - start)

    print(f"\n[Латентность CPU (batch=1)]")
    print(f"  Среднее: {np.mean(times_cpu)*1000:.2f} ms")
    print(f"  Медиана: {np.median(times_cpu)*1000:.2f} ms")
    print(f"  FPS:     {1/np.mean(times_cpu):.1f}")

    model.to(device)

    return {
        'total_params': total_params,
        'fc_params':    fc_params,
        'compression':  compression,
        'size_mb':      size_mb,
        'gpu_ms':       np.mean(times_gpu) * 1000,
        'cpu_ms':       np.mean(times_cpu) * 1000,
        'fps_gpu':      1 / np.mean(times_gpu),
        'fps_cpu':      1 / np.mean(times_cpu),
    }


def _print_table(title, tt_r, base_r, tt_cross_r):
    print(f"\n{'='*60}")
    print(f"  ИТОГОВОЕ СРАВНЕНИЕ ({title})")
    print(f"{'='*60}")
    print(f"{'Метрика':<30} {'TT':>10} {'Base':>10} {'TT-Cross':>10}")
    print("-" * 60)
    for label, key, fmt in [
        ('Всего параметров',    'total_params', ','),
        ('Параметры FC-слоя',   'fc_params',    ','),
        ('Размер файла (MB)',   'size_mb',      '.2f'),
        ('Латентность GPU (ms)','gpu_ms',       '.2f'),
        ('Латентность CPU (ms)','cpu_ms',       '.2f'),
        ('FPS GPU',             'fps_gpu',      '.1f'),
        ('FPS CPU',             'fps_cpu',      '.1f'),
    ]:
        print(f"{label:<30} {tt_r[key]:>10{fmt}} {base_r[key]:>10{fmt}} {tt_cross_r[key]:>10{fmt}}")


def _run_benchmark(cfg, suffixes):
    """suffixes = {'tt': 'best_model_tt.pth', 'base': ..., 'tt_cross': ...}"""
    mp = cfg.MODEL_SAVE_PATH

    tt_path  = os.path.join(mp, suffixes['tt'])
    tt_model = TT_GhostNetV2_FER(num_classes=cfg.NUM_CLASSES, dropout=0.3,
                                  in_channels=cfg.IN_CHANNELS).to(cfg.DEVICE)
    tt_model.load_state_dict(torch.load(tt_path, map_location=cfg.DEVICE)['model_state'])
    tt_r = benchmark_model(tt_model, tt_path, cfg, name="TT-GhostNetV2")

    base_path  = os.path.join(mp, suffixes['base'])
    base_model = GhostNetV2_Base(num_classes=cfg.NUM_CLASSES, dropout=0.3,
                                  in_channels=cfg.IN_CHANNELS).to(cfg.DEVICE)
    base_model.load_state_dict(torch.load(base_path, map_location=cfg.DEVICE)['model_state'])
    base_r = benchmark_model(base_model, base_path, cfg, name="GhostNetV2-Base")

    tt_cross_path  = os.path.join(mp, suffixes['tt_cross'])
    tt_cross_model = GhostNetV2_Base(num_classes=cfg.NUM_CLASSES, dropout=0.3,
                                      in_channels=cfg.IN_CHANNELS)
    tt_cross_model.fc = convert_linear_to_tt_cross(tt_cross_model.fc, rank=16)
    tt_cross_model.to(cfg.DEVICE)
    tt_cross_model.load_state_dict(torch.load(tt_cross_path, map_location=cfg.DEVICE)['model_state'])
    tt_cross_r = benchmark_model(tt_cross_model, tt_cross_path, cfg, name="TT-Cross-GhostNetV2")

    return tt_r, base_r, tt_cross_r


def main():
    # FER2013
    tt_r, base_r, tt_cross_r = _run_benchmark(Config, {
        'tt':       'best_model_tt.pth',
        'base':     'best_model_base.pth',
        'tt_cross': 'best_model_tt_cross.pth',
    })
    _print_table('Dataset FER2013', tt_r, base_r, tt_cross_r)

    # ExpW
    tt_r, base_r, tt_cross_r = _run_benchmark(ConfigExpW, {
        'tt':       'best_model_tt_expw.pth',
        'base':     'best_model_base_expw.pth',
        'tt_cross': 'best_model_tt_cross_expw.pth',
    })
    _print_table('Dataset ExpW', tt_r, base_r, tt_cross_r)


if __name__ == '__main__':
    main()