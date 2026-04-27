import os
import time
import torch
import numpy as np

from config import Config, ConfigExpW
from models import TT_GhostNetV2_FER, GhostNetV2_Base, TTCrossLinear

from models.basemodel import GhostNetV2_Base
from models.tt_cross import convert_linear_to_tt_cross


def benchmark_model(model, model_path, device, name="Model"):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")

    #1. Параметры и сжатие
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
    print(f"  Всего параметров:        {total_params:,}")
    print(f"  Параметры {fc_name}:     {fc_params:,}")
    print(f"  Параметры Linear(960,128): {linear_full:,}")
    print(f"  Сжатие FC-слоя:          {compression:.1f}x")

    #2. Размер файла
    size_mb = None
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / 1e6
        print(f"\n[Размер модели]")
        print(f"  Файл .pth:               {size_mb:.2f} MB")

    #3. Латентность на GPU
    model.eval()
    dummy = torch.randn(1, 1, 48, 48).to(device)

    with torch.no_grad():
        for _ in range(20):
            model(dummy)

    times_gpu = []
    with torch.no_grad():
        for _ in range(200):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(dummy)
            torch.cuda.synchronize()
            times_gpu.append(time.perf_counter() - start)

    print(f"\n[Латентность GPU (batch=1)]")
    print(f"  Среднее:   {np.mean(times_gpu)*1000:.2f} ms")
    print(f"  Медиана:   {np.median(times_gpu)*1000:.2f} ms")
    print(f"  Мин:       {np.min(times_gpu)*1000:.2f} ms")
    print(f"  FPS:       {1/np.mean(times_gpu):.1f}")

    #4. Латентность на CPU
    model_cpu = model.cpu()
    dummy_cpu = torch.randn(1, 1, 48, 48)

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
    print(f"  Среднее:   {np.mean(times_cpu)*1000:.2f} ms")
    print(f"  Медиана:   {np.median(times_cpu)*1000:.2f} ms")
    print(f"  FPS:       {1/np.mean(times_cpu):.1f}")

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


def main():
    #TT-модель
    tt_path  = os.path.join(Config.MODEL_SAVE_PATH, 'best_model.pth')
    tt_model = TT_GhostNetV2_FER(num_classes=Config.NUM_CLASSES, dropout=0.3).to(Config.DEVICE)
    tt_ckpt  = torch.load(tt_path, map_location=Config.DEVICE)
    tt_model.load_state_dict(tt_ckpt['model_state'])

    tt_results = benchmark_model(tt_model, tt_path, Config.DEVICE, name="TT-GhostNetV2")


    #Базовая модель
    base_path  = os.path.join(Config.MODEL_SAVE_PATH, 'best_model_base.pth')
    base_model = GhostNetV2_Base(num_classes=Config.NUM_CLASSES, dropout=0.3).to(Config.DEVICE)
    base_ckpt  = torch.load(base_path, map_location=Config.DEVICE)
    base_model.load_state_dict(base_ckpt['model_state'])

    base_results = benchmark_model(base_model, base_path, Config.DEVICE, name="GhostNetV2-Base")


    #TT-Cross-модель
    tt_cross_path  = os.path.join(Config.MODEL_SAVE_PATH, 'best_model_tt_cross.pth')
    tt_cross_model = GhostNetV2_Base(num_classes=Config.NUM_CLASSES, dropout=0.3)
    tt_cross_model.fc = convert_linear_to_tt_cross(tt_cross_model.fc, rank=16)
    tt_cross_model.to(Config.DEVICE)
    tt_cross_ckpt  = torch.load(tt_cross_path, map_location=Config.DEVICE)
    tt_cross_model.load_state_dict(tt_cross_ckpt['model_state'])

    tt_cross_results = benchmark_model(tt_cross_model, tt_cross_path, Config.DEVICE, name="TT-Cross-GhostNetV2")


    #Итоговая таблица
    print(f"\n{'='*60}")
    print("  ИТОГОВОЕ СРАВНЕНИЕ Dataset FER2013")
    print(f"{'='*60}")
    print(f"{'Метрика':<30} {'TT':>10} {'Base':>10} {'TT-Cross':>10}")
    print("-" * 60)
    print(f"{'Всего параметров':<30} {tt_results['total_params']:>10,} {base_results['total_params']:>10,} {tt_cross_results['total_params']:>10,}")
    print(f"{'Параметры FC-слоя':<30} {tt_results['fc_params']:>10,} {base_results['fc_params']:>10,} {tt_cross_results['fc_params']:>10,}")
    print(f"{'Размер файла (MB)':<30} {tt_results['size_mb']:>10.2f} {base_results['size_mb']:>10.2f} {tt_cross_results['size_mb']:>10.2f}")
    print(f"{'Латентность GPU (ms)':<30} {tt_results['gpu_ms']:>10.2f} {base_results['gpu_ms']:>10.2f} {tt_cross_results['gpu_ms']:>10.2f}")
    print(f"{'Латентность CPU (ms)':<30} {tt_results['cpu_ms']:>10.2f} {base_results['cpu_ms']:>10.2f} {tt_cross_results['cpu_ms']:>10.2f}")
    print(f"{'FPS GPU':<30} {tt_results['fps_gpu']:>10.1f} {base_results['fps_gpu']:>10.1f} {tt_cross_results['fps_gpu']:>10.1f}")
    print(f"{'FPS CPU':<30} {tt_results['fps_cpu']:>10.1f} {base_results['fps_cpu']:>10.1f} {tt_cross_results['fps_cpu']:>10.1f}")


#ExpW benchmark

#TT-модель
    tt_path  = os.path.join(ConfigExpW.MODEL_SAVE_PATH, 'best_model_tt_expw.pth')
    tt_model = TT_GhostNetV2_FER(num_classes=ConfigExpW.NUM_CLASSES, dropout=0.3).to(ConfigExpW.DEVICE)
    tt_ckpt  = torch.load(tt_path, map_location=ConfigExpW.DEVICE)
    tt_model.load_state_dict(tt_ckpt['model_state'])

    tt_results = benchmark_model(tt_model, tt_path, ConfigExpW.DEVICE, name="TT-GhostNetV2")


    #Базовая модель
    base_path  = os.path.join(ConfigExpW.MODEL_SAVE_PATH, 'best_model_base.pth')
    base_model = GhostNetV2_Base(num_classes=ConfigExpW.NUM_CLASSES, dropout=0.3).to(ConfigExpW.DEVICE)
    base_ckpt  = torch.load(base_path, map_location=ConfigExpW.DEVICE)
    base_model.load_state_dict(base_ckpt['model_state'])

    base_results = benchmark_model(base_model, base_path, ConfigExpW.DEVICE, name="GhostNetV2-Base")


    #TT-Cross-модель
    tt_cross_path  = os.path.join(ConfigExpW.MODEL_SAVE_PATH, 'best_model_tt_cross.pth') # TODO: add "_expw" when tt-cross model be ready
    tt_cross_model = GhostNetV2_Base(num_classes=ConfigExpW.NUM_CLASSES, dropout=0.3)
    tt_cross_model.fc = convert_linear_to_tt_cross(tt_cross_model.fc, rank=16)
    tt_cross_model.to(ConfigExpW.DEVICE)
    tt_cross_ckpt  = torch.load(tt_cross_path, map_location=ConfigExpW.DEVICE)
    tt_cross_model.load_state_dict(tt_cross_ckpt['model_state'])

    tt_cross_results = benchmark_model(tt_cross_model, tt_cross_path, ConfigExpW.DEVICE, name="TT-Cross-GhostNetV2")


    #Итоговая таблица
    print(f"\n{'='*60}")
    print("  ИТОГОВОЕ СРАВНЕНИЕ (Dataset ExpW)")
    print(f"{'='*60}")
    print(f"{'Метрика':<30} {'TT':>10} {'Base':>10} {'TT-Cross':>10}")
    print("-" * 60)
    print(f"{'Всего параметров':<30} {tt_results['total_params']:>10,} {base_results['total_params']:>10,} {tt_cross_results['total_params']:>10,}")
    print(f"{'Параметры FC-слоя':<30} {tt_results['fc_params']:>10,} {base_results['fc_params']:>10,} {tt_cross_results['fc_params']:>10,}")
    print(f"{'Размер файла (MB)':<30} {tt_results['size_mb']:>10.2f} {base_results['size_mb']:>10.2f} {tt_cross_results['size_mb']:>10.2f}")
    print(f"{'Латентность GPU (ms)':<30} {tt_results['gpu_ms']:>10.2f} {base_results['gpu_ms']:>10.2f} {tt_cross_results['gpu_ms']:>10.2f}")
    print(f"{'Латентность CPU (ms)':<30} {tt_results['cpu_ms']:>10.2f} {base_results['cpu_ms']:>10.2f} {tt_cross_results['cpu_ms']:>10.2f}")
    print(f"{'FPS GPU':<30} {tt_results['fps_gpu']:>10.1f} {base_results['fps_gpu']:>10.1f} {tt_cross_results['fps_gpu']:>10.1f}")
    print(f"{'FPS CPU':<30} {tt_results['fps_cpu']:>10.1f} {base_results['fps_cpu']:>10.1f} {tt_cross_results['fps_cpu']:>10.1f}")

if __name__ == '__main__':
    main()