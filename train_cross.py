"""
train_tt_cross.py — Вариант A TT-Cross:

    1. Загружаем обученную GhostNetV2_Base
    2. Конвертируем Linear(960→128) → TTCrossLinear через TT-SVD аппроксимацию
    3. Дообучаем несколько эпох чтобы восстановить точность
    4. Сохраняем лучшую модель

Логика:
    - Backbone (GhostModule, stages, head) заморожен на первых FREEZE_EPOCHS эпохах
    - Затем размораживаем всю сеть для joint fine-tuning
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config, train_transform_full
from data import build_dataloaders
from models.basemodel import GhostNetV2_Base
from models.tt_cross import TTCrossLinear, convert_linear_to_tt_cross
from utils import set_seed, get_scheduler, run_epoch


# ─── Параметры дообучения ──────────────────────────────────────────────────────
FINETUNE_EPOCHS = 20    # эпох дообучения после конвертации
FREEZE_EPOCHS   = 5     # первые N эпох обучаем только TT-слой
TT_RANK         = 16    # ранг TT-Cross (совпадает с TT-from-scratch для честного сравнения)
LR_TT           = 5e-4  # lr для TT-слоя
LR_BACKBONE     = 1e-4  # lr для backbone при размораживании


def build_tt_cross_model(base_checkpoint_path: str) -> GhostNetV2_Base:
    """
    Загружает обученную базовую модель и конвертирует FC-слой в TTCrossLinear.
    """
    # Загружаем базовую модель
    model = GhostNetV2_Base(num_classes=Config.NUM_CLASSES, dropout=0.3)
    checkpoint = torch.load(base_checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    print(f"Базовая модель загружена | val_acc={checkpoint['val_acc']:.2f}%")

    # Конвертируем FC → TTCrossLinear
    model.fc = convert_linear_to_tt_cross(model.fc, rank=TT_RANK)
    model.to(Config.DEVICE)

    total = sum(p.numel() for p in model.parameters())
    tt_p  = sum(p.numel() for p in model.fc.parameters())
    print(f"После конвертации | Всего параметров: {total:,} | TT-слой: {tt_p:,}")

    return model


def freeze_backbone(model: GhostNetV2_Base):
    """Замораживаем всё кроме fc (TT-слой) и classifier."""
    for name, param in model.named_parameters():
        if 'fc' not in name and 'bn_fc' not in name and 'classifier' not in name:
            param.requires_grad = False
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Backbone заморожен | {frozen:,} параметров не обучается")


def unfreeze_all(model: GhostNetV2_Base):
    """Размораживаем все параметры для joint fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    print("Backbone разморожен — joint fine-tuning")


def main():
    set_seed(Config.SEED)

    # ─── Данные ────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = build_dataloaders(Config)

    # ─── Модель: загрузка + конвертация ────────────────────────────────────────
    base_path = os.path.join(Config.MODEL_SAVE_PATH, 'best_model_base.pth')
    assert os.path.exists(base_path), (
        f"Не найден файл базовой модели: {base_path}\n"
        f"Сначала запустите train_base.py"
    )
    model = build_tt_cross_model(base_path)

    # ─── Loss ──────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ─── Фаза 1: замораживаем backbone, обучаем только TT-слой ────────────────
    freeze_backbone(model)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_TT,
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = get_scheduler(optimizer,
                               warmup_epochs=2,
                               total_epochs=FINETUNE_EPOCHS)
    use_amp = Config.DEVICE.type == 'cuda'
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_acc  = 0.0
    best_path = os.path.join(Config.MODEL_SAVE_PATH, 'best_model_tt_cross.pth')
    start     = time.time()
    history   = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    AUGMENT_EPOCH = 3  # раньше чем при обучении с нуля — модель уже знает базу

    for epoch in range(1, FINETUNE_EPOCHS + 1):

        # Размораживаем backbone после FREEZE_EPOCHS эпох
        if epoch == FREEZE_EPOCHS + 1:
            unfreeze_all(model)
            # Пересоздаём optimizer с разными lr для backbone и TT-слоя
            optimizer = optim.AdamW([
                {'params': model.fc.parameters(),         'lr': LR_TT},
                {'params': model.bn_fc.parameters(),      'lr': LR_TT},
                {'params': model.classifier.parameters(), 'lr': LR_TT},
                {'params': [p for n, p in model.named_parameters()
                            if 'fc' not in n and 'bn_fc' not in n
                            and 'classifier' not in n],   'lr': LR_BACKBONE},
            ], weight_decay=Config.WEIGHT_DECAY)
            scheduler = get_scheduler(optimizer,
                                       warmup_epochs=1,
                                       total_epochs=FINETUNE_EPOCHS - FREEZE_EPOCHS)
            scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
            print(f"Эпоха {epoch}: backbone разморожен, lr_tt={LR_TT}, lr_backbone={LR_BACKBONE}")

        # Прогрессивная аугментация
        if epoch == AUGMENT_EPOCH:
            train_loader.dataset.dataset.transform = train_transform_full
            print(f"  → RandomErasing включён с эпохи {epoch}")

        print(f'\nЭпоха {epoch}/{FINETUNE_EPOCHS}  '
              f'lr={optimizer.param_groups[0]["lr"]:.2e}')

        train_loss, train_acc = run_epoch(model, train_loader, criterion,
                                          optimizer, scaler, Config.DEVICE,
                                          use_amp, training=True)
        val_loss, val_acc     = run_epoch(model, val_loader, criterion,
                                          optimizer, scaler, Config.DEVICE,
                                          use_amp, training=False)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'  Train → loss: {train_loss:.4f}  acc: {train_acc:.2f}%')
        print(f'  Val   → loss: {val_loss:.4f}  acc: {val_acc:.2f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_acc':     best_acc,
                'config': {
                    'num_classes':   Config.NUM_CLASSES,
                    'batch_size':    Config.BATCH_SIZE,
                    'learning_rate': LR_TT,
                    'weight_decay':  Config.WEIGHT_DECAY,
                    'rank':          TT_RANK,
                    'image_size':    Config.IMAGE_SIZE,
                    'method':        'tt_cross',
                }
            }, best_path)
            print(f'  ✓ Сохранена лучшая модель (acc={best_acc:.2f}%)')

    elapsed = time.time() - start
    print(f'\nДообучение завершено за {elapsed/60:.1f} мин')
    print(f'Лучшая val accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()