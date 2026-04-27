import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from config import ConfigExpW, train_transform_full
from data import build_dataloaders_expw
from models.basemodel import GhostNetV2_Base
from utils import set_seed, get_scheduler, run_epoch

AUGMENT_EPOCH = 4


def main():
    set_seed(ConfigExpW.SEED)

    train_loader, val_loader, _ = build_dataloaders_expw(ConfigExpW)

    model = GhostNetV2_Base(num_classes=ConfigExpW.NUM_CLASSES, dropout=0.3).to(ConfigExpW.DEVICE)
    print(f"Модель OK | Параметры: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(),
                            lr=ConfigExpW.LEARNING_RATE,
                            weight_decay=ConfigExpW.WEIGHT_DECAY)
    scheduler = get_scheduler(optimizer, warmup_epochs=5, total_epochs=ConfigExpW.NUM_EPOCHS)
    use_amp   = ConfigExpW.DEVICE.type == 'cuda'
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_acc  = 0.0
    best_path = os.path.join(ConfigExpW.MODEL_SAVE_PATH, 'best_model_base_expw.pth')
    start     = time.time()
    history   = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, ConfigExpW.NUM_EPOCHS + 1):
        print(f'\nЭпоха {epoch}/{ConfigExpW.NUM_EPOCHS}  lr={optimizer.param_groups[0]["lr"]:.2e}')

        if epoch == AUGMENT_EPOCH:
            train_loader.dataset.dataset.transform = train_transform_full
            print(f"  → RandomErasing включён с эпохи {epoch}")

        train_loss, train_acc = run_epoch(model, train_loader, criterion,
                                          optimizer, scaler, ConfigExpW.DEVICE,
                                          use_amp, training=True)
        val_loss, val_acc     = run_epoch(model, val_loader, criterion,
                                          optimizer, scaler, ConfigExpW.DEVICE,
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
                    'num_classes':   ConfigExpW.NUM_CLASSES,
                    'batch_size':    ConfigExpW.BATCH_SIZE,
                    'learning_rate': ConfigExpW.LEARNING_RATE,
                    'weight_decay':  ConfigExpW.WEIGHT_DECAY,
                    'rank':          ConfigExpW.RANK,
                    'image_size':    ConfigExpW.IMAGE_SIZE,
                    'dataset':       'expw',
                    'method':        'base',
                }
            }, best_path)
            print(f'  ✓ Сохранена лучшая модель (acc={best_acc:.2f}%)')

    elapsed = time.time() - start
    print(f'\nОбучение завершено за {elapsed/60:.1f} мин')
    print(f'Лучшая val accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()