import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config, train_transform_full
from data import build_dataloaders
from models import GhostNetV2_Base
from utils import set_seed, get_scheduler, run_epoch


def main():
    set_seed(Config.SEED)

    #Данные
    train_loader, val_loader, _ = build_dataloaders(Config)

    #Модель
    model = GhostNetV2_Base(num_classes=Config.NUM_CLASSES, dropout=0.3).to(Config.DEVICE)
    print(f"Модель OK | Параметры: {sum(p.numel() for p in model.parameters()):,}")

    #Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(),
                            lr=Config.LEARNING_RATE,
                            weight_decay=Config.WEIGHT_DECAY)
    scheduler = get_scheduler(optimizer,
                               warmup_epochs=5,
                               total_epochs=Config.NUM_EPOCHS)
    use_amp = Config.DEVICE.type == 'cuda'
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

    #Цикл обучения
    best_acc  = 0.0 
    best_path = os.path.join(Config.MODEL_SAVE_PATH, 'best_model_base.pth')
    start     = time.time()
    history   = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    AUGMENT_EPOCH = 7

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        print(f'\nЭпоха {epoch}/{Config.NUM_EPOCHS}  lr={optimizer.param_groups[0]["lr"]:.2e}')

        if epoch == AUGMENT_EPOCH:
            train_loader.dataset.dataset.transform = train_transform_full
            print(f"RandomErasing включён с эпохи {epoch}")

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

        print(f'  Train: loss: {train_loss:.4f}  acc: {train_acc:.2f}%')
        print(f'  Val:   loss: {val_loss:.4f}  acc: {val_acc:.2f}%')

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
                    'learning_rate': Config.LEARNING_RATE,
                    'weight_decay':  Config.WEIGHT_DECAY,
                    'rank':          Config.RANK,
                    'image_size':    Config.IMAGE_SIZE,
                }
            }, best_path)
            print(f'Сохранена лучшая модель (acc={best_acc:.2f}%)')

    elapsed = time.time() - start
    print(f'\nОбучение завершено за {elapsed/60:.1f} мин')
    print(f'Лучшая val accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()