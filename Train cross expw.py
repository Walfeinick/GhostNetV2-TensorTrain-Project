import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from config import ConfigExpW, train_transform_full
from data import build_dataloaders_expw
from models.basemodel import GhostNetV2_Base
from models.tt_cross import convert_linear_to_tt_cross
from utils import set_seed, get_scheduler, run_epoch

FINETUNE_EPOCHS = 20
FREEZE_EPOCHS   = 5
TT_RANK         = 16
LR_TT           = 5e-4
LR_BACKBONE     = 1e-4
AUGMENT_EPOCH   = 4


def freeze_backbone(model):
    for name, param in model.named_parameters():
        if 'fc' not in name and 'bn_fc' not in name and 'classifier' not in name:
            param.requires_grad = False
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Backbone заморожен | {frozen:,} параметров не обновляется")


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
    print("Backbone разморожен — joint fine-tuning")


def main():
    set_seed(ConfigExpW.SEED)

    train_loader, val_loader, _ = build_dataloaders_expw(ConfigExpW)

    # Загружаем базовую модель обученную на ExpW
    base_path = os.path.join(ConfigExpW.MODEL_SAVE_PATH, 'best_model_base_expw.pth')
    assert os.path.exists(base_path), (
        f"Не найден файл: {base_path}\n"
        f"Сначала запустите train_base_expw.py"
    )

    model = GhostNetV2_Base(num_classes=ConfigExpW.NUM_CLASSES, dropout=0.3)
    checkpoint = torch.load(base_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    print(f"Базовая модель загружена | val_acc={checkpoint['val_acc']:.2f}%")

    model.fc = convert_linear_to_tt_cross(model.fc, rank=TT_RANK)
    model.to(ConfigExpW.DEVICE)
    print(f"Модель OK | Параметры: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    freeze_backbone(model)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_TT, weight_decay=ConfigExpW.WEIGHT_DECAY
    )
    scheduler = get_scheduler(optimizer, warmup_epochs=2, total_epochs=FINETUNE_EPOCHS)
    use_amp   = ConfigExpW.DEVICE.type == 'cuda'
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_acc  = 0.0
    best_path = os.path.join(ConfigExpW.MODEL_SAVE_PATH, 'best_model_cross_expw.pth')
    start     = time.time()
    history   = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, FINETUNE_EPOCHS + 1):

        if epoch == FREEZE_EPOCHS + 1:
            unfreeze_all(model)
            tt_params = list(model.fc.parameters()) + \
                        list(model.bn_fc.parameters()) + \
                        list(model.classifier.parameters())
            backbone_params = [
                p for n, p in model.named_parameters()
                if 'fc' not in n and 'bn_fc' not in n and 'classifier' not in n
            ]
            optimizer = optim.AdamW([
                {'params': tt_params,       'lr': LR_TT},
                {'params': backbone_params, 'lr': LR_BACKBONE},
            ], weight_decay=ConfigExpW.WEIGHT_DECAY)
            scheduler = get_scheduler(optimizer, warmup_epochs=1,
                                      total_epochs=FINETUNE_EPOCHS - FREEZE_EPOCHS)
            scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
            print(f"Эпоха {epoch}: lr_tt={LR_TT}, lr_backbone={LR_BACKBONE}")

        if epoch == AUGMENT_EPOCH:
            train_loader.dataset.dataset.transform = train_transform_full
            print(f"  → RandomErasing включён с эпохи {epoch}")

        print(f'\nЭпоха {epoch}/{FINETUNE_EPOCHS}  lr={optimizer.param_groups[0]["lr"]:.2e}')

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
                    'learning_rate': LR_TT,
                    'weight_decay':  ConfigExpW.WEIGHT_DECAY,
                    'rank':          TT_RANK,
                    'image_size':    ConfigExpW.IMAGE_SIZE,
                    'dataset':       'expw',
                    'method':        'tt_cross',
                }
            }, best_path)
            print(f'  ✓ Сохранена лучшая модель (acc={best_acc:.2f}%)')

    elapsed = time.time() - start
    print(f'\nОбучение завершено за {elapsed/60:.1f} мин')
    print(f'Лучшая val accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()