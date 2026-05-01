import math
import random
import time
import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
from config import Config, ConfigExpW, PostTrainConfig
from data import build_dataloaders
from models.tt_cross import TTCrossLinear, convert_linear_to_tt_cross
from models.basemodel import GhostNetV2_Base

#Seed
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


#Scheduler
def get_scheduler(optimizer, warmup_epochs=5, total_epochs=60):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


#Запуск эпохи обучения
def run_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, training: bool):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc='Train' if training else 'Val  ', leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss    = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += labels.size(0)
            pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{100*correct/total:.1f}%')

    return total_loss / len(loader), 100 * correct / total

#Запуск цикла обучения
def start_train_FER2013(model, best_path:str, conf:Config|ConfigExpW, criterion, optimizer, scheduler, use_amp, scaler, n_epoch = Config.NUM_EPOCHS, train_type=0, AUGMENT_EPOCH:int = 7):
    

    set_seed(conf.SEED)

    #Данные
    train_loader, val_loader, _ = build_dataloaders(conf)

    #Модель
    print(f"Модель OK | Параметры: {sum(p.numel() for p in model.parameters()):,}")

    #Цикл обучения
    best_acc  = 0.0 
    start     = time.time()
    history   = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    if train_type == 1:
            freeze_backbone(model)
            
    for epoch in range(1, n_epoch + 1):
        print(f'\nЭпоха {epoch}/{n_epoch}  lr={optimizer.param_groups[0]["lr"]:.2e}')
        # Размораживаем backbone после FREEZE_EPOCHS эпох
        if train_type==1:
            if epoch == PostTrainConfig.FREEZE_EPOCHS + 1:
                unfreeze_all(model)
                # Пересоздаём optimizer с разными lr для backbone и TT-слоя
                optimizer = optim.AdamW([
                    {'params': model.fc.parameters(),         'lr': PostTrainConfig.LR_TT},
                    {'params': model.bn_fc.parameters(),      'lr': PostTrainConfig.LR_TT},
                    {'params': model.classifier.parameters(), 'lr': PostTrainConfig.LR_TT},
                    {'params': [p for n, p in model.named_parameters()
                                if 'fc' not in n and 'bn_fc' not in n
                                and 'classifier' not in n],   'lr': PostTrainConfig.LR_BACKBONE},
                ], weight_decay=Config.WEIGHT_DECAY)
                scheduler = get_scheduler(optimizer,
                                        warmup_epochs=1,
                                        total_epochs=PostTrainConfig.FINETUNE_EPOCHS - PostTrainConfig.FREEZE_EPOCHS)
                scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
                print(f"Эпоха {epoch}: backbone разморожен, lr_tt={PostTrainConfig.LR_TT}, lr_backbone={PostTrainConfig.LR_BACKBONE}")


        #Прогрессивная аугментация 
        if epoch == AUGMENT_EPOCH:
            train_loader.dataset.dataset.transform = conf.train_transform_full
            print(f"RandomErasing включён с эпохи {epoch}")

        train_loss, train_acc = run_epoch(model, train_loader, criterion,
                                          optimizer, scaler, conf.DEVICE,
                                          use_amp, training=True)
        val_loss, val_acc     = run_epoch(model, val_loader, criterion,
                                          optimizer, scaler, conf.DEVICE,
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
                    'num_classes':   conf.NUM_CLASSES,
                    'batch_size':    conf.BATCH_SIZE,
                    'learning_rate': conf.LEARNING_RATE,
                    'weight_decay':  conf.WEIGHT_DECAY,
                    'rank':          conf.RANK,
                    'image_size':    conf.IMAGE_SIZE,
                }
            }, os.path.join(Config.MODEL_SAVE_PATH, best_path))
            print(f'Сохранена лучшая модель (acc={best_acc:.2f}%)')

    elapsed = time.time() - start
    print(f'\nОбучение завершено за {elapsed/60:.1f} мин')
    print(f'Лучшая val accuracy: {best_acc:.2f}%')

def start_train_ExpW(): #TODO: Сделать функцию для ExpW
    return


#Утилки для Cross

def build_tt_cross_model(base_checkpoint_path: str, TT_RANK:int, cfg:Config|ConfigExpW = Config) -> GhostNetV2_Base:
    """
    Загружает обученную базовую модель и конвертирует FC-слой в TTCrossLinear.
    """
    # Загружаем базовую модель
    model = GhostNetV2_Base(num_classes=cfg.NUM_CLASSES, dropout=0.3, in_channels=cfg.IN_CHANNELS)
    checkpoint = torch.load(base_checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    print(f"Базовая модель загружена | val_acc={checkpoint['val_acc']:.2f}%")

    # Конвертируем FC → TTCrossLinear
    model.fc = convert_linear_to_tt_cross(model.fc, rank=TT_RANK)
    model.to(cfg.DEVICE)

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
    print("Backbone разморожен - joint fine-tuning")


