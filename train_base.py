import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config, PostTrainConfig, train_transform_full
from data import build_dataloaders
from models import GhostNetV2_Base, TT_GhostNetV2_FER, TTCrossLinear
from utils import set_seed, get_scheduler, run_epoch, start_train_FER2013, build_tt_cross_model, freeze_backbone, get_scheduler



def main():
    #Base

    model = GhostNetV2_Base(num_classes=Config.NUM_CLASSES, dropout=0.3).to(Config.DEVICE)
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
    start_train_FER2013(model, 'best_model_base.pth', Config, criterion, optimizer, scheduler, use_amp, scaler)


    #TT-model

    model = TT_GhostNetV2_FER(num_classes=Config.NUM_CLASSES, dropout=0.3).to(Config.DEVICE)
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
    start_train_FER2013(model, 'best_model_tt.pth', Config, criterion, optimizer, scheduler, use_amp, scaler)
    

    #TT-Cross-model

    model = GhostNetV2_Base(num_classes=Config.NUM_CLASSES, dropout=0.3).to(Config.DEVICE)
    #загрузка + конвертация
    base_path = os.path.join(Config.MODEL_SAVE_PATH, 'best_model_base.pth')
    assert os.path.exists(base_path), (
        f"Не найден файл базовой модели: {base_path}\n"
    )
    model = build_tt_cross_model(base_path)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PostTrainConfig.LR_TT,
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = get_scheduler(optimizer,
                               warmup_epochs=2,
                               total_epochs=PostTrainConfig.FINETUNE_EPOCHS)
    use_amp = Config.DEVICE.type == 'cuda'
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

    start_train_FER2013(model, 'best_model_tt_cross.pth', Config, criterion, optimizer, scheduler, use_amp, scaler, PostTrainConfig.FINETUNE_EPOCHS, train_type=1, AUGMENT_EPOCH=3)



if __name__ == '__main__':
    main()