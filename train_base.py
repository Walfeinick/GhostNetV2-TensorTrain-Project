import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config, ConfigExpW, PostTrainConfig
from data import build_dataloaders
from models import GhostNetV2_Base, TT_GhostNetV2_FER, TTCrossLinear
from utils import set_seed, get_scheduler, run_epoch, start_train_FER2013, build_tt_cross_model, freeze_backbone, get_scheduler
from benchmark import main as run_benchmark, benchmark_model
from evaluate import main as run_evaluate, evaluate, print_metrics, save_confusion_matrix

#FER2013

#Base
def train_base_FER2013():
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
    start_train_FER2013(model, 'best_model_base.pth', Config, 
                        criterion, optimizer, scheduler, 
                        use_amp, scaler, )

#TT-model
def train_tt_FER2013():
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
    start_train_FER2013(model, 'best_model_tt.pth', Config, 
                        criterion, optimizer, scheduler, 
                        use_amp, scaler)
    

#TT-Cross-model
def train_tt_cross_FER2013():
    model = GhostNetV2_Base(num_classes=Config.NUM_CLASSES, dropout=0.3).to(Config.DEVICE)
    #загрузка + конвертация
    base_path = os.path.join(Config.MODEL_SAVE_PATH, 'best_model_base.pth')
    assert os.path.exists(base_path), (
        f"Не найден файл базовой модели: {base_path}\n"
    )
    model = build_tt_cross_model(base_path, PostTrainConfig.TT_RANK)
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

    start_train_FER2013(model, 'best_model_tt_cross.pth', Config, 
                        criterion, optimizer, scheduler, use_amp, scaler, 
                        PostTrainConfig.FINETUNE_EPOCHS, train_type=1, AUGMENT_EPOCH=3)

#===================================================================================================
#ExpW

#Base
def train_base_ExpW():
    model = GhostNetV2_Base(num_classes=ConfigExpW.NUM_CLASSES, dropout=0.3, in_channels=ConfigExpW.IN_CHANNELS).to(ConfigExpW.DEVICE)
    #Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(),
                            lr=ConfigExpW.LEARNING_RATE,
                            weight_decay=ConfigExpW.WEIGHT_DECAY)
    scheduler = get_scheduler(optimizer,
                               warmup_epochs=5,
                               total_epochs=ConfigExpW.NUM_EPOCHS)
    use_amp = ConfigExpW.DEVICE.type == 'cuda'
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)
    start_train_FER2013(model, 'best_model_base_expw.pth', 
                        ConfigExpW, criterion, optimizer, 
                        scheduler, use_amp, scaler, 
                        ConfigExpW.NUM_EPOCHS)

#TT-model
def train_tt_ExpW():
    model = TT_GhostNetV2_FER(num_classes=ConfigExpW.NUM_CLASSES, dropout=0.3, in_channels=ConfigExpW.IN_CHANNELS).to(ConfigExpW.DEVICE)
    #Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(),
                            lr=ConfigExpW.LEARNING_RATE,
                            weight_decay=ConfigExpW.WEIGHT_DECAY)
    scheduler = get_scheduler(optimizer,
                               warmup_epochs=5,
                               total_epochs=ConfigExpW.NUM_EPOCHS)
    use_amp = ConfigExpW.DEVICE.type == 'cuda'
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)
    start_train_FER2013(model, 'best_model_tt_expw.pth', 
                        ConfigExpW, criterion, optimizer, 
                        scheduler, use_amp, scaler,
                        ConfigExpW.NUM_EPOCHS)


#TT-Cross-model
def train_tt_cross_ExpW():
    model = GhostNetV2_Base(num_classes=ConfigExpW.NUM_CLASSES, dropout=0.3, in_channels=ConfigExpW.IN_CHANNELS).to(ConfigExpW.DEVICE)
    #загрузка + конвертация
    base_path = os.path.join(ConfigExpW.MODEL_SAVE_PATH, 'best_model_base_expw.pth')
    assert os.path.exists(base_path), (
        f"Не найден файл базовой модели: {base_path}\n"
    )
    model = build_tt_cross_model(base_path, PostTrainConfig.TT_RANK)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PostTrainConfig.LR_TT,
        weight_decay=ConfigExpW.WEIGHT_DECAY
    )
    scheduler = get_scheduler(optimizer,
                               warmup_epochs=2,
                               total_epochs=PostTrainConfig.FINETUNE_EPOCHS)
    use_amp = ConfigExpW.DEVICE.type == 'cuda'
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

    start_train_FER2013(model, 'best_model_tt_cross_expw.pth', 
                        ConfigExpW, criterion, optimizer, 
                        scheduler, use_amp, scaler, 
                        PostTrainConfig.FINETUNE_EPOCHS, 
                        train_type=1, AUGMENT_EPOCH=3)


def main():
    print(f"Select the option:\n1 - Train all models\n2 - Train all models uses FER2013\n3 - Train all models uses ExpW")
    print(f"4 - Train base model use FER2013\n5 - Train TT-model use FER2013\n6 - Train TT-Cross model use FER2013")
    print(f"7 - Train base model use ExpW\n8 - Train TT-model use ExpW\n9 - Train TT-Cross model use ExpW\n")
    print(f"10 - run comparison\n11 - run evaluate")
    print(f'0 to exit')

    choice = input()
    if choice == "1":
        train_base_FER2013(), train_tt_FER2013(), train_tt_cross_FER2013(), 
        train_base_ExpW(), train_tt_ExpW(), train_tt_cross_ExpW()
    elif choice == "2":
        train_base_FER2013(), train_tt_FER2013(), train_tt_cross_FER2013()
    elif choice == "3":
        train_base_ExpW(), train_tt_ExpW(), train_tt_cross_ExpW()
    elif choice == "4":
        train_base_FER2013()
    elif choice == "5":
        train_tt_FER2013()
    elif choice == "6":
        train_tt_cross_FER2013()
    elif choice == "7":
        train_base_ExpW()
    elif choice == "8":
        train_tt_ExpW()
    elif choice == "9":
        train_tt_cross_ExpW()
    elif choice == "10":
        run_benchmark()
    elif choice == "11":
        run_evaluate()
    elif choice == "0":
        sys.exit
    else:
        print("Invalid command, try 0-11, u badass")
    
if __name__ == '__main__':
    main()