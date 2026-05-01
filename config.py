import os
import torch
from torchvision import transforms
#Config

#Второй датасет ExpW
#https://www.kaggle.com/datasets/nguhaduong/expression-in-the-wild-expw-dataset/data


class Config:
    DATA_PATH       = r'C:\Users\levsh\Documents\TT-GhostNetV2\data\fer2013'
    MODEL_SAVE_PATH = r'C:\Users\levsh\Documents\TT-GhostNetV2\models'

    IN_CHANNELS   = 1
    IMAGE_SIZE    = 48
    NUM_CLASSES   = 7

    BATCH_SIZE    = 32
    NUM_EPOCHS    = 50
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY  = 3e-4
    RANK          = 16

    VAL_SPLIT     = 0.2
    NUM_WORKERS   = 0 if os.name == 'nt' else 4
    SEED          = 42
    DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Трансформы
    train_transform_base = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_transform_full = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)

class ConfigExpW:
    DATA_PATH       = r'C:\Users\levsh\Documents\TT-GhostNetV2\data\ExpWConv'
    MODEL_SAVE_PATH = r'C:\Users\levsh\Documents\TT-GhostNetV2\models'
    TEST_SPLIT = 0.1  
    IN_CHANNELS   = 3
    IMAGE_SIZE    = 224
    NUM_CLASSES   = 7

    BATCH_SIZE    =64
    NUM_EPOCHS    = 35
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY  = 3e-4
    RANK          = 16

    VAL_SPLIT     = 0.2
    NUM_WORKERS   = 0 if os.name == 'nt' else 4
    SEED          = 42
    DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Трансформы
    train_transform_base = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_transform_full = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

os.makedirs(ConfigExpW.MODEL_SAVE_PATH, exist_ok=True)

class PostTrainConfig:
    # ─── Параметры дообучения ──────────────────────────────────────────────────────
    FINETUNE_EPOCHS = 12    # эпох дообучения после конвертации
    FREEZE_EPOCHS   = 5    # первые N эпох обучаем только TT-слой
    TT_RANK         = 16    # ранг TT-Cross (совпадает с TT-from-scratch для честного сравнения)
    LR_TT           = 5e-4  # lr для TT-слоя
    LR_BACKBONE     = 1e-4  # lr для backbone при размораживании
    



