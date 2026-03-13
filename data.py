import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

from config import train_transform_base, val_transform


#Данные
def build_dataloaders(cfg):
    train_path = os.path.join(cfg.DATA_PATH, 'train')
    test_path  = os.path.join(cfg.DATA_PATH, 'test')

    assert os.path.exists(train_path), f"Нет папки train: {train_path}"
    assert os.path.exists(test_path),  f"Нет папки test:  {test_path}"

    full_train = ImageFolder(train_path, transform=train_transform_base)
    test_ds    = ImageFolder(test_path,  transform=val_transform)

    # Разбивка train на train + val
    n_val   = int(len(full_train) * cfg.VAL_SPLIT)
    n_train = len(full_train) - n_val
    train_ds, val_ds = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.SEED)
    )
    # val не должен использовать аугментации
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  num_workers=cfg.NUM_WORKERS,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=cfg.NUM_WORKERS,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=cfg.NUM_WORKERS,
                              pin_memory=True)

    print(f"Train: {n_train} | Val: {n_val} | Test: {len(test_ds)}")
    print(f"Классы: {full_train.classes}")
    return train_loader, val_loader, test_loader