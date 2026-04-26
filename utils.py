import math
import random
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image

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


#run_epoch
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


def _split(dataset, val_fraction, seed):
    n_val   = int(len(dataset) * val_fraction)
    n_train = len(dataset) - n_val
    return _split_sizes(dataset, [n_train, n_val], seed)


def _split_sizes(dataset, sizes, seed):
    return random_split(dataset, sizes,
                        generator=torch.Generator().manual_seed(seed))


def _make_loaders(train_ds, val_ds, test_ds, cfg, classes):
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  num_workers=cfg.NUM_WORKERS,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=cfg.NUM_WORKERS,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=cfg.NUM_WORKERS,
                              pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"Классы: {classes}")
    return train_loader, val_loader, test_loader

