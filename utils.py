import math
import random
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm


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
