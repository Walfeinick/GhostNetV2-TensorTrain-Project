import os
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image

from config import train_transform_base, val_transform
from utils import _split, _split_sizes, _make_loaders


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

class ExpWDataset(Dataset):
    CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def __init__(self, root, transform=None):
        self.transform = transform
        self.img_dir   = os.path.join(root, 'image', 'origin')
        label_path     = os.path.join(root, 'label', 'label.lst')

        assert os.path.exists(self.img_dir), f"Нет папки с изображениями: {self.img_dir}"
        assert os.path.exists(label_path),   f"Нет файла меток: {label_path}"

        self.samples = self._parse_labels(label_path)

    def _parse_labels(self, label_path):
        samples = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                img_name = parts[0]
                label    = int(parts[7])
                img_path = os.path.join(self.img_dir, img_name)
                if os.path.exists(img_path):
                    samples.append((img_path, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def classes(self):
        return self.CLASSES


def build_dataloaders_expw(cfg):
    full_ds = ExpWDataset(cfg.DATA_PATH, transform=train_transform_base)

    n_test  = int(len(full_ds) * cfg.TEST_SPLIT)
    n_rest  = len(full_ds) - n_test
    rest_ds, test_ds = _split_sizes(full_ds, [n_rest, n_test], cfg.SEED)

    train_ds, val_ds = _split(rest_ds, cfg.VAL_SPLIT, cfg.SEED)
    val_ds.dataset.transform = val_transform

    return _make_loaders(train_ds, val_ds, test_ds, cfg, full_ds.classes)