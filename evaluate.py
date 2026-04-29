import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from config import Config, ConfigExpW
from data import build_dataloaders
from models import TT_GhostNetV2_FER, GhostNetV2_Base, TTCrossLinear
from models.tt_cross import convert_linear_to_tt_cross

CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def evaluate(model, test_loader, device):
    """Прогоняем тест, возвращаем предсказания и лейблы."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Test'):
            images  = images.to(device)
            outputs = model(images)
            preds   = outputs.argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    return all_preds, all_labels


def print_metrics(all_preds, all_labels, model_name):
    """Выводим accuracy и classification report."""
    test_acc = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"\n{model_name} | Test accuracy: {test_acc:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0))
    return test_acc


def save_confusion_matrix(all_preds, all_labels, model_name, save_path):
    """Строим и сохраняем confusion matrix."""
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                cmap='Blues')
    plt.title(f'{model_name} — Confusion Matrix')
    plt.ylabel('Настоящий класс')
    plt.xlabel('Предсказанный класс')
    plt.tight_layout()
    plt.savefig(f'artifacts/{save_path}', dpi=150)
    plt.close()
    print(f"Confusion matrix сохранена: {save_path}")


def main():

    #FER2013
    _, _, test_loader = build_dataloaders(Config)

    #TT-модель
    tt_path    = os.path.join(Config.MODEL_SAVE_PATH, 'best_model.pth')
    tt_model   = TT_GhostNetV2_FER(num_classes=Config.NUM_CLASSES, dropout=0.3).to(Config.DEVICE)
    checkpoint = torch.load(tt_path, map_location=Config.DEVICE)
    tt_model.load_state_dict(checkpoint['model_state'])
    print(f"TT-модель загружена | val_acc={checkpoint['val_acc']:.2f}%")

    tt_preds, tt_labels = evaluate(tt_model, test_loader, Config.DEVICE)
    print_metrics(tt_preds, tt_labels, 'TT-GhostNetV2')
    save_confusion_matrix(tt_preds, tt_labels,
                          'TT-GhostNetV2', 'confusion_matrix_tt.png')

    #Базовая модель
    base_path    = os.path.join(Config.MODEL_SAVE_PATH, 'best_model_base.pth')
    base_model   = GhostNetV2_Base(num_classes=Config.NUM_CLASSES, dropout=0.3).to(Config.DEVICE)
    checkpoint   = torch.load(base_path, map_location=Config.DEVICE)
    base_model.load_state_dict(checkpoint['model_state'])
    print(f"Base-модель загружена | val_acc={checkpoint['val_acc']:.2f}%")

    base_preds, base_labels = evaluate(base_model, test_loader, Config.DEVICE)
    print_metrics(base_preds, base_labels, 'GhostNetV2-Base')
    save_confusion_matrix(base_preds, base_labels,
                          'GhostNetV2-Base', 'confusion_matrix_base.png')

    #TT-Cross-модель
    tt_cross_path  = os.path.join(Config.MODEL_SAVE_PATH, 'best_model_tt_cross.pth')
    tt_cross_model = GhostNetV2_Base(num_classes=Config.NUM_CLASSES, dropout=0.3)
    tt_cross_model.fc = convert_linear_to_tt_cross(tt_cross_model.fc, rank=16)
    tt_cross_model.to(Config.DEVICE)
    checkpoint  = torch.load(tt_cross_path, map_location=Config.DEVICE)
    tt_cross_model.load_state_dict(checkpoint['model_state'])
    print(f"TT-Cross-модель загружена | val_acc={checkpoint['val_acc']:.2f}%")

    tt_preds, tt_labels = evaluate(tt_cross_model, test_loader, Config.DEVICE)
    print_metrics(tt_preds, tt_labels, 'TT-Cross-GhostNetV2')
    save_confusion_matrix(tt_preds, tt_labels,
                          'TT-Cross-GhostNetV2', 'confusion_matrix_tt_cross.png')


    #ExpW
    _, _, test_loader = build_dataloaders(ConfigExpW)

    #TT-модель
    tt_path    = os.path.join(ConfigExpW.MODEL_SAVE_PATH, 'best_model_tt_expw.pth')
    tt_model   = TT_GhostNetV2_FER(num_classes=ConfigExpW.NUM_CLASSES, dropout=0.3).to(ConfigExpW.DEVICE)
    checkpoint = torch.load(tt_path, map_location=ConfigExpW.DEVICE)
    tt_model.load_state_dict(checkpoint['model_state'])
    print(f"TT-модель (ExpW) загружена | val_acc={checkpoint['val_acc']:.2f}%")

    tt_preds, tt_labels = evaluate(tt_model, test_loader, ConfigExpW.DEVICE)
    print_metrics(tt_preds, tt_labels, 'TT-GhostNetV2 (ExpW)')
    save_confusion_matrix(tt_preds, tt_labels,
                          'TT-GhostNetV2_expw', 'confusion_matrix_tt_expw.png')

    #Базовая модель
    base_path    = os.path.join(ConfigExpW.MODEL_SAVE_PATH, 'best_model_base_expw.pth')
    base_model   = GhostNetV2_Base(num_classes=ConfigExpW.NUM_CLASSES, dropout=0.3).to(ConfigExpW.DEVICE)
    checkpoint   = torch.load(base_path, map_location=ConfigExpW.DEVICE)
    base_model.load_state_dict(checkpoint['model_state'])
    print(f"Base-модель загружена | val_acc={checkpoint['val_acc']:.2f}%")

    base_preds, base_labels = evaluate(base_model, test_loader, ConfigExpW.DEVICE)
    print_metrics(base_preds, base_labels, 'GhostNetV2-Base')
    save_confusion_matrix(base_preds, base_labels,
                          'GhostNetV2-Base_expw', 'confusion_matrix_base_expw.png')

    #TT-Cross-модель
    tt_cross_path  = os.path.join(ConfigExpW.MODEL_SAVE_PATH, 'best_model_tt_cross_expw.pth') #TODO:
    tt_cross_model = GhostNetV2_Base(num_classes=ConfigExpW.NUM_CLASSES, dropout=0.3)
    tt_cross_model.fc = convert_linear_to_tt_cross(tt_cross_model.fc, rank=16)
    tt_cross_model.to(ConfigExpW.DEVICE)
    checkpoint  = torch.load(tt_cross_path, map_location=ConfigExpW.DEVICE)
    tt_cross_model.load_state_dict(checkpoint['model_state'])
    print(f"TT-Cross-модель загружена | val_acc={checkpoint['val_acc']:.2f}%")

    tt_preds, tt_labels = evaluate(tt_cross_model, test_loader, ConfigExpW.DEVICE)
    print_metrics(tt_preds, tt_labels, 'TT-Cross-GhostNetV2_expw')
    save_confusion_matrix(tt_preds, tt_labels,
                          'TT-Cross-GhostNetV2_expw', 'confusion_matrix_tt_cross_expw.png')

if __name__ == '__main__':
    main()