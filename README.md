# GhostNetV2-TensorTrain-Project: TT-GhostNetV2 — Tensor Train Compression for FER

Интеграция Tensor Train разложения в архитектуру GhostNetV2 для задачи
распознавания эмоций на лице (FER2013). Цель - уменьшить число параметров
FC-слоя без потери точности, сохранив пригодность для мобильного деплоя.

> Работа оформляется в виде научной статьи (публикация ожидается в 2026 г. при публикации здесь будет ссылка)

## Результаты

| Метрика | GhostNetV2-Base | TT-GhostNetV2 | Разница |
|---------|----------------|---------------|---------|
| Test Accuracy | 61.87% | **63.32%** | +1.45% |
| Параметры (всего) | 491,751 | **379,495** | −22.8% |
| Параметры FC-слоя | 123,008 | **10,752** | −11.4x |
| Размер модели | 6.14 MB | **4.79 MB** | −22% |
| Латентность GPU | 3.14 ms | 3.30 ms | ~equal |
| Латентность CPU | 10.08 ms | 10.42 ms | ~equal |

TT-разложение сжало FC-слой в **11.4 раза** при этом точность на тесте
выросла на 1.45% - за счёт регуляризирующего эффекта разложения.

## Датасет

**FER2013** 7 классов эмоций: angry, disgust, fear, happy, neutral, sad, surprise.  
Train: 22,968 · Val: 5,741 · Test: 7,178

## Архитектура
Для эксперимента оригинальный FC-слой `Linear(960 → 128)` заменён на `TTLinear`, это
три малых тензорных ядра с рангом `r=16`:

## Структура проекта

```
GhostNetV2-TensorTrain-Project/
├── models/
│   ├── __init__.py         # экспорт всех моделей
│   ├── tt_linear.py        # TTLinear — FC-слой с TT-разложением
│   ├── ghostnet.py         # GhostModule, GhostBottleneck, TT_GhostNetV2_FER
│   └── basemodel.py        # GhostNetV2_Base — базовая модель без TT
├── config.py               # гиперпараметры, пути, трансформы
├── data.py                 # загрузка датасета и аугментации
├── utils.py                # seed, scheduler, цикл обучения
├── train.py                # обучение TT-GhostNetV2
├── train_base.py           # обучение GhostNetV2-Base
├── evaluate.py             # точность на тесте + confusion matrix для обеих моделей
├── benchmark.py            # сравнение латентности и параметров
└── requirements.txt
```

## Воспроизведение результатов

**1. Клонировать репозиторий**
```bash
git clone https://github.com/Walfeinick/GhostNetV2-TensorTrain-Project.git
cd GhostNetV2-TensorTrain-Project
```

**2. Установить зависимости**

Сначала установить PyTorch: выбрать команду под свою версию CUDA на [pytorch.org](https://pytorch.org/get-started/locally/).
Пример для CUDA 12.4:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
Затем остальные зависимости:
```bash
pip install -r requirements.txt
```

**3. Подготовить датасет**

Скачать FER2013 с [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) и разложить в следующую структуру:
```
data/fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   └── ...
└── test/
    ├── angry/
    └── ...
```
После этого обновить `DATA_PATH` и `MODEL_SAVE_PATH` в `config.py` под свои локальные пути.

**4. Обучение**
```bash
python train.py        # TT-GhostNetV2
python train_base.py   # GhostNetV2-Base
```

**5. Оценка**
```bash
python evaluate.py     # точность на тесте + confusion matrix (обе модели)
python benchmark.py    # число параметров, размер модели, латентность GPU/CPU
```

## Требования

- Python 3.10+
- PyTorch 2.6+ с поддержкой CUDA (CPU поддерживается, но обучение будет медленным)
- Полный список зависимостей в `requirements.txt`
