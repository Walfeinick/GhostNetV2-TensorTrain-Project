# GhostNetV2-TensorTrain-Project: TT-GhostNetV2 — Tensor Train Compression for FER

Интеграция Tensor Train разложения в архитектуру GhostNetV2 для задачи
распознавания эмоций на лице (FER2013). Цель - уменьшить число параметров
FC-слоя без потери точности, сохранив пригодность для мобильного деплоя.

> Работа оформляется в виде научной статьи (публикация ожидается в 2026 г. при публикации здесь будет ссылка)

## Датасет

**FER2013** 7 классов эмоций: angry, disgust, fear, happy, neutral, sad, surprise.  
Черно-белые изображения, одно лицо на фото.
[Скачать датасет FER2013](https://www.kaggle.com/datasets/msambare/fer2013)

**ExpW** 7 классов эмоций: angry, disgust, fear, happy, neutral, sad, surprise.
Цветные изображения, несколько лиц на одном.
[Скачать датасет ExpW](https://www.kaggle.com/datasets/nguhaduong/expression-in-the-wild-expw-dataset)


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
├── utils.py                # вспомогательные функции     
├── train_base.py           # обучение, просмотр результатов
├── evaluate.py             # точность на тесте + confusion matrix для моделей
├── benchmark.py            # сравнение латентности и параметров
├── convert_dataset.py      # конвертация ExpW в формат папок как у FER2013
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

Скачать FER2013 с Kaggle, проверить, что структура соответсвует примеру нижен (иначе обучение упадет)
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

Скачать ExpW с Kaggle, запустить скрипт на трансформацию структуры датасета 
```bash
python convert_dataset.py
```

После этого обновить `DATA_PATH` и `MODEL_SAVE_PATH` в `config.py`, аргументы функции `convert_dataset`: `src_images_dir`, `label_lst_path`, `dst_root` также под свои локальные пути. Без конвертации обучение упадет. 

**4. Обучение и Оценка**
Для запуска CLi интерфейса
```bash
python train_base
```
После выбираете нужный вариант.
Обучение моделей на FER2013 занимало около 20-25 минут, на ExpW 2 часа, на GPU NVIDIA GeForce RTX 4060 laptop

## Требования

- Python 3.10+
- PyTorch 2.6+ с поддержкой CUDA (CPU поддерживается, но обучение будет медленным)
- Полный список зависимостей в `requirements.txt`
