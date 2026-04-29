"""
Конвертер датасета с разметкой label.lst в структуру формата FER2013.

Структура на выходе:
    dst_root/
        train/
            angry/   disgust/   fear/   happy/   sad/   surprise/   neutral/
        test/
            angry/   disgust/   fear/   happy/   sad/   surprise/   neutral/

Каждое лицо вырезается по bbox-координатам из label.lst и сохраняется
как отдельный файл. Из одного фото может получиться несколько кропов.
"""

import os
import random
from pathlib import Path
from PIL import Image
from config import Config

# ─── Константы ────────────────────────────────────────────────────────────────

LABEL_TO_CLASS = {
    "0": "angry",
    "1": "disgust",
    "2": "fear",
    "3": "happy",
    "4": "sad",
    "5": "surprise",
    "6": "neutral",
}

# Минимальный размер кропа в пикселях. Слишком маленькие лица — шум.
MIN_CROP_SIZE = 16


# ─── Парсинг label.lst ─────────────────────────────────────────────────────────

def parse_label_lst(label_lst_path: str) -> list[dict]:
    """
    Читает label.lst и возвращает список записей.

    Каждая запись — словарь:
        {
            "image_name": str,
            "face_id":    int,
            "top":        int,   # bbox: верхний край
            "left":       int,   # bbox: левый край
            "right":      int,   # bbox: правый край
            "bottom":     int,   # bbox: нижний край
            "label":      str,   # строка "0"–"6"
            "class_name": str,   # "angry", "happy", ...
        }

    Формат строки в файле:
        image_name  face_id  top  left  right  bottom  confidence  label
    """
    records = []

    with open(label_lst_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 8:
                print(f"[!] Строка {line_num}: мало полей ({len(parts)}), пропускаем")
                continue

            image_name = parts[0]
            face_id    = int(parts[1])
            top        = int(float(parts[2]))   # иногда бывают float в файлах
            left       = int(float(parts[3]))
            right      = int(float(parts[4]))
            bottom     = int(float(parts[5]))
            # parts[6] — confidence, нам не нужна
            label      = parts[7]

            if label not in LABEL_TO_CLASS:
                print(f"[!] Строка {line_num}: неизвестная метка '{label}', пропускаем")
                continue

            records.append({
                "image_name": image_name,
                "face_id":    face_id,
                "top":        top,
                "left":       left,
                "right":      right,
                "bottom":     bottom,
                "label":      label,
                "class_name": LABEL_TO_CLASS[label],
            })

    return records


# ─── Кропание лица ─────────────────────────────────────────────────────────────

def crop_face(image: Image.Image, rec: dict, padding: float = 0.0) -> Image.Image | None:
    """
    Вырезает лицо из PIL-изображения по bbox-координатам из записи rec.

    padding — дополнительный отступ вокруг bbox в долях от размера bbox.
               0.1 = добавить 10% ширины/высоты с каждой стороны.
               По умолчанию 0.0 — вырезаем ровно по разметке.

    Возвращает PIL.Image или None, если кроп слишком маленький.

    Как работает:
        1. Берём координаты (top, left, right, bottom) из label.lst.
        2. Опционально расширяем bbox на padding — это помогает захватить
           контекст лица (уши, лоб), который может улучшить качество модели.
        3. Зажимаем координаты в границы изображения (clip), чтобы не выйти
           за пределы при вызове crop().
        4. Проверяем минимальный размер — маленькие кропы (< MIN_CROP_SIZE)
           скорее всего являются ошибками разметки.
        5. Возвращаем вырезанный фрагмент.
    """
    img_w, img_h = image.size

    top    = rec["top"]
    left   = rec["left"]
    right  = rec["right"]
    bottom = rec["bottom"]

    # Расширяем bbox на padding
    if padding > 0.0:
        bw = right - left   # ширина bbox
        bh = bottom - top   # высота bbox
        pad_x = int(bw * padding)
        pad_y = int(bh * padding)
        left   -= pad_x
        right  += pad_x
        top    -= pad_y
        bottom += pad_y

    # Зажимаем в границы изображения
    left   = max(0, left)
    top    = max(0, top)
    right  = min(img_w, right)
    bottom = min(img_h, bottom)

    # Проверка минимального размера
    if (right - left) < MIN_CROP_SIZE or (bottom - top) < MIN_CROP_SIZE:
        return None

    # PIL.Image.crop принимает (left, upper, right, lower)
    return image.crop((left, top, right, bottom))


# ─── Основная функция конвертации ─────────────────────────────────────────────

def convert_dataset(
    src_images_dir: str,
    label_lst_path: str,
    dst_root: str,
    test_split: float = 0.2,
    padding: float = 0.0,
    seed: int = 42,
):
    """
    Конвертирует датасет с label.lst в структуру FER2013.

    Аргументы:
        src_images_dir  — папка с исходными изображениями
        label_lst_path  — путь к файлу label.lst
        dst_root        — папка, куда сохранить результат
        test_split      — доля данных для test (по умолчанию 0.2 = 20%)
        padding         — отступ вокруг bbox лица (0.0 — без отступа)
        seed            — seed для воспроизводимости разбивки

    Логика работы:
        1. Парсим label.lst → получаем список записей (одна запись = одно лицо).
        2. Группируем записи по классу для стратифицированного сплита:
           соотношение классов в train и test будет одинаковым.
        3. Для каждой записи открываем исходное изображение и вырезаем кроп.
        4. Сохраняем кроп в нужную папку: dst_root/split/class_name/filename.
           Имя файла: <image_name_без_расширения>_face<face_id>.jpg
    """
    random.seed(seed)

    # 1. Парсинг
    print("Читаем label.lst...")
    records = parse_label_lst(label_lst_path)
    print(f"  Всего записей (лиц): {len(records)}")

    # 2. Группировка по классу
    class_to_records: dict[str, list[dict]] = {cls: [] for cls in LABEL_TO_CLASS.values()}
    for rec in records:
        class_to_records[rec["class_name"]].append(rec)

    # 3. Стратифицированный сплит и копирование
    stats = {"copied": 0, "skipped_no_file": 0, "skipped_small": 0}

    for class_name, class_records in class_to_records.items():
        random.shuffle(class_records)

        n_test  = int(len(class_records) * test_split)
        n_train = len(class_records) - n_test

        split_map = {
            "train": class_records[:n_train],
            "test":  class_records[n_train:],
        }

        print(f"  {class_name}: train={n_train}, test={n_test}")

        for split_name, split_records in split_map.items():
            dst_dir = Path(dst_root) / split_name / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)

            for rec in split_records:
                src_path = Path(src_images_dir) / rec["image_name"]

                if not src_path.exists():
                    stats["skipped_no_file"] += 1
                    continue

                # Открываем изображение
                try:
                    image = Image.open(src_path).convert("RGB")
                except Exception as e:
                    print(f"  [!] Не удалось открыть {src_path}: {e}")
                    stats["skipped_no_file"] += 1
                    continue

                # Вырезаем лицо
                crop = crop_face(image, rec, padding=padding)
                if crop is None:
                    stats["skipped_small"] += 1
                    continue

                # Формируем имя файла: photo_001_face0.jpg
                stem = Path(rec["image_name"]).stem          # "photo_001"
                dst_filename = f"{stem}_face{rec['face_id']}.jpg"

                # Если в image_name есть подпапки — сохраняем в одну папку (stem)
                dst_path = dst_dir / dst_filename
                crop.save(dst_path, "JPEG", quality=95)
                stats["copied"] += 1

    # 4. Итог
    print("\n─── Готово ───")
    print(f"  Сохранено кропов : {stats['copied']}")
    print(f"  Файл не найден   : {stats['skipped_no_file']}")
    print(f"  Кроп слишком мал : {stats['skipped_small']}")
    print(f"  Датасет в папке  : {dst_root}")


# ─── Точка входа ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    convert_dataset(
        src_images_dir=r'C:\Users\levsh\Documents\TT-GhostNetV2\data\ExpW\expw_dataset\data\image\origin',  # папка с исходными фото
        label_lst_path=r'C:\Users\levsh\Documents\TT-GhostNetV2\data\ExpW\expw_dataset\data\label\label.lst',
        dst_root=r'C:\Users\levsh\Documents\TT-GhostNetV2\data\ExpWConv',          # итоговая папка
        test_split=0.2,                             # 20% на тест
        padding=0.05,                               # +5% отступ вокруг лица
        seed=Config.SEED,
    )