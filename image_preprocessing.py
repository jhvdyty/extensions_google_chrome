"""
Скрипт для предобработки изображений перед обучением
Решает проблемы:
- Слишком большие изображения
- Различные форматы
- Поврежденные файлы
"""

from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
import shutil

# увеличиваем лимит для PIL
Image.MAX_IMAGE_PIXELS = 200000000

def preprocess_dataset(input_dir, output_dir, max_size=1500, quality=95):
    """Предобрабатывает все изображения в датасете"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # создаем выходную структуру
    output_path.mkdir(parents=True, exist_ok=True)
    
    stats = {
        'processed': 0,
        'skipped': 0,
        'errors': 0,
        'too_large': 0
    }
    
    # обрабатываем все подпапки (high_quality, low_quality и т.д.)
    for subfolder in input_path.iterdir():
        if not subfolder.is_dir():
            continue
            
        print(f"\n Folder processing: {subfolder.name}")
        
        # создаем выходную папку
        output_subfolder = output_path / subfolder.name
        output_subfolder.mkdir(exist_ok=True)
        
        # находим все изображения
        image_files = list(subfolder.glob('*.jpg')) + \
                     list(subfolder.glob('*.png')) + \
                     list(subfolder.glob('*.jpeg')) + \
                     list(subfolder.glob('*.JPEG')) + \
                     list(subfolder.glob('*.JPG')) + \
                     list(subfolder.glob('*.PNG'))
        
        print(f"Images found: {len(image_files)}")
        
        # обрабатываем каждое изображение
        for img_path in tqdm(image_files, desc="Обработка"):
            try:
                # открываем изображение
                img = Image.open(img_path)
                
                # конвертируем в RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # проверяем размер
                width, height = img.size
                
                # если изображение слишком большое
                if width > max_size or height > max_size:
                    # уменьшаем с сохранением пропорций
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    stats['too_large'] += 1
                
                # сохраняем как JPEG с указанным качеством
                output_file = output_subfolder / f"{img_path.stem}.jpg"
                img.save(output_file, 'JPEG', quality=quality, optimize=True)
                
                stats['processed'] += 1
                
            except Exception as e:
                print(f"\nX processing error {img_path.name}: {e}")
                stats['errors'] += 1
                continue
    
    # выводим статистику
    print("\n" + "="*60)
    print("PROCESSING STATISTICS")
    print("="*60)
    print(f" Processed successfully: {stats['processed']}")
    print(f" Reduced (too large): {stats['too_large']}")
    print(f"X Errors: {stats['errors']}")
    print(f">>  Missed: {stats['skipped']}")
    print("="*60)
    
    return stats


def check_dataset_structure(data_dir):
    """Проверяет структуру датасета и выводит статистику"""
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"X Folder {data_dir} does not exist!")
        return
    
    print("\n DATA SET STRUCTURE")
    print("="*60)
    
    total_images = 0
    
    for subfolder in sorted(data_path.iterdir()):
        if not subfolder.is_dir():
            continue
        
        # считаем изображения
        images = list(subfolder.glob('*.jpg')) + list(subfolder.glob('*.png'))
        count = len(images)
        total_images += count
        
        print(f" {subfolder.name}: {count} images")
        
        # проверяем несколько изображений на размер
        if count > 0:
            sample_sizes = []
            for img_path in images[:5]:  # берем первые 5
                try:
                    with Image.open(img_path) as img:
                        sample_sizes.append(img.size)
                except:
                    pass
            
            if sample_sizes:
                avg_width = sum(s[0] for s in sample_sizes) / len(sample_sizes)
                avg_height = sum(s[1] for s in sample_sizes) / len(sample_sizes)
                print(f" Average size (sample): {int(avg_width)}x{int(avg_height)}")
    
    print("="*60)
    print(f" TOTAL images: {total_images}")
    print("="*60)


def split_dataset(input_dir, train_ratio=0.8):
    """Разделяет датасет на train и val"""
    import random
    
    input_path = Path(input_dir)
    train_path = input_path.parent / 'train'
    val_path = input_path.parent / 'val'
    
    # создаем структуру
    train_path.mkdir(exist_ok=True)
    val_path.mkdir(exist_ok=True)
    
    print("\n DATA SET DIVISION")
    print("="*60)
    
    for subfolder in input_path.iterdir():
        if not subfolder.is_dir():
            continue
        
        print(f"Class processing: {subfolder.name}")
        
        # создаем подпапки
        (train_path / subfolder.name).mkdir(exist_ok=True)
        (val_path / subfolder.name).mkdir(exist_ok=True)
        
        # получаем все изображения
        images = list(subfolder.glob('*.jpg')) + list(subfolder.glob('*.png'))
        random.shuffle(images)
        
        # разделяем
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # копируем в train
        for img in train_images:
            shutil.copy2(img, train_path / subfolder.name / img.name)
        
        # копируем в val
        for img in val_images:
            shutil.copy2(img, val_path / subfolder.name / img.name)
        
        print(f"  Train: {len(train_images)}, Val: {len(val_images)}")
    
    print("="*60)
    print(" The separation is complete!")
    print(f" Train: {train_path}")
    print(f" Val: {val_path}")


# ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ

if __name__ == '__main__':
    
    # проверяем структуру исходных данных
    print(" Step 1: Verify the source data")
    check_dataset_structure('data/raw')
    
    # предобрабатываем изображения
    print("\n-O-  Step 2: Image preprocessing")
    preprocess_dataset(
        input_dir='data/raw',
        output_dir='data/processed',
        max_size=1500,  # уменьшаем слишком большие изображения
        quality=95      # качество JPEG
    )
    
    # разделяем на train/val
    print("\n Step 3: Split into train/val")
    split_dataset('data/processed', train_ratio=0.8)
    
    # проверяем финальную структуру
    print("\n Final structure:")
    check_dataset_structure('data/train')
    check_dataset_structure('data/val')