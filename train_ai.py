import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path

# датасет для загрузки изображений
class ImageQualityDataset(Dataset):
    """датасет для обучения модели оценки качества изображений"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # загружаем изображения высокого качества
        high_q_dir = self.root_dir / 'high_quality'
        if high_q_dir.exists():
            for img_path in high_q_dir.glob('*.jpg'):
                self.images.append(str(img_path))
                self.labels.append(1)
        
        # загружаем изображения низкого качества
        low_q_dir = self.root_dir / 'low_quality'
        if low_q_dir.exists():
            for img_path in low_q_dir.glob('*.jpg'):
                self.images.append(str(img_path))
                self.labels.append(0)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# модель с Transfer Learning (ResNet50)
class ImageClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(ImageClassifier, self).__init__()
        
        # загружаем предобученную ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # заменяем последний слой для нашей задачи
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


# трансформации для аугментации данных
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


# функция обучения
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # тренировка
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # валидация
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # метрики
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 60)
        
        # сохраняем лучшую модель
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('✓ Saved best model')
        
        scheduler.step(avg_val_loss)
    
    return model


#функция предсказания
def predict_image(model, image_path, transform):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence


#пример использования
if __name__ == '__main__':
    # параметры
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 0.0001
    
    # создаем датасеты
    train_dataset = ImageQualityDataset(
        root_dir='data/train',
        transform=get_transforms(train=True)
    )
    
    val_dataset = ImageQualityDataset(
        root_dir='data/val',
        transform=get_transforms(train=False)
    )
    
    # dataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=4)
    
    # создаем модель
    model = ImageClassifier(num_classes=2, pretrained=True)
    
    # обучаем
    print("Начинаем обучение...")
    print(f"Тренировочных изображений: {len(train_dataset)}")
    print(f"Валидационных изображений: {len(val_dataset)}")
    print("=" * 60)
    
    trained_model = train_model(model, train_loader, val_loader, 
                               epochs=EPOCHS, lr=LEARNING_RATE)
    
    # пример предсказания
    print("\nПример предсказания:")
    test_image = 'test_image.jpg'
    predicted_class, confidence = predict_image(
        trained_model, 
        test_image, 
        get_transforms(train=False)
    )
    
    class_names = ['Низкое качество', 'Высокое качество']
    print(f"Класс: {class_names[predicted_class]}")
    print(f"Уверенность: {confidence*100:.2f}%")