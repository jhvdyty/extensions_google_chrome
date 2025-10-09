import torch
from torchvision import transforms
from PIL import Image

# Импортируй класс модели из своего файла
from train_ai import LightweightClassifier, get_simple_transforms

# === Настройки ===
model_path = 'model_for_detect_main/best_model_lightweight.pth'  # путь к сохранённой модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Создание модели ===
model = LightweightClassifier(num_classes=2)  # должна совпадать с исходной архитектурой
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

print(" Model successfully loaded!")

# Пути
#image_path = 'data/val_for_stuff/low_quality/agGIKYs4mYs.jpg'  # любое изображение
image_path = 'input/ugliest-people-in-the-world-8-62628152.jpg'
class_names = ['Low Quality', 'High Quality']

# Преобразования (должны совпадать с обучением)
transform = get_simple_transforms(train=False)

# Загрузка и преобразование изображения
img = Image.open(image_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)

# Предсказание
with torch.no_grad():
    output = model(img_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()

print(f"\n Image: {image_path}")
print(f" Predicted: {class_names[predicted_class]}")
print(f" Confidence: {confidence*100:.2f}%")

