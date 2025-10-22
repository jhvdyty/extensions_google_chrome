import torch
import torch.onnx
import onnxruntime
import numpy as np
from train_ai import LightweightClassifier

# Загружаем PyTorch модель
model = LightweightClassifier(num_classes=2)
model.load_state_dict(torch.load('model_for_detect_girl/best_model_lightweight.pth'))
model.eval()

# Создаём dummy input для трассировки
dummy_input = torch.randn(1, 3, 224, 224)

# Экспортируем в ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("✓ Model converted to ONNX")

# Затем конвертируем ONNX в TensorFlow.js:
# pip install onnx-tf tensorflowjs
# onnx-tf convert -i model.onnx -o model_tf
# tensorflowjs_converter --input_format=tf_saved_model model_tf model_tfjs