import torch
import sys
import os
from train_ai import LightweightClassifier

print("="*60)
print("PyTorch -> ONNX Conversion (для использования в браузере)")
print("="*60)

# === 1. Загружаем модель ===
print("\n[1/2] Loading PyTorch model...")
model = LightweightClassifier(num_classes=2)
state_dict = torch.load("model_for_detect_main/best_model_lightweight.pth", map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()
print("Model loaded")

# === 2. Экспорт в ONNX (оптимизированный для веба) ===
print("\n[2/2] Exporting to ONNX (web-optimized)...")
dummy_input = torch.randn(1, 3, 128, 128)
onnx_path = "model_web.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=['input'],
    output_names=['output'],
    opset_version=13,
    do_constant_folding=True,
    export_params=True,
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

file_size = os.path.getsize(onnx_path) / (1024 * 1024)
print(f"Saved {onnx_path} ({file_size:.2f} MB)")

# Проверяем модель
import onnx
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("ONNX model is valid")

# === 3. Создаем HTML для тестирования с ONNX Runtime Web ===
print("\n[3/3] Creating test files...")

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ONNX Model Classifier</title>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px; 
            margin: 50px auto; 
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 { color: #333; margin-top: 0; }
        .upload-area {
            border: 2px dashed #ccc;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            margin: 20px 0;
            cursor: pointer;
            transition: all 0.3s;
        }
        .upload-area:hover { border-color: #4CAF50; background: #f9f9f9; }
        .upload-area.dragover { border-color: #4CAF50; background: #e8f5e9; }
        #imageInput { display: none; }
        #preview { 
            max-width: 256px; 
            max-height: 256px;
            border: 2px solid #ddd;
            border-radius: 8px;
            margin: 20px auto;
            display: none;
        }
        button {
            width: 100%;
            padding: 15px;
            font-size: 18px;
            cursor: pointer;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 8px;
            transition: background 0.3s;
            margin: 10px 0;
        }
        button:hover { background: #45a049; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        #result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 8px;
            display: none;
        }
        .success { background: #d4edda; border: 2px solid #c3e6cb; color: #155724; }
        .error { background: #f8d7da; border: 2px solid #f5c6cb; color: #721c24; }
        .info { background: #d1ecf1; border: 2px solid #bee5eb; color: #0c5460; }
        .loading { background: #fff3cd; border: 2px solid #ffeaa7; color: #856404; }
        .result-score {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            margin: 10px 0;
            background: #f8f9fa;
            border-radius: 5px;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            transition: width 0.5s ease;
        }
        code { 
            background: #f4f4f4; 
            padding: 2px 6px; 
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>AI Image Classifier</h1>
        <p>Upload an image to classify it using your trained model</p>
        
        <div class="upload-area" id="uploadArea">
            <p>Click to select or drag & drop an image</p>
            <input type="file" id="imageInput" accept="image/*">
        </div>
        
        <img id="preview" alt="Preview">
        
        <button id="inferBtn" onclick="runInference()" disabled>Classify Image</button>
        
        <div id="result"></div>
    </div>

    <script>
        let session = null;
        let imageLoaded = false;
        
        // Настройка ONNX Runtime
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
        
        async function loadModel() {
            try {
                showResult('Loading AI model...', 'loading');
                session = await ort.InferenceSession.create('model_web.onnx');
                console.log('✓ Model loaded successfully');
                console.log('Input:', session.inputNames, session.inputs);
                console.log('Output:', session.outputNames, session.outputs);
                showResult('✓ Model ready! Upload an image to classify.', 'success');
            } catch (e) {
                console.error('Failed to load model:', e);
                showResult(
                    '✗ Failed to load model: ' + e.message + 
                    '<br><br><strong>Make sure to:</strong><br>' +
                    '1. Run a local server: <code>python -m http.server 8000</code><br>' +
                    '2. Open: <code>http://localhost:8000/test_onnx.html</code>',
                    'error'
                );
            }
        }
        
        function showResult(message, type) {
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = message;
            resultDiv.className = type;
            resultDiv.style.display = 'block';
        }
        
        // Обработка загрузки файла
        const uploadArea = document.getElementById('uploadArea');
        const imageInput = document.getElementById('imageInput');
        const preview = document.getElementById('preview');
        const inferBtn = document.getElementById('inferBtn');
        
        uploadArea.onclick = () => imageInput.click();
        
        imageInput.addEventListener('change', handleImage);
        
        // Drag & Drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                imageInput.files = e.dataTransfer.files;
                handleImage();
            }
        });
        
        function handleImage() {
            const file = imageInput.files[0];
            if (!file) return;
            
            const reader = new FileReader();
            reader.onload = function(e) {
                preview.src = e.target.result;
                preview.style.display = 'block';
                imageLoaded = true;
                inferBtn.disabled = !session;
            };
            reader.readAsDataURL(file);
        }
        
        async function runInference() {
            if (!session || !imageLoaded) {
                alert('Please wait for the model to load and select an image!');
                return;
            }
            
            inferBtn.disabled = true;
            
            try {
                showResult('Processing image...', 'loading');
                
                // Создаем canvas для обработки изображения
                const canvas = document.createElement('canvas');
                canvas.width = 128;
                canvas.height = 128;
                const ctx = canvas.getContext('2d');
                
                // Рисуем и изменяем размер изображения
                const img = preview;
                ctx.drawImage(img, 0, 0, 128, 128);
                
                // Получаем pixel data
                const imageData = ctx.getImageData(0, 0, 128, 128);
                const pixels = imageData.data;
                
                // Преобразуем в тензор [1, 3, 128, 128] (NCHW format)
                const red = [];
                const green = [];
                const blue = [];
                
                for (let i = 0; i < pixels.length; i += 4) {
                    red.push(pixels[i] / 255.0);
                    green.push(pixels[i + 1] / 255.0);
                    blue.push(pixels[i + 2] / 255.0);
                }
                
                const tensorData = Float32Array.from([...red, ...green, ...blue]);
                const inputTensor = new ort.Tensor('float32', tensorData, [1, 3, 128, 128]);
                
                console.log('Input tensor shape:', inputTensor.dims);
                
                // Запускаем инференс
                const startTime = performance.now();
                const results = await session.run({ input: inputTensor });
                const inferenceTime = (performance.now() - startTime).toFixed(2);
                
                const output = results.output.data;
                console.log('Raw output:', output);
                
                // Применяем softmax для получения вероятностей
                const expScores = Array.from(output).map(x => Math.exp(x));
                const sumExp = expScores.reduce((a, b) => a + b, 0);
                const probabilities = expScores.map(x => x / sumExp);
                
                const class0Prob = probabilities[0] * 100;
                const class1Prob = probabilities[1] * 100;
                const predictedClass = class0Prob > class1Prob ? 0 : 1;
                const confidence = Math.max(class0Prob, class1Prob);
                
                // Показываем результаты
                showResult(
                    '<h3>Classification Results</h3>' +
                    '<div class="result-score">' +
                    '<strong>Class 0:</strong> ' +
                    '<span>' + class0Prob.toFixed(2) + '%</span>' +
                    '</div>' +
                    '<div class="progress-bar">' +
                    '<div class="progress-fill" style="width: ' + class0Prob + '%"></div>' +
                    '</div>' +
                    '<div class="result-score">' +
                    '<strong>Class 1:</strong> ' +
                    '<span>' + class1Prob.toFixed(2) + '%</span>' +
                    '</div>' +
                    '<div class="progress-bar">' +
                    '<div class="progress-fill" style="width: ' + class1Prob + '%"></div>' +
                    '</div>' +
                    '<hr>' +
                    '<p style="font-size: 24px; text-align: center; margin: 20px 0;">' +
                    '<strong>Predicted:</strong> ' +
                    '<span style="color: ' + (predictedClass === 0 ? '#2196F3' : '#F44336') + ';">' +
                    'Class ' + predictedClass +
                    '</span></p>' +
                    '<p style="text-align: center; color: #666;">' +
                    'Confidence: ' + confidence.toFixed(1) + '% | ' +
                    'Inference time: ' + inferenceTime + 'ms' +
                    '</p>',
                    'success'
                );
                
            } catch (e) {
                console.error('Inference failed:', e);
                showResult('✗ Classification failed: ' + e.message, 'error');
            } finally {
                inferBtn.disabled = false;
            }
        }
        
        // Загружаем модель при старте
        window.onload = loadModel;
    </script>
</body>
</html>"""

with open("test_onnx.html", "w", encoding="utf-8") as f:
    f.write(html_content)
print("Created test_onnx.html")

# Создаем простой манифест для Chrome Extension
manifest_json = """{
  "manifest_version": 3,
  "name": "AI Image Classifier",
  "version": "1.0",
  "description": "Image classification using ONNX model",
  "permissions": ["storage"],
  "web_accessible_resources": [
    {
      "resources": ["model_web.onnx"],
      "matches": ["<all_urls>"]
    }
  ],
  "content_scripts": [
    {
      "matches": ["<all_urls>"],
      "js": ["content.js"]
    }
  ]
}"""

with open("manifest_example.json", "w", encoding="utf-8") as f:
    f.write(manifest_json)
print("✓ Created manifest_example.json")

# Создаем пример использования в расширении
extension_code = """// Пример использования в Chrome Extension
// Загрузите onnxruntime-web: npm install onnxruntime-web

import * as ort from 'onnxruntime-web';

let session = null;

// Загрузка модели
async function loadModel() {
  try {
    const modelUrl = chrome.runtime.getURL('model_web.onnx');
    session = await ort.InferenceSession.create(modelUrl);
    console.log('Model loaded successfully');
  } catch (error) {
    console.error('Failed to load model:', error);
  }
}

// Классификация изображения
async function classifyImage(imageElement) {
  if (!session) {
    await loadModel();
  }
  
  // Создаем canvas для обработки
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 128;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(imageElement, 0, 0, 128, 128);
  
  // Получаем пиксели
  const imageData = ctx.getImageData(0, 0, 128, 128);
  const pixels = imageData.data;
  
  // Преобразуем в тензор [1, 3, 128, 128]
  const red = [], green = [], blue = [];
  for (let i = 0; i < pixels.length; i += 4) {
    red.push(pixels[i] / 255.0);
    green.push(pixels[i + 1] / 255.0);
    blue.push(pixels[i + 2] / 255.0);
  }
  
  const tensorData = Float32Array.from([...red, ...green, ...blue]);
  const inputTensor = new ort.Tensor('float32', tensorData, [1, 3, 128, 128]);
  
  // Инференс
  const results = await session.run({ input: inputTensor });
  const output = results.output.data;
  
  // Softmax для вероятностей
  const expScores = Array.from(output).map(x => Math.exp(x));
  const sumExp = expScores.reduce((a, b) => a + b, 0);
  const probabilities = expScores.map(x => x / sumExp);
  
  return {
    class0: probabilities[0],
    class1: probabilities[1],
    predicted: probabilities[0] > probabilities[1] ? 0 : 1
  };
}

// Инициализация
loadModel();

// Экспорт для использования
export { classifyImage };
"""

with open("extension_example.js", "w", encoding="utf-8") as f:
    f.write(extension_code)
print(" Created extension_example.js")

# === Финальные инструкции ===
print("\n" + "="*60)
print(" CONVERSION COMPLETED!")
print("="*60)
print(f"\n Your ONNX model: {onnx_path} ({file_size:.2f} MB)")
print("\n To test locally:")
print("  1. python -m http.server 8000")
print("  2. Open: http://localhost:8000/test_onnx.html")
print("\n For Chrome Extension:")
print("  1. Copy model_web.onnx to your extension folder")
print("  2. Install: npm install onnxruntime-web")
print("  3. See extension_example.js for usage")
print("  4. Update manifest.json (see manifest_example.json)")
print("\n ONNX Runtime Web is faster and easier than TensorFlow.js!")
print("="*60)