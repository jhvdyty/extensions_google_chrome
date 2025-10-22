let model = null;
let originalImages = new Map();
const IMAGE_SIZE = 224;
const QUALITY_THRESHOLD = 0.5; // порог для определения низкого качества

// Загрузка TensorFlow.js модели
async function loadModel() {
  if (model) return model;
  
  try {
    const modelUrl = chrome.runtime.getURL('model_tfjs/model.json');
    model = await tf.loadGraphModel(modelUrl);
    console.log('✓ AI Model loaded successfully');
    return model;
  } catch (error) {
    console.error('Failed to load model:', error);
    return null;
  }
}

// Предобработка изображения
function preprocessImage(imageElement) {
  return tf.tidy(() => {
    // Создаём canvas для обработки изображения
    const canvas = document.createElement('canvas');
    canvas.width = IMAGE_SIZE;
    canvas.height = IMAGE_SIZE;
    const ctx = canvas.getContext('2d');
    
    // Рисуем изображение на canvas с изменением размера
    ctx.drawImage(imageElement, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
    
    // Конвертируем в тензор
    let tensor = tf.browser.fromPixels(canvas);
    
    // Нормализация: [0,255] -> [0,1]
    tensor = tensor.toFloat().div(tf.scalar(255.0));
    
    // Добавляем batch dimension: [224, 224, 3] -> [1, 224, 224, 3]
    tensor = tensor.expandDims(0);
    
    return tensor;
  });
}

// Классификация одного изображения
async function classifyImage(imageElement) {
  try {
    if (!model) {
      await loadModel();
    }
    
    if (!model) {
      return { isLowQuality: false, confidence: 0 };
    }
    
    const tensor = preprocessImage(imageElement);
    const predictions = await model.predict(tensor);
    const probabilities = await tf.softmax(predictions).data();
    
    // probabilities[0] = Low Quality, probabilities[1] = High Quality
    const isLowQuality = probabilities[0] > QUALITY_THRESHOLD;
    const confidence = probabilities[0];
    
    // Очистка памяти
    tensor.dispose();
    predictions.dispose();
    
    return {
      isLowQuality: isLowQuality,
      confidence: confidence
    };
  } catch (error) {
    console.error('Classification error:', error);
    return { isLowQuality: false, confidence: 0 };
  }
}

// Обработка всех изображений на странице
async function analyzeAllImages(progressCallback) {
  const imgElements = Array.from(document.querySelectorAll('img'));
  const results = [];
  
  for (let i = 0; i < imgElements.length; i++) {
    const img = imgElements[i];
    
    // Пропускаем маленькие изображения (иконки и т.д.)
    if (img.naturalWidth < 100 || img.naturalHeight < 100) {
      continue;
    }
    
    // Ждём полной загрузки изображения
    if (!img.complete) {
      await new Promise(resolve => {
        img.onload = resolve;
        img.onerror = resolve;
      });
    }
    
    const result = await classifyImage(img);
    
    results.push({
      element: img,
      src: img.src,
      width: img.naturalWidth,
      height: img.naturalHeight,
      isLowQuality: result.isLowQuality,
      confidence: result.confidence,
      index: i
    });
    
    // Отправляем прогресс
    if (progressCallback) {
      progressCallback(i + 1, imgElements.length);
    }
  }
  
  return results;
}

// Замена изображений
function replaceImages(imagesToReplace, newUrl) {
  let count = 0;
  
  imagesToReplace.forEach(imgInfo => {
    const img = imgInfo.element || document.querySelector(`img[src="${imgInfo.src}"]`);
    if (img) {
      if (!originalImages.has(img)) {
        originalImages.set(img, img.src);
      }
      img.src = newUrl;
      if (img.srcset) {
        img.srcset = newUrl;
      }
      count++;
    }
  });
  
  return count;
}

// Восстановление оригинальных изображений
function restoreOriginalImages() {
  originalImages.forEach((originalSrc, element) => {
    if (element.tagName === 'IMG') {
      element.src = originalSrc;
    } else {
      element.style.backgroundImage = originalSrc;
    }
  });
  originalImages.clear();
}

// Обработка сообщений от popup
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.action === "analyzeImages") {
    (async () => {
      const results = await analyzeAllImages((current, total) => {
        chrome.runtime.sendMessage({
          action: "progress",
          current: current,
          total: total
        });
      });
      sendResponse({ success: true, images: results });
    })();
    return true; // асинхронный ответ
    
  } else if (request.action === "replaceImages") {
    const count = replaceImages(request.images, request.newUrl);
    sendResponse({ success: true, count: count });
    
  } else if (request.action === "restoreImages") {
    restoreOriginalImages();
    sendResponse({ success: true });
  }
  
  return true;
});

// Загружаем модель при загрузке страницы
loadModel();