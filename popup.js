let analyzedImages = [];

document.addEventListener('DOMContentLoaded', function() {
  chrome.storage.local.get(['replaceUrl'], function(result) {
    if (result.replaceUrl) {
      document.getElementById('replaceUrl').value = result.replaceUrl;
    }
  });
});

// Анализ изображений с AI
document.getElementById('analyzeImages').addEventListener('click', function() {
  const btn = this;
  btn.disabled = true;
  btn.textContent = 'Анализ...';
  
  document.getElementById('progressBar').style.display = 'block';
  document.getElementById('stats').style.display = 'none';
  
  chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
    chrome.tabs.sendMessage(tabs[0].id, {action: "analyzeImages"}, function(response) {
      btn.disabled = false;
      btn.textContent = 'Анализировать изображения с AI';
      document.getElementById('progressBar').style.display = 'none';
      
      if (response && response.success) {
        analyzedImages = response.images;
        displayAnalyzedImages(analyzedImages);
        showStats(analyzedImages);
        
        const lowQualityCount = analyzedImages.filter(img => img.isLowQuality).length;
        showStatus(`Анализ завершён! Найдено ${lowQualityCount} изображений низкого качества`, 'success');
      } else {
        showStatus('Ошибка анализа. Проверьте консоль.', 'error');
      }
    });
  });
});

// Прогресс анализа
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.action === "progress") {
    const percent = Math.round((request.current / request.total) * 100);
    const fill = document.getElementById('progressFill');
    fill.style.width = percent + '%';
    fill.textContent = `${request.current}/${request.total}`;
  }
});

// Замена изображений
document.getElementById('replaceSelected').addEventListener('click', function() {
  const replaceUrl = document.getElementById('replaceUrl').value;
  if (!replaceUrl) {
    showStatus('Введите URL изображения для замены', 'error');
    return;
  }
  
  chrome.storage.local.set({replaceUrl: replaceUrl});
  
  const lowQualityImages = analyzedImages.filter(img => img.isLowQuality);
  
  if (lowQualityImages.length === 0) {
    showStatus('Нет изображений низкого качества для замены', 'error');
    return;
  }
  
  chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
    chrome.tabs.sendMessage(tabs[0].id, {
      action: "replaceImages",
      images: lowQualityImages,
      newUrl: replaceUrl
    }, function(response) {
      if (response && response.success) {
        showStatus(`Заменено ${response.count} изображений низкого качества`, 'success');
      }
    });
  });
});

// Восстановление
document.getElementById('restoreOriginal').addEventListener('click', function() {
  chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
    chrome.tabs.sendMessage(tabs[0].id, {action: "restoreImages"}, function(response) {
      if (response && response.success) {
        showStatus('Изображения восстановлены', 'success');
      }
    });
  });
});

// Отображение результатов
function displayAnalyzedImages(images) {
  const listDiv = document.getElementById('imageList');
  listDiv.innerHTML = '';
  
  const lowQualityImages = images.filter(img => img.isLowQuality);
  
  if (lowQualityImages.length === 0) {
    listDiv.innerHTML = '<p style="text-align: center; color: #4CAF50;">✓ Все изображения высокого качества!</p>';
    return;
  }
  
  lowQualityImages.forEach((img) => {
    const itemDiv = document.createElement('div');
    itemDiv.className = `image-item ${img.isLowQuality ? 'low-quality' : 'high-quality'}`;
    
    const qualityLabel = img.isLowQuality ? 
      '<span class="quality-badge low-quality-badge">LOW QUALITY</span>' :
      '<span class="quality-badge high-quality-badge">HIGH QUALITY</span>';
    
    itemDiv.innerHTML = `
      <img src="${img.src}" onerror="this.style.display='none'">
      <div class="image-info">
        <div>
          Размер: ${img.width}x${img.height}
          ${qualityLabel}
        </div>
        <div class="confidence">Уверенность: ${(img.confidence * 100).toFixed(1)}%</div>
        <div style="color: #999; font-size: 11px; margin-top: 3px;">${img.src.substring(0, 50)}...</div>
      </div>
    `;
    
    listDiv.appendChild(itemDiv);
  });
}

// Статистика
function showStats(images) {
  const statsDiv = document.getElementById('stats');
  statsDiv.style.display = 'flex';
  
  const total = images.length;
  const lowQuality = images.filter(img => img.isLowQuality).length;
  const highQuality = total - lowQuality;
  
  document.getElementById('totalCount').textContent = total;
  document.getElementById('lowQualityCount').textContent = lowQuality;
  document.getElementById('highQualityCount').textContent = highQuality;
}

function showStatus(message, type) {
  const statusDiv = document.getElementById('status');
  statusDiv.textContent = message;
  statusDiv.className = type;
  
  setTimeout(() => {
    statusDiv.textContent = '';
    statusDiv.className = '';
  }, 4000);
}