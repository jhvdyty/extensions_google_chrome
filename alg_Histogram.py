import cv2 
import numpy as np
import os
from scipy.spatial.distance import cosine 

def extract_color_histogram(image_path):
    """извлекаем цветовую гистограму"""
    img = cv2.imread(image_path)
    #для каждого канала 
    hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])

    #в один вектор
    hist = np.concatenate([hist_b, hist_g, hist_r]).flatten()
    #нормализуем 
    hist = hist / (hist.sum() + 1e-8)
    return hist 

def find_similar_by_histogram(target_image, image_folder, threshold=0.5):
    """поиск в гистограме"""
    target_hist = extract_color_histogram(target_image)
    similar_images = []


    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img_path = os.path.join(image_folder, filename)
                img_hist = extract_color_histogram(img_path)

                #сходство косинусов 
                similarity = 1 - cosine(target_hist, img_hist)
                if similarity >= threshold:
                    similar_images.append((img_path, similarity))
            except:
                #print("nothing")
                continue

    return sorted(similar_images, key=lambda x: x[1], reverse=True)

print(find_similar_by_histogram("input/2.jpg", "replacement", 0.3))     