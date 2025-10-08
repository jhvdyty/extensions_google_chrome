import cv2 
import numpy as np
import os
from scipy.spatial.distance import cosine 

from duckduckgo_SearchToRep import DuckDuckGoImageDownloader

def extract_color_histogram(image_path):
    """извлекаем цветовую гистограму"""
    img = cv2.imread(image_path)
    if img is None:
        return None
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
    if target_hist is None:
        return []
    
    similar_images = []


    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img_path = os.path.join(image_folder, filename)
                if img_path == target_image:
                    continue

                img_hist = extract_color_histogram(img_path)
                #сходство косинусов 
                if img_hist is not None:
                    similarity = 1 - cosine(target_hist, img_hist)
                    if similarity >= threshold:
                        similar_images.append((img_path, similarity))
            except Exception as e:
                #print("nothing")
                continue

    return sorted(similar_images, key=lambda x: x[1], reverse=True)

print("image download")
#print(find_similar_by_histogram("input/2.jpg", "replacement", 0.3))  

print(find_similar_by_histogram("input/ugliest-people-in-the-world-8-62628152.jpg", "replacement", 0.6))     

downloader = DuckDuckGoImageDownloader()
downloaded = downloader.search_and_download(
    query="fat girl ",
    image_folder="data/train/low_quality",
    count=200 
)

print(find_similar_by_histogram("input/ugliest-people-in-the-world-8-62628152.jpg", "replacement", 0.8))  