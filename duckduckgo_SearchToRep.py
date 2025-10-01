from ddgs import DDGS
import requests
from PIL import Image
from io import BytesIO
import os 
import time

class DuckDuckGoImageDownloader:
    def search_and_download(self, query, image_folder, count=20):
        """Поиск и загрузка через DuckDuckGo"""
        os.makedirs(image_folder, exist_ok=True)

        with DDGS() as ddgs:
            results = list(ddgs.images(
                query,
                max_results=count,
                safesearch="moderate"
            ))



        downloaded = []
        for i, result in enumerate(results):
            try:
                image_url = result['image']
                filename = f"{query.replace(' ', '_')}_{i+1}.jpg"
                save_path = os.path.join(image_folder, filename)

                print(f"dowunload {i+1}/{len(results)}: {filename}")

                #print(f"save path: {save_path}")

                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(image_url, headers=headers, timeout=10)

                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(save_path)
                    downloaded.append(save_path)
                time.sleep(1)
            except Exception as e:
                print(f"error {i+1}: {e}")
                continue
        
        return downloaded