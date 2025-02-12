import requests
import os
import re
import time
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

# Главная папка для всех изображений
BASE_FOLDER = "IMAGES"

# Список классов и подклассов
keywords = {
    "Animals": ["Bird", "Reptile", "Fish", "Mammals", "Insect"],
    "Objects": ["Furniture", "Electronics", "Vehicles", "Tools", "Dishes"],
    "Scenes": ["Urban landscapes", "Natural vistas", "Indoor settings", "Architectural structures", "Wastelands"],
    "Activities": ["Sports", "Cooking", "Dancing", "Working", "Studying"],
    "Emotions": ["Happiness", "Sadness", "Anger", "Surprise", "Suspicion"],
}

# Заголовки для запроса
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

# Базовый URL Google Images (добавили фильтр isz:l - большие изображения)
BASE_URL = "https://www.google.com/search?tbm=isch&tbs=isz:l&q="

# Создаем сессию для повторного использования соединения
session = requests.Session()


# Функция для очистки имени папки от запрещенных символов
def sanitize_filename(name):
    return re.sub(r'[^\w\-_]', '_', name)


# Функция загрузки изображений
def download_images(image_urls, save_folder, max_images=100):
    os.makedirs(save_folder, exist_ok=True)
    count = 0

    for img_url in image_urls:
        if count >= max_images:
            break

        try:
            response = session.get(img_url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                file_path = os.path.join(save_folder, f"{count + 1}.jpg")
                with open(file_path, "wb") as file:
                    file.write(response.content)
                count += 1
                print(f"[✓] Downloaded: {file_path}")
            else:
                print(f"[x] Failed to download: {img_url} (Status: {response.status_code})")

        except Exception as e:
            print(f"[!] Error downloading {img_url}: {e}")

    print(f"Downloaded {count} images in '{save_folder}'")


# Функция получения URL-адресов изображений (парсит 5 страниц по 20 картинок)
def fetch_image_urls(query, max_images=100):
    query_encoded = quote_plus(query)
    image_urls = []

    for start in range(0, max_images, 20):  # Получаем 5 страниц по 20 изображений
        search_url = f"{BASE_URL}{query_encoded}&start={start}"

        try:
            response = session.get(search_url, headers=HEADERS, timeout=10)
            if response.status_code != 200:
                print(f"[x] Failed to fetch images for '{query}' (Status: {response.status_code})")
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            img_tags = soup.find_all("img")

            for img in img_tags:
                src = img.get("data-src") or img.get("src")
                if src and src.startswith("http"):
                    image_urls.append(src)
                if len(image_urls) >= max_images:
                    break

        except Exception as e:
            print(f"Error fetching images for '{query}': {e}")
            break

        time.sleep(2)  # Ожидание между запросами

    return image_urls[:max_images]


# Основной цикл по классам и подклассам
for category, subclasses in keywords.items():
    category_folder = os.path.join(BASE_FOLDER, sanitize_filename(category))
    os.makedirs(category_folder, exist_ok=True)

    for subclass in subclasses:
        subclass_folder = os.path.join(category_folder, sanitize_filename(subclass))
        os.makedirs(subclass_folder, exist_ok=True)

        print(f"\nSearching for images: {category} > {subclass}")
        image_urls = fetch_image_urls(subclass, max_images=100)
        download_images(image_urls, subclass_folder, max_images=100)

        # Сон, чтобы избежать блокировки от Google
        time.sleep(5)

print("\nImage scraping complete!")
