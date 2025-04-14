import os
import json
import torch
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, AutoModel, AutoTokenizer, pipeline
from transformers import CLIPModel, CLIPProcessor
import numpy as np
from concurrent.futures import ThreadPoolExecutor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Используемое устройство:", device)

torch.backends.cudnn.benchmark = True

clip_model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(clip_model_name, use_fast=True)
model = CLIPModel.from_pretrained(clip_model_name).to(device)

print("Есть метод get_text_features:", hasattr(model, 'get_text_features'))
print("Есть метод get_image_features:", hasattr(model, 'get_image_features'))

tokenizer = AutoTokenizer.from_pretrained(clip_model_name)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

MAX_TOKENS = 64  # ориентировочно для CLIP, можно по длине текста

annotations_path = "D:/ir_project/DCI/data/densely_captioned_images/annotations"
photos_path = "D:/ir_project/DCI/data/densely_captioned_images/photos"


def load_single_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[!] Ошибка при загрузке изображения {path}: {e}")
        # Возвращаем заглушку: серая картинка 224x224
        return Image.new("RGB", (224, 224), color="gray")

def load_data(annotations_path, photos_path):
    data_items = []
    annotation_files = [f for f in os.listdir(annotations_path) if f.endswith('.json')]

    for annotation_file in tqdm(annotation_files, desc="Подготовка данных"):
        try:
            with open(os.path.join(annotations_path, annotation_file), "r", encoding="utf-8") as f:
                data = json.load(f)
            text = (data.get("short_caption", "") + " " + data.get("extra_caption", "")).strip()
            if not text or len(text) < 3:
                continue
            image_name = data["image"]
            image_path = os.path.join(photos_path, image_name)
            if os.path.exists(image_path):
                data_items.append((image_name, text, image_path))
        except Exception as e:
            print(f"[!] Ошибка обработки файла {annotation_file}: {e}")
    return data_items


def get_text_embeddings(texts, summarizer_batch_size=4):
    processed_texts = []
    texts_to_summarize = []
    summarize_indices = []

    for idx, text in enumerate(texts):
        if text is None or not isinstance(text, str) or text.strip() == "":
            print(f"[!] Пустой текст на позиции {idx}. Заменяем на заглушку.")
            processed_texts.append((idx, "no caption"))
            continue

        if len(text.split()) > MAX_TOKENS:
            print(f"Текст превышает {MAX_TOKENS} токенов. Подготовка к суммаризации.")
            texts_to_summarize.append(text)
            summarize_indices.append(idx)
        else:
            processed_texts.append((idx, text))

    summarized_texts = []
    if texts_to_summarize:
        try:
            summarized_outputs = summarizer(
                texts_to_summarize,
                max_length=MAX_TOKENS,
                min_length=20,
                batch_size=summarizer_batch_size,
                do_sample=False
            )
            torch.cuda.empty_cache()
            summarized_texts = [s['summary_text'] for s in summarized_outputs]
        except Exception as e:
            print(f"[!] Ошибка при суммаризации: {e}")
            summarized_texts = ["no caption"] * len(summarize_indices)

    for idx, summary in zip(summarize_indices, summarized_texts):
        processed_texts.append((idx, summary))

    processed_texts = [text for idx, text in sorted(processed_texts, key=lambda x: x[0])]

    try:
        inputs = processor(text=processed_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            embeddings = model.get_text_features(**inputs)
        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
        return embeddings.cpu().numpy()
    except Exception as e:
        print(f"[!] Ошибка при извлечении текстовых эмбеддингов: {e}")
        out_dim = model.text_projection.out_features if hasattr(model, 'text_projection') else model.text_projection.weight.shape[1]
        return np.zeros((len(processed_texts), out_dim), dtype=np.float32)


def get_image_embeddings(image_paths, num_workers=8):
    # Параллельная загрузка изображений
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        images = list(executor.map(load_single_image, image_paths))

    try:
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            embeddings = model.get_image_features(**inputs)
        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
        return embeddings.cpu().numpy()
    except Exception as e:
        print(f"[!] Ошибка при извлечении эмбеддингов изображений: {e}")
        out_dim = (
            model.visual_projection.out_features
            if hasattr(model, 'visual_projection')
            else model.visual_projection.weight.shape[1]
        )
        return np.zeros((len(images), out_dim), dtype=np.float32)


def process_all_data_batch(data_items, batch_size=4):
    embeddings = []
    for i in tqdm(range(0, len(data_items), batch_size), desc="Извлечение эмбеддингов"):
        batch = data_items[i:i + batch_size]
        image_names, texts, image_paths = zip(*batch)

        text_embeds = get_text_embeddings(texts)
        image_embeds = get_image_embeddings(image_paths)

        torch.cuda.empty_cache()
        for img_name, txt_embed, img_embed in zip(image_names, text_embeds, image_embeds):
            embeddings.append({
                "id": f"text_{img_name.split('.')[0]}",
                "embedding": txt_embed,
                "type": "text"
            })
            embeddings.append({
                "id": f"image_{img_name.split('.')[0]}",
                "embedding": img_embed,
                "type": "image"
            })

    return embeddings



if __name__ == "__main__":
    # Запуск
    data_items = load_data(annotations_path, photos_path)

    embeddings = process_all_data_batch(data_items, batch_size=8)

    # Сохранение
    import pickle
    with open("embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    print(f"Обработано пар: {len(embeddings) // 2}")


def encode_text(text: str) -> np.ndarray:
    try:
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            embedding = model.get_text_features(**inputs)
        embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
        return embedding[0].cpu().numpy()
    except Exception as e:
        print(f"[!] Ошибка при кодировании текста: {e}")
        out_dim = model.text_projection.out_features if hasattr(model, 'text_projection') else model.text_projection.weight.shape[1]
        return np.zeros((out_dim,), dtype=np.float32)