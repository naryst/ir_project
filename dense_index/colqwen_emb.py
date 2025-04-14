import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available
from torch.nn.functional import cosine_similarity
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
import os
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

model_name = "nomic-ai/colnomic-embed-multimodal-7b"

model = ColQwen2_5.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()

processor = ColQwen2_5_Processor.from_pretrained(model_name, use_fast=True)

annotations_path = "/home/nikita/ir_project/DCI/data/densely_captioned_images/annotations"
photos_path = "/home/nikita/ir_project/DCI/data/densely_captioned_images/photos"

def load_single_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[!] Ошибка при загрузке изображения {path}: {e}")
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

class EmbeddingDataset(Dataset):
    def __init__(self, data_items, processor):
        self.data_items = data_items
        self.processor = processor

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        _, text, image_path = self.data_items[idx]
        image = load_single_image(image_path)
        return image, text

def collate_fn(batch):
    images, texts = zip(*batch)
    return list(images), list(texts)

data = load_data(annotations_path, photos_path)

batch_size = 1
SAVE_EVERY = 1000
num_workers = min(8, multiprocessing.cpu_count())

dataset = EmbeddingDataset(data, processor)
dataloader = DataLoader(
    dataset, 
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    collate_fn=collate_fn,
    pin_memory=True
)

total_items = len(data)
images_embeddings = []
queries_embeddings = []
count = 0
partition = 1

for images, queries in tqdm(dataloader, desc="Computing embeddings"):
    with torch.amp.autocast('cuda'):  # Use mixed precision
        batch_images = processor.process_images(images).to(model.device)
        batch_queries = processor.process_queries(queries).to(model.device)
        
        with torch.no_grad():
            image_embeddings = model(**batch_images)
            query_embeddings = model(**batch_queries)
        
        image_embeddings = image_embeddings.mean(dim=1)
        query_embeddings = query_embeddings.mean(dim=1)
    
    # normalized_query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    # normalized_image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=1)
    # sim = cosine_similarity(normalized_query_embeddings, normalized_image_embeddings)
    
    images_embeddings.extend(image_embeddings.cpu())
    queries_embeddings.extend(query_embeddings.cpu())
    
    count += len(images)
    if count > SAVE_EVERY:
        print(f"Saving embeddings at {count}...")
        images_embeddings_tensor = torch.stack(images_embeddings, dim=0)
        queries_embeddings_tensor = torch.stack(queries_embeddings, dim=0)
        torch.save(images_embeddings_tensor, f"results/images_embeddings_tensor_{partition}.pt")
        torch.save(queries_embeddings_tensor, f"results/queries_embeddings_tensor_{partition}.pt")
        images_embeddings = []
        queries_embeddings = []
        partition += 1
        count = 0

# save remaining embeddings
if images_embeddings:
    images_embeddings_tensor = torch.stack(images_embeddings, dim=0)
    queries_embeddings_tensor = torch.stack(queries_embeddings, dim=0)
    torch.save(images_embeddings_tensor, "results/images_embeddings_tensor_final.pt")
    torch.save(queries_embeddings_tensor, "results/queries_embeddings_tensor_final.pt")
