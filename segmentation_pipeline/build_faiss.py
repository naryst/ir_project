import faiss
import torch
import os

def build_faiss(image_embeds, text_embeds):
    index_image = faiss.IndexFlatIP(image_embeds.shape[1])
    index_image.add(image_embeds)
    index_text = faiss.IndexFlatIP(text_embeds.shape[1])
    index_text.add(text_embeds)
    return index_image, index_text


MAX_FILES_PAD = 36 # max number of files in a directory

base_dir = 'masks'

mask_dirs = sorted(os.listdir(base_dir), key=lambda x: int(x.split('_')[1]))
print(mask_dirs) # check order

total_image_embeds = []
total_text_embeds = []
for mask_dir in mask_dirs:
    image_embeds = torch.load(os.path.join(base_dir, mask_dir, "image_embeds.pt"))
    text_embeds = torch.load(os.path.join(base_dir, mask_dir, "text_embeds.pt"))

    # pad with zeros
    if image_embeds.shape[0] < MAX_FILES_PAD:
        image_embeds = torch.cat([image_embeds, torch.zeros(MAX_FILES_PAD - image_embeds.shape[0], image_embeds.shape[1])])
        text_embeds = torch.cat([text_embeds, torch.zeros(MAX_FILES_PAD - text_embeds.shape[0], text_embeds.shape[1])])

    total_image_embeds.append(image_embeds)
    total_text_embeds.append(text_embeds)

total_image_embeds = torch.stack(total_image_embeds)
total_text_embeds = torch.stack(total_text_embeds)

total_image_embeds = total_image_embeds.reshape(-1, 512)
total_text_embeds = total_text_embeds.reshape(-1, 512)

print(total_image_embeds.shape, total_text_embeds.shape)

index_image, index_text = build_faiss(total_image_embeds, total_text_embeds)
faiss_dir = "faiss_index"
os.makedirs(faiss_dir, exist_ok=True)
faiss.write_index(index_image, os.path.join(faiss_dir, "index_image.bin"))
faiss.write_index(index_text, os.path.join(faiss_dir, "index_text.bin"))

    