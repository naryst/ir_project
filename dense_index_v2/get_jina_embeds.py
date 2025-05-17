import PIL
from PIL import Image
from transformers import AutoModel
from densely_captioned_images.dataset.dense_image import get_dci_count
from densely_captioned_images.dataset.dense_image import DenseCaptionedImage
from tqdm import tqdm
import torch
import gc

model = AutoModel.from_pretrained(
    "jinaai/jina-clip-v2", 
    trust_remote_code=True, 
    device_map="mps"
)
model.eval()

truncate_dim = 512
TOTAL_DCIS = get_dci_count()

annotations_path = "/Users/sergeevnikita/edu/ir_project/DCI/data/densely_captioned_images/annotations"
photos_path = "/Users/sergeevnikita/edu/ir_project/DCI/data/densely_captioned_images/photos"

MASK_PADDING = 104 # value to pad tensors on mask dimension
BATCH_SIZE = 30

def clear_cache():
    torch.cuda.empty_cache()
    gc.collect()

for i in tqdm(range(TOTAL_DCIS)):
    dci = DenseCaptionedImage(img_id=i)
    all_masks = dci.get_all_masks()
    image_embeds = []
    description_embeds = []
    image_embeds_count = 0
    for i in range(0, len(all_masks), BATCH_SIZE):
        batch_masks = all_masks[i:i+BATCH_SIZE]
        batch_samples = []
        for mask in batch_masks:
            sample = dci.get_caption_with_subcaptions(mask)
            batch_samples.append(sample)

        batch_images = [PIL.Image.fromarray(sample[0]['image'].astype("uint8"), "RGB") for sample in batch_samples]
        batch_descriptions = [sample[0]['caption'] for sample in batch_samples]
        with torch.no_grad():
            batch_image_embeddings = model.encode_image(batch_images, truncate_dim=truncate_dim)
            batch_description_embeddings = model.encode_text(batch_descriptions, truncate_dim=truncate_dim)
        batch_image_embeddings = torch.tensor(batch_image_embeddings)
        batch_description_embeddings = torch.tensor(batch_description_embeddings)

        image_embeds.append(batch_image_embeddings)
        description_embeds.append(batch_description_embeddings)
        if image_embeds_count > MASK_PADDING:
            print(f"Warning: {i} has more than {MASK_PADDING} truncated embeddings")
            break
        image_embeds_count += len(batch_masks)

    image_embeds = torch.cat(image_embeds, dim=0)
    description_embeds = torch.cat(description_embeds, dim=0)
    
    # pad dim 0 to MASK_PADDING
    image_embeds = torch.cat(
        [
            image_embeds, 
            torch.zeros(MASK_PADDING - image_embeds.shape[0], *image_embeds.shape[1:])
        ], 
        dim=0
    )
    description_embeds = torch.cat(
        [
            description_embeds, 
            torch.zeros(MASK_PADDING - description_embeds.shape[0], *description_embeds.shape[1:])
        ], 
        dim=0
    )
    print(image_embeds.shape, description_embeds.shape)
    clear_cache()
    


