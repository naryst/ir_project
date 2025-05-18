import os
import logging
from transformers import AutoModel
from PIL import Image
import torch
from tqdm import tqdm
logger = logging.getLogger(__name__)

def load_model():
    """Load the JINA model with proper error handling"""
    try:
        logger.info("Loading JINA model...")
        model = AutoModel.from_pretrained(
            "jinaai/jina-clip-v2", 
            trust_remote_code=True, 
            device_map="mps"
        )
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


images_dir = "masks"
mask_dirs = sorted(os.listdir(images_dir))

model = load_model()
TRUNCATE_DIM = 512
for mask_dir in tqdm(mask_dirs):
    current_path = os.path.join(images_dir, mask_dir)
    images = sorted(os.listdir(current_path))
    images = [image for image in images if image.endswith(".png")]
    image_paths = [os.path.join(current_path, image) for image in images]
    texts = sorted(os.listdir(current_path))
    texts = [text for text in texts if text.endswith(".txt")]
    text_paths = [os.path.join(current_path, text) for text in texts]

    assert len(image_paths) == len(text_paths)

    dir_image_embeds = []
    dir_text_embeds = []
    for image_path, text_path in tqdm(zip(image_paths, text_paths)):
        image = Image.open(image_path)
        text = open(text_path, "r").read()
        image_embeds = model.encode_image(image, truncate_dim=TRUNCATE_DIM)
        text_embeds = model.encode_text(text, truncate_dim=TRUNCATE_DIM)
        image_embeds = torch.tensor(image_embeds)
        text_embeds = torch.tensor(text_embeds)
        dir_image_embeds.append(image_embeds)
        dir_text_embeds.append(text_embeds)
    
    dir_image_embeds = torch.stack(dir_image_embeds)
    dir_text_embeds = torch.stack(dir_text_embeds)

    torch.save(dir_image_embeds, os.path.join(current_path, "image_embeds.pt"))
    torch.save(dir_text_embeds, os.path.join(current_path, "text_embeds.pt"))