import PIL
from PIL import Image
from transformers import AutoModel
from densely_captioned_images.dataset.dense_image import get_dci_count
from densely_captioned_images.dataset.dense_image import DenseCaptionedImage
from tqdm import tqdm
import torch
import gc
import os
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants and configuration
MASK_PADDING = 104  # value to pad tensors on mask dimension
BATCH_SIZE = 50
SAVE_EVERY = 1000
DEFAULT_SAVE_PATH = '/root/ir_project/dense_index_v2/embeds'
TRUNCATE_DIM = 512

def parse_args():
    parser = argparse.ArgumentParser(description='Generate JINA embeddings for dense captioned images')
    parser.add_argument('--save_path', type=str, default=DEFAULT_SAVE_PATH, help='Path to save embeddings')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for processing')
    parser.add_argument('--save_every', type=int, default=SAVE_EVERY, help='Save embeddings every N images')
    parser.add_argument('--resume_from', type=int, default=0, help='Resume processing from this image index')
    return parser.parse_args()

def load_model():
    """Load the JINA model with proper error handling"""
    try:
        logger.info("Loading JINA model...")
        model = AutoModel.from_pretrained(
            "jinaai/jina-clip-v2", 
            trust_remote_code=True, 
            device_map="cuda"
        )
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def clear_cache():
    """Clear CUDA cache and garbage collect"""
    torch.cuda.empty_cache()
    gc.collect()

def process_batch(model, dci, batch_masks):
    """Process a batch of masks and return their embeddings"""
    try:
        batch_samples = []
        for mask in batch_masks:
            sample = dci.get_caption_with_subcaptions(mask)
            batch_samples.append(sample)

        batch_images = [PIL.Image.fromarray(sample[0]['image'].astype("uint8"), "RGB") for sample in batch_samples]
        batch_descriptions = [sample[0]['caption'] for sample in batch_samples]
        
        with torch.no_grad():
            batch_image_embeddings = model.encode_image(batch_images, truncate_dim=TRUNCATE_DIM)
            batch_description_embeddings = model.encode_text(batch_descriptions, truncate_dim=TRUNCATE_DIM)
            
        return (
            torch.tensor(batch_image_embeddings, device="cuda"),
            torch.tensor(batch_description_embeddings, device="cuda")
        )
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        raise

def save_embeddings(image_embeds, description_embeds, save_path, save_iter, final=False):
    """Save embeddings to disk"""
    try:
        suffix = "final" if final else str(save_iter)
        logger.info(f"Saving embeddings batch {suffix}")
        
        # Move tensors to CPU before saving
        image_embeds = image_embeds.cpu()
        description_embeds = description_embeds.cpu()
        
        torch.save(image_embeds, f"{save_path}/image_embeds_{suffix}.pt")
        torch.save(description_embeds, f"{save_path}/description_embeds_{suffix}.pt")
        
        # Help garbage collection
        del image_embeds
        del description_embeds
        clear_cache()
    except Exception as e:
        logger.error(f"Error saving embeddings: {str(e)}")
        raise

def main():
    args = parse_args()
    
    # Ensure save directory exists
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    
    # Load model
    model = load_model()
    
    # Get total number of images to process
    try:
        total_dcis = get_dci_count()
        logger.info(f"Processing {total_dcis} dense captioned images")
    except Exception as e:
        logger.error(f"Failed to get DCI count: {str(e)}")
        return
    
    total_image_embeds = []
    total_description_embeds = []
    save_iter = args.resume_from // args.save_every
    
    # Process each image
    for j in tqdm(range(args.resume_from, total_dcis)):
        try:
            dci = DenseCaptionedImage(img_id=j)
            all_masks = dci.get_all_masks()
            image_embeds = []
            description_embeds = []
            image_embeds_count = 0
            
            # Check if we would exceed mask padding limit
            if len(all_masks) > MASK_PADDING:
                logger.warning(f"Image {j} has {len(all_masks)} masks, exceeding limit of {MASK_PADDING}")
                all_masks = all_masks[:MASK_PADDING]
            
            # Process masks in batches
            for i in range(0, len(all_masks), args.batch_size):
                batch_masks = all_masks[i:i+args.batch_size]
                
                # Don't process if we'd exceed the mask padding
                if image_embeds_count + len(batch_masks) > MASK_PADDING:
                    logger.warning(f"Stopping at {image_embeds_count} masks for image {j} to avoid exceeding padding")
                    break
                
                batch_image_embeddings, batch_description_embeddings = process_batch(model, dci, batch_masks)
                
                image_embeds.append(batch_image_embeddings)
                description_embeds.append(batch_description_embeddings)
                image_embeds_count += len(batch_masks)
            
            # Concatenate and pad embeddings
            if image_embeds:
                image_embeds = torch.cat(image_embeds, dim=0)
                description_embeds = torch.cat(description_embeds, dim=0)
                
                # Pad dimension 0 to MASK_PADDING
                image_embeds = torch.cat(
                    [
                        image_embeds, 
                        torch.zeros(MASK_PADDING - image_embeds.shape[0], *image_embeds.shape[1:], device=image_embeds.device)
                    ], 
                    dim=0
                )
                description_embeds = torch.cat(
                    [
                        description_embeds, 
                        torch.zeros(MASK_PADDING - description_embeds.shape[0], *description_embeds.shape[1:], device=description_embeds.device)
                    ], 
                    dim=0
                )
                
                total_image_embeds.append(image_embeds)
                total_description_embeds.append(description_embeds)
            
            # Clear CUDA cache after each image
            clear_cache()
            
            # Save periodically
            if j > 0 and (j + 1) % args.save_every == 0:
                if total_image_embeds:
                    stacked_image_embeds = torch.stack(total_image_embeds)
                    stacked_description_embeds = torch.stack(total_description_embeds)
                    save_embeddings(stacked_image_embeds, stacked_description_embeds, save_path, save_iter)
                    save_iter += 1
                    total_image_embeds = []
                    total_description_embeds = []
        
        except Exception as e:
            logger.error(f"Error processing image {j}: {str(e)}")
            # Continue with next image rather than failing completely
            continue
    
    # Save any remaining embeddings
    if total_image_embeds:
        stacked_image_embeds = torch.stack(total_image_embeds)
        stacked_description_embeds = torch.stack(total_description_embeds)
        save_embeddings(stacked_image_embeds, stacked_description_embeds, save_path, save_iter, final=True)

if __name__ == "__main__":
    main()


