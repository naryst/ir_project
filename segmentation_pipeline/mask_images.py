import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
import argparse
from tqdm import tqdm

min_area_ratio = 0.01  # 1% of the image
max_area_ratio = 0.9   # to filter out huge masks that might be background

def load_model():
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator

def slice_image(cv2_img, x1, y1, x2, y2):
    # Slice the image using the coordinates
    # Note: In OpenCV, y comes first in the slicing
    sliced_img = cv2_img[y1:y2, x1:x2]
    return sliced_img

def save_masks(masks: list, image_path: str, save_dir: str, image: np.ndarray, image_subfolder: str):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/{image_subfolder}", exist_ok=True)
    image_area = image.shape[0] * image.shape[1]
    for i, mask in enumerate(masks):
        coords = mask['bbox']
        mask_area = mask['area']
        # filter out masks that are too small or too large
        if mask_area < min_area_ratio * image_area or mask_area > max_area_ratio * image_area:
            continue
        croped_img = slice_image(image, x1=coords[0], y1=coords[1], x2=coords[0]+coords[2], y2=coords[1]+coords[3])
        cv2.imwrite(f"{save_dir}/{image_subfolder}/{image_path.split('.')[0]}_{i}.png", croped_img)

def mask_image(image_path: str, mask_generator: SamAutomaticMaskGenerator):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    return masks

def main(args):
    mask_generator = load_model()
    images_dir = args.images_dir
    save_dir = args.save_dir
    for image_path in tqdm(sorted(os.listdir(images_dir))):
        image_subfolder = image_path.split('.')[0]
        image = cv2.imread(os.path.join(images_dir, image_path))
        masks = mask_image(os.path.join(images_dir, image_path), mask_generator)
        save_masks(masks, image_path, save_dir, image, image_subfolder)
        # save whole image
        cv2.imwrite(f"{save_dir}/{image_subfolder}/{image_path.split('.')[0]}_whole.png", image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, default="images")
    parser.add_argument("--save_dir", type=str, default="masks")
    args = parser.parse_args()
    main(args)