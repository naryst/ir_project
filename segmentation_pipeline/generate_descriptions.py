from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import torch
import os
import argparse
from tqdm import tqdm

def load_model():

    model_id = "google/gemma-3-4b-it"
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="mps"
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

    return model, processor

def generate_description(model, processor, image_path):
    image = Image.open(image_path)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant. Your task is to create a CLIP-like descriptions for the images. You should response only with descriptions without any additional information."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in detail."}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=150, do_sample=True)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)

    return decoded

def save_description(description, images_dir, image_path):
    with open(os.path.join(images_dir, image_path.split('.')[0] + ".txt"), "w") as f:
        f.write(description)

def generate_descriptions(model, processor, images_dir):
    images = sorted(os.listdir(images_dir))
    images = [image for image in images if image.endswith(".png")]
    for image_path in tqdm(images):
        description = generate_description(model, processor, os.path.join(images_dir, image_path))
        save_description(description, images_dir, image_path)

def main(args):
    model, processor = load_model()
    masked_imgs_dir = sorted(os.listdir(args.images_dir))
    for masked_img_path in masked_imgs_dir:
        print(f"Generating descriptions for {masked_img_path}")
        generate_descriptions(model, processor, os.path.join(args.images_dir, masked_img_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, default="masks")
    args = parser.parse_args()
    main(args)