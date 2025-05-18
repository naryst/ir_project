import requests
import os
import time
import argparse

def download_image(download_dir: str, file_name: str, width: int, height: int):
    os.makedirs(download_dir, exist_ok=True)
    addr = f"https://picsum.photos/{width}/{height}"
    response = requests.get(addr)
    with open(f"{download_dir}/{file_name}.jpg", "wb") as f:
        f.write(response.content)
    time.sleep(1)

def download_images(n: int, download_dir: str, file_name: str, width: int, height: int):
    for i in range(n):
        download_image(download_dir, file_name, width, height)

def main(args):
    download_images(args.n, args.download_dir, args.file_name, args.width, args.height)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--download_dir", type=str, default="images")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--file_name", type=str, default="image")
    args = parser.parse_args()
    main(args)