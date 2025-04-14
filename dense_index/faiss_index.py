import pickle
import faiss
import numpy as np
import torch
import os
from glob import glob
from tqdm import tqdm

# Folder containing PyTorch embeddings
embeddings_folder = "embeddings_qwen"
print(f"Loading embeddings from: {embeddings_folder}")

# Output folder for all FAISS index files
output_folder = "faiss_index"
os.makedirs(output_folder, exist_ok=True)
print(f"Output will be saved to: {output_folder}")

# Find all tensor files
image_files = sorted(glob(os.path.join(embeddings_folder, "images_embeddings_tensor_*.pt")))
query_files = sorted(glob(os.path.join(embeddings_folder, "queries_embeddings_tensor_*.pt")))

print(f"Found {len(image_files)} image embedding files and {len(query_files)} query embedding files")

# Process image embeddings
image_vectors = []
total_images = 0

for file_path in tqdm(image_files, desc="Loading image embeddings"):
    tensor = torch.load(file_path)
    vectors = tensor.cpu().numpy().astype('float32')
    image_vectors.append(vectors)
    total_images += vectors.shape[0]

# Stack all image vectors into a single array
image_vectors_array = np.vstack(image_vectors).astype('float32')
print(f"Loaded {len(image_vectors_array)} image vectors with shape {image_vectors_array.shape}")

# Generate sequential IDs for all image vectors
image_ids = [f"img_{i}" for i in range(total_images)]

# Process query embeddings
query_vectors = []
total_queries = 0

for file_path in tqdm(query_files, desc="Loading query embeddings"):
    tensor = torch.load(file_path)
    vectors = tensor.cpu().numpy().astype('float32')
    query_vectors.append(vectors)
    total_queries += vectors.shape[0]

# Stack all query vectors into a single array
query_vectors_array = np.vstack(query_vectors).astype('float32')
print(f"Loaded {len(query_vectors_array)} query vectors with shape {query_vectors_array.shape}")

# Generate sequential IDs for all query vectors
query_ids = [f"qry_{i}" for i in range(total_queries)]

# Normalize image vectors
image_vectors_array = image_vectors_array / np.linalg.norm(image_vectors_array, axis=1, keepdims=True)
image_vectors_array = image_vectors_array.astype("float32")

print("Image vectors dtype:", image_vectors_array.dtype)
print("Image vectors - Contains NaN:", np.isnan(image_vectors_array).any())
print("Image vectors - Contains inf:", np.isinf(image_vectors_array).any())

# Normalize query vectors
query_vectors_array = query_vectors_array / np.linalg.norm(query_vectors_array, axis=1, keepdims=True)
query_vectors_array = query_vectors_array.astype("float32")

print("Query vectors dtype:", query_vectors_array.dtype)
print("Query vectors - Contains NaN:", np.isnan(query_vectors_array).any())
print("Query vectors - Contains inf:", np.isinf(query_vectors_array).any())

# Build FAISS index for images
image_dim = image_vectors_array.shape[1]
print("Image embedding dimension:", image_dim)

image_index = faiss.IndexFlatIP(image_dim)

# Build FAISS index for queries
query_dim = query_vectors_array.shape[1]
print("Query embedding dimension:", query_dim)

query_index = faiss.IndexFlatIP(query_dim)

print("FAISS working with CPU")

# Process image index
image_index.add(image_vectors_array)
faiss.write_index(image_index, os.path.join(output_folder, "image_faiss_index.bin"))

# Process query index
query_index.add(query_vectors_array)
faiss.write_index(query_index, os.path.join(output_folder, "query_faiss_index.bin"))

# Save the ID lists
with open(os.path.join(output_folder, "image_id_list.pkl"), "wb") as f:
    pickle.dump(image_ids, f)

with open(os.path.join(output_folder, "query_id_list.pkl"), "wb") as f:
    pickle.dump(query_ids, f)

print(f"FAISS indices created and saved to {output_folder}:")
print(f"- Image index: {image_index.ntotal} vectors")
print(f"- Query index: {query_index.ntotal} vectors")
