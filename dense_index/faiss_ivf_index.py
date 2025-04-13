import pickle
import faiss
import numpy as np

# === parameters ===
EMBEDDINGS_PATH = "embeddings.pkl"
INDEX_PATH = "faiss_ivf_index.bin"
ID_LIST_PATH = "faiss__ivf_id_list.pkl"
ID_TYPE_PATH = "id__ivf_to_type.pkl"

# === Step 1: Load embeddings ===
with open(EMBEDDINGS_PATH, "rb") as f:
    embeddings = pickle.load(f)

print(f"[INFO] Uploaded embeddings: {len(embeddings)}")

vectors = np.array([item["embedding"] for item in embeddings], dtype="float32")
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

ids = [item["id"] for item in embeddings]
types = [item["type"] for item in embeddings]
id_to_type = dict(zip(ids, types))

# === Step 2: Index Parameters ===
dim = vectors.shape[1]
nlist = 100  
quantizer = faiss.IndexFlatIP(dim)  # internal accurate search
index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

# === Step 3: Training (required for IVF) ===
print("[INFO] Index training...")
index.train(vectors)

# === Step 4: Adding embeddings ===
print("[INFO] adding embeddings...")
index.add(vectors)
print(f"[INFO] index is done : {index.ntotal} vectors")

# === Step 5: Saving the Index ===
faiss.write_index(index, INDEX_PATH)
print(f"[INFO] Saved index: {INDEX_PATH}")

# === Step 6: Saving auxiliary data ===
with open(ID_LIST_PATH, "wb") as f:
    pickle.dump(ids, f)

with open(ID_TYPE_PATH, "wb") as f:
    pickle.dump(id_to_type, f)

print("[INFO] Everything is saved: IDs and types")
