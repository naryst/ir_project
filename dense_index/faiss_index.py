import pickle
import faiss
import numpy as np

# Загружаем ранее созданные эмбеддинги
with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

print(f"Всего загружено объектов: {len(embeddings)}")

# Подготавливаем данные для FAISS
vectors = np.array([item["embedding"] for item in embeddings]).astype('float32')
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
vectors = vectors.astype("float32")

print(vectors.dtype)
print("Есть NaN:", np.isnan(vectors).any())
print("Есть inf:", np.isinf(vectors).any())

ids = [item["id"] for item in embeddings]
types = [item["type"] for item in embeddings]

# Сохраняем соответствие id и type
id_to_type = dict(zip(ids, types))
with open("id_to_type.pkl", "wb") as f:
    pickle.dump(id_to_type, f)

# Строим FAISS индекс
vector_dim = vectors.shape[1]

print("Форма массива эмбеддингов:", vectors.shape)

# Используем IndexFlatIP (для cosine similarity на нормализованных эмбеддингах)
index = faiss.IndexFlatIP(vector_dim)

if faiss.get_num_gpus() > 0:
    print("FAISS работает с GPU")
    index = faiss.index_cpu_to_all_gpus(index)  # переносим на GPU
    index.add(vectors)

    faiss.write_index(faiss.index_gpu_to_cpu(index), "faiss_index.bin")  # сохраняем как CPU-индекс
else:
    print("FAISS работает с CPU")
    index.add(vectors)
    faiss.write_index(index, "faiss_index.bin")

with open("faiss_id_list.pkl", "wb") as f:
    pickle.dump(ids, f)

print(f"FAISS индекс создан и сохранен. Всего векторов: {index.ntotal}")
