import faiss
import torch

index = faiss.read_index("faiss_index/image_faiss_index.bin")

query_embedding = torch.load("embeds/description_embeds_0.pt")


sample_query_embedding = query_embedding.cpu().numpy().astype('float32')[0, 0, :]
print(sample_query_embedding.shape)

distances, indices = index.search(sample_query_embedding.reshape(1, -1), 5)

print(distances)
print(indices)
