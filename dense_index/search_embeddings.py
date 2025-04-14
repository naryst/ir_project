import torch
import numpy as np
import os
from ball_tree import BallTree
from typing import List, Tuple
import glob

class EmbeddingSearcher:
    def __init__(self, embeddings_dir: str = "embeddings_qwen"):
        self.embeddings_dir = embeddings_dir
        self.ball_tree = None
        self.image_embeddings = None
        self.query_embeddings = None
        self.image_paths = []  # You'll need to populate this with actual image paths
        
    def load_embeddings(self) -> None:
        """Load all embeddings from the directory."""
        # Load image embeddings
        image_files = sorted(glob.glob(os.path.join(self.embeddings_dir, "images_embeddings_tensor_*.pt")))
        image_embeddings_list = []
        
        for file in image_files:
            embeddings = torch.load(file)
            image_embeddings_list.append(embeddings)
            
        # Concatenate along the first dimension instead of stacking
        self.image_embeddings = torch.cat(image_embeddings_list, dim=0).numpy()
        
        # Load query embeddings
        query_files = sorted(glob.glob(os.path.join(self.embeddings_dir, "queries_embeddings_tensor_*.pt")))
        query_embeddings_list = []
        
        for file in query_files:
            embeddings = torch.load(file)
            query_embeddings_list.append(embeddings)
            
        # Concatenate along the first dimension instead of stacking
        self.query_embeddings = torch.cat(query_embeddings_list, dim=0).numpy()

        print(self.image_embeddings.shape)
        print(self.query_embeddings.shape)
        
    def build_index(self, leaf_size: int = 40) -> None:
        """Build the ball tree index."""
        if self.image_embeddings is None:
            self.load_embeddings()
            
        self.ball_tree = BallTree(leaf_size=leaf_size)
        self.ball_tree.build(self.image_embeddings)
        
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors of a query embedding."""
        if self.ball_tree is None:
            self.build_index()
            
        # Ensure query_embedding is 2D array
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        indices, distances = self.ball_tree.query(query_embedding[0], k=k)
        return indices, distances
        
    def batch_search(self, query_embeddings: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors of multiple query embeddings."""
        if self.ball_tree is None:
            self.build_index()
            
        # Ensure query_embeddings is 2D array
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
            
        indices, distances = self.ball_tree.batch_query(query_embeddings, k=k)
        return indices, distances
        
    def save_index(self, path: str) -> None:
        """Save the ball tree index to disk."""
        if self.ball_tree is None:
            raise ValueError("Index has not been built yet")
        self.ball_tree.save(path)
        
    @classmethod
    def load_index(cls, path: str, embeddings_dir: str = "embeddings_qwen") -> 'EmbeddingSearcher':
        """Load a saved ball tree index."""
        searcher = cls(embeddings_dir)
        searcher.load_embeddings()
        searcher.ball_tree = BallTree.load(path)
        return searcher
    

def main():
    # Example usage
    searcher = EmbeddingSearcher()
    searcher.load_embeddings()
    searcher.build_index()
    
    # Example search
    query_idx = 0  # Using the first query embedding as an example
    query_embedding = searcher.query_embeddings[query_idx]
    indices, distances = searcher.search(query_embedding, k=5)
    
    print(f"Found {len(indices)} nearest neighbors:")
    for idx, dist in zip(indices, distances):
        print(f"Index: {idx}, Distance: {dist}")
        
    # Save the index for later use
    searcher.save_index("ball_tree_index.pkl")

if __name__ == "__main__":
    main() 