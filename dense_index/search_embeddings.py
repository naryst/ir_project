import torch
import numpy as np
import os
from ball_tree import BallTree
from typing import List, Tuple
import glob

class EmbeddingSearcher:
    def __init__(self, embeddings_dir: str = "embeddings_qwen", is_image_searcher: bool = True):
        self.embeddings_dir = embeddings_dir
        self.ball_tree = None
        self.embeddings = None
        self.is_image_searcher = is_image_searcher
        self.image_paths = []  # You'll need to populate this with actual image paths
        
    def load_embeddings(self) -> None:
        """Load embeddings from the directory."""
        if self.is_image_searcher:
            # Load image embeddings
            files = sorted(glob.glob(os.path.join(self.embeddings_dir, "images_embeddings_tensor_*.pt")))
            embeddings_list = []
            
            for file in files:
                embeddings = torch.load(file)
                embeddings_list.append(embeddings)
                
            # Concatenate along the first dimension instead of stacking
            self.embeddings = torch.cat(embeddings_list, dim=0).numpy()
        else:
            # Load query embeddings
            files = sorted(glob.glob(os.path.join(self.embeddings_dir, "queries_embeddings_tensor_*.pt")))
            embeddings_list = []
            
            for file in files:
                embeddings = torch.load(file)
                embeddings_list.append(embeddings)
                
            # Concatenate along the first dimension instead of stacking
            self.embeddings = torch.cat(embeddings_list, dim=0).numpy()

        # normalize embeddings
        self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        print(f"Loaded {self.embeddings.shape[0]} {'image' if self.is_image_searcher else 'query'} embeddings")
        
    def build_index(self, leaf_size: int = 40) -> None:
        """Build the ball tree index."""
        if self.embeddings is None:
            self.load_embeddings()
            
        self.ball_tree = BallTree(leaf_size=leaf_size)
        self.ball_tree.build(self.embeddings)
        
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
    def load_index(cls, path: str, embeddings_dir: str = "embeddings_qwen", is_image_searcher: bool = True) -> 'EmbeddingSearcher':
        """Load a saved ball tree index."""
        searcher = cls(embeddings_dir, is_image_searcher)
        searcher.load_embeddings()
        searcher.ball_tree = BallTree.load(path)
        return searcher
    

def main():
    # Create separate searchers for images and queries
    image_searcher = EmbeddingSearcher(is_image_searcher=True)
    query_searcher = EmbeddingSearcher(is_image_searcher=False)
    
    # Load embeddings and build indices
    image_searcher.load_embeddings()
    query_searcher.load_embeddings()
    
    image_searcher.build_index()
    query_searcher.build_index()
    
    # Example search for images
    query_idx = 0  # Using the first query embedding as an example
    query_embedding = query_searcher.embeddings[query_idx]
    image_indices, image_distances = image_searcher.search(query_embedding, k=5)
    
    print(f"Found {len(image_indices)} nearest image neighbors:")
    for idx, dist in zip(image_indices, image_distances):
        print(f"Image Index: {idx}, Distance: {dist}")
        
    # Example search for similar queries
    query_indices, query_distances = query_searcher.search(query_embedding, k=5)
    
    print(f"\nFound {len(query_indices)} nearest query neighbors:")
    for idx, dist in zip(query_indices, query_distances):
        print(f"Query Index: {idx}, Distance: {dist}")
        
    # Save the indices for later use
    image_searcher.save_index("ball_tree_index/image_ball_tree.pkl")
    query_searcher.save_index("ball_tree_index/query_ball_tree.pkl")

if __name__ == "__main__":
    main() 