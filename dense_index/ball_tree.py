import numpy as np
import torch
from typing import List, Tuple, Optional
import pickle
import os

class BallTree:
    def __init__(self, leaf_size: int = 40):
        self.leaf_size = leaf_size
        self.root = None
        self.dim = None
        self.data = None
        
    class Node:
        def __init__(self, center: np.ndarray, radius: float, left: Optional['BallTree.Node'] = None, 
                    right: Optional['BallTree.Node'] = None, indices: Optional[np.ndarray] = None):
            self.center = center
            self.radius = radius
            self.left = left
            self.right = right
            self.indices = indices  # Only for leaf nodes
            
    def build(self, data: np.ndarray) -> None:
        self.dim = data.shape[1]
        self.data = data  # Store the data
        self.root = self._build_tree(data, np.arange(len(data)))
        
    def _build_tree(self, data: np.ndarray, indices: np.ndarray) -> Node:
        if len(indices) <= self.leaf_size:
            # Create leaf node
            center = np.mean(data[indices], axis=0)
            radius = np.max(np.linalg.norm(data[indices] - center, axis=1))
            return self.Node(center, radius, indices=indices)
        
        # Find the dimension with maximum variance
        variances = np.var(data[indices], axis=0)
        split_dim = np.argmax(variances)
        
        # Split data along the dimension with maximum variance
        split_val = np.median(data[indices, split_dim])
        left_mask = data[indices, split_dim] <= split_val
        right_mask = ~left_mask
        
        left_indices = indices[left_mask]
        right_indices = indices[right_mask]
        
        # Recursively build left and right subtrees
        left_node = self._build_tree(data, left_indices)
        right_node = self._build_tree(data, right_indices)
        
        # Compute center and radius for the current node
        center = np.mean(data[indices], axis=0)
        radius = np.max(np.linalg.norm(data[indices] - center, axis=1))
        
        return self.Node(center, radius, left_node, right_node)
    
    def query(self, query: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        if self.root is None:
            raise ValueError("Tree has not been built yet")
            
        # Initialize priority queue with root node
        queue = [(0, self.root)]
        best_dist = float('inf')
        best_indices = []
        best_distances = []
        
        while queue:
            dist, node = queue.pop(0)
            
            if dist > best_dist and len(best_indices) >= k:
                continue
                
            if node.indices is not None:  # Leaf node
                # Compute distances to all points in the leaf
                leaf_data = self.data[node.indices]
                leaf_distances = np.linalg.norm(leaf_data - query, axis=1)
                
                for i, idx in enumerate(node.indices):
                    if len(best_indices) < k or leaf_distances[i] < best_dist:
                        best_indices.append(idx)
                        best_distances.append(leaf_distances[i])
                        if len(best_indices) > k:
                            # Remove the worst point
                            worst_idx = np.argmax(best_distances)
                            best_indices.pop(worst_idx)
                            best_distances.pop(worst_idx)
                            best_dist = max(best_distances)
            else:
                # Compute distances to child nodes
                left_dist = np.linalg.norm(query - node.left.center)
                right_dist = np.linalg.norm(query - node.right.center)
                
                # Add child nodes to queue
                if left_dist <= right_dist:
                    queue.append((left_dist, node.left))
                    queue.append((right_dist, node.right))
                else:
                    queue.append((right_dist, node.right))
                    queue.append((left_dist, node.left))
                
        return np.array(best_indices), np.array(best_distances)
    
    def batch_query(self, queries: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        all_indices = []
        all_distances = []
        
        for query in queries:
            indices, distances = self.query(query, k)
            all_indices.append(indices)
            all_distances.append(distances)
            
        return np.array(all_indices), np.array(all_distances)
    
    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, path: str) -> 'BallTree':
        with open(path, 'rb') as f:
            return pickle.load(f)
