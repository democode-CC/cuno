"""
Data loading utilities for various graph datasets
Supports: Cora, CiteSeer, PubMed (Planetoid node-classification benchmarks).
"""

import os
import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
import pickle




def load_dataset(dataset_name, data_dir='./data'):
    """
    Load a graph dataset for node classification.

    Supported: Cora, CiteSeer, PubMed (Planetoid).

    Returns:
        data:  PyTorch Geometric Data object with train_mask / val_mask / test_mask
        is_kg: kept in the return signature for downstream compatibility;
               always False in this release (no knowledge-graph datasets are
               shipped).
    """
    if dataset_name in ('Cora', 'CiteSeer', 'PubMed'):
        return load_planetoid(dataset_name, data_dir)
    raise ValueError(
        f"Unknown dataset: {dataset_name}. "
        f"This release supports Cora, CiteSeer, and PubMed only."
    )



def load_homogeneous_graph(dataset_name, data_dir):
    """
    Load homogeneous graph datasets (Cora, CiteSeer, PubMed)
    
    Returns:
        data: PyG Data object
        is_kg: False (not a knowledge graph)
    """
    dataset = Planetoid(root=os.path.join(data_dir, dataset_name), name=dataset_name)
    data = dataset[0]
    
    # Ensure undirected
    data.edge_index = to_undirected(data.edge_index)
    
    # Add useful statistics
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of features: {data.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
    print(f"Has isolated nodes: {data.has_isolated_nodes()}")
    print(f"Has self-loops: {data.has_self_loops()}")
    print(f"Is undirected: {data.is_undirected()}")
    print(f"{'='*60}\n")
    
    return data, False




def split_data(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, is_kg=False):
    """
    Split data into train/val/test sets
    
    Args:
        data: Graph data
        train_ratio: Ratio of training nodes
        val_ratio: Ratio of validation nodes
        test_ratio: Ratio of test nodes
        is_kg: Whether it's a knowledge graph
        
    Returns:
        train_mask, val_mask, test_mask
    """
    if is_kg:
        # For KG, split edges
        num_edges = data['train']['edge_index'].size(1)
        indices = torch.randperm(num_edges)
        
        train_size = int(train_ratio * num_edges)
        val_size = int(val_ratio * num_edges)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        return train_idx, val_idx, test_idx
    else:
        # For homogeneous graphs, use existing masks if available
        if hasattr(data, 'train_mask') and data.train_mask is not None:
            return data.train_mask, data.val_mask, data.test_mask
        
        # Otherwise create new masks
        num_nodes = data.num_nodes
        indices = torch.randperm(num_nodes)
        
        train_size = int(train_ratio * num_nodes)
        val_size = int(val_ratio * num_nodes)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True
        
        return train_mask, val_mask, test_mask


def get_forget_retain_split(data, forget_ratio, is_kg=False, seed=42):
    """
    Split training data into forget and retain sets
    
    Args:
        data: Graph data
        forget_ratio: Ratio of training data to forget
        is_kg: Whether it's a knowledge graph
        seed: Random seed
        
    Returns:
        forget_mask, retain_mask (for nodes or edges)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if is_kg:
        # For KG, split training edges
        num_edges = data['train']['edge_index'].size(1)
        indices = torch.randperm(num_edges)
        
        forget_size = int(forget_ratio * num_edges)
        forget_idx = indices[:forget_size]
        retain_idx = indices[forget_size:]
        
        forget_mask = torch.zeros(num_edges, dtype=torch.bool)
        retain_mask = torch.zeros(num_edges, dtype=torch.bool)
        
        forget_mask[forget_idx] = True
        retain_mask[retain_idx] = True
        
        return forget_mask, retain_mask
    else:
        # For homogeneous graphs, split training nodes
        train_mask = data.train_mask
        train_indices = train_mask.nonzero(as_tuple=True)[0]
        
        num_train = len(train_indices)
        perm = torch.randperm(num_train)
        
        forget_size = int(forget_ratio * num_train)
        forget_train_idx = train_indices[perm[:forget_size]]
        retain_train_idx = train_indices[perm[forget_size:]]
        
        forget_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        retain_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        
        forget_mask[forget_train_idx] = True
        retain_mask[retain_train_idx] = True
        
        return forget_mask, retain_mask



