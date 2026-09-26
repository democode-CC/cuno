"""
Data loading utilities for various graph datasets
Supports: Cora, CiteSeer, PubMed (homogeneous) and FB15k237, WN18RR (knowledge graphs)
"""

import os
import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
import pickle


ADDITIONAL_HOMOGENEOUS_DATASETS = ['AmazonComputers', 'CoauthorCS', 'Actor',
                                   'RomanEmpire', 'AmazonRatings', 'AmazonPhoto']


def load_dataset(dataset_name, data_dir='./data'):
    """
    Load dataset based on name

    Args:
        dataset_name: Name of the dataset
        data_dir: Directory to store/load datasets

    Returns:
        data: PyG Data object or dictionary containing graph data
        is_kg: Boolean indicating if it's a knowledge graph
    """
    os.makedirs(data_dir, exist_ok=True)

    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        return load_homogeneous_graph(dataset_name, data_dir)
    elif dataset_name in ADDITIONAL_HOMOGENEOUS_DATASETS:
        return load_additional_graph(dataset_name, data_dir)
    elif dataset_name in ['FB15k237', 'WN18RR']:
        return load_knowledge_graph(dataset_name, data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


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


def load_additional_graph(dataset_name, data_dir, split_seed=0):
    """
    Load additional homogeneous graph datasets without predefined splits.
    Supported: AmazonComputers, CoauthorCS, Actor.
    A deterministic 60/20/20 node split is created and attached to the data object
    so the rest of the codebase behaves identically to Planetoid datasets.
    """
    if dataset_name == 'AmazonComputers':
        from torch_geometric.datasets import Amazon
        dataset = Amazon(root=os.path.join(data_dir, dataset_name), name='Computers')
    elif dataset_name == 'CoauthorCS':
        from torch_geometric.datasets import Coauthor
        dataset = Coauthor(root=os.path.join(data_dir, dataset_name), name='CS')
    elif dataset_name == 'Actor':
        from torch_geometric.datasets import Actor
        dataset = Actor(root=os.path.join(data_dir, dataset_name))
    elif dataset_name == 'RomanEmpire':
        from torch_geometric.datasets import HeterophilousGraphDataset
        dataset = HeterophilousGraphDataset(root=os.path.join(data_dir, dataset_name),
                                            name='Roman-empire')
    elif dataset_name == 'AmazonRatings':
        from torch_geometric.datasets import HeterophilousGraphDataset
        dataset = HeterophilousGraphDataset(root=os.path.join(data_dir, dataset_name),
                                            name='Amazon-ratings')
    elif dataset_name == 'AmazonPhoto':
        from torch_geometric.datasets import Amazon
        dataset = Amazon(root=os.path.join(data_dir, dataset_name), name='Photo')
    else:
        raise ValueError(f"Unknown additional dataset: {dataset_name}")

    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index)
    num_nodes = data.num_nodes

    if dataset_name in ('RomanEmpire', 'AmazonRatings'):
        # Official splits from Platonov et al. (ICLR 2023): 10 fixed random
        # 50/25/25 splits shipped with the dataset. We use split 0 throughout.
        data.train_mask = data.train_mask[:, 0]
        data.val_mask   = data.val_mask[:, 0]
        data.test_mask  = data.test_mask[:, 0]
    elif dataset_name == 'AmazonPhoto':
        # Protocol of Shchur et al. (2018): 20 labelled nodes per class for
        # training, 30 per class for validation, remainder for test.
        g = torch.Generator().manual_seed(split_seed)
        y = data.y
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask  = torch.ones(num_nodes, dtype=torch.bool)
        for c in y.unique():
            idx = (y == c).nonzero(as_tuple=True)[0]
            idx = idx[torch.randperm(idx.numel(), generator=g)]
            train_mask[idx[:20]] = True
            val_mask[idx[20:50]] = True
        test_mask[train_mask | val_mask] = False
        data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
    else:
        # AmazonComputers / CoauthorCS / Actor: deterministic 60/20/20
        torch.manual_seed(split_seed)
        perm = torch.randperm(num_nodes)
        train_end = int(0.6 * num_nodes)
        val_end = train_end + int(0.2 * num_nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[perm[:train_end]]       = True
        val_mask[perm[train_end:val_end]]  = True
        test_mask[perm[val_end:]]          = True
        data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    print(f"Number of nodes:    {data.num_nodes}")
    print(f"Number of edges:    {data.num_edges}")
    print(f"Number of features: {data.num_features}")
    print(f"Number of classes:  {dataset.num_classes}")
    print(f"Average node degree:{data.num_edges / data.num_nodes:.2f}")
    print(f"Train/Val/Test:     {int(data.train_mask.sum())}/{int(data.val_mask.sum())}/{int(data.test_mask.sum())}")
    print(f"Has isolated nodes: {data.has_isolated_nodes()}")
    print(f"Has self-loops:     {data.has_self_loops()}")
    print(f"Is undirected:      {data.is_undirected()}")
    print(f"{'='*60}\n")

    return data, False


def load_knowledge_graph(dataset_name, data_dir):
    """
    Load knowledge graph datasets (FB15k237, WN18RR)
    
    Returns:
        data: Dictionary containing KG data
        is_kg: True (is a knowledge graph)
    """
    kg_dir = os.path.join(data_dir, dataset_name)
    os.makedirs(kg_dir, exist_ok=True)
    
    # Check if already processed
    processed_file = os.path.join(kg_dir, 'processed_data.pkl')
    if os.path.exists(processed_file):
        print(f"Loading processed {dataset_name} from {processed_file}")
        with open(processed_file, 'rb') as f:
            data = pickle.load(f)
        print_kg_statistics(data, dataset_name)
        return data, True
    
    # Download and process
    print(f"Processing {dataset_name}...")
    if dataset_name == 'FB15k237':
        data = download_and_process_fb15k237(kg_dir)
    elif dataset_name == 'WN18RR':
        data = download_and_process_wn18rr(kg_dir)
    
    # Save processed data
    with open(processed_file, 'wb') as f:
        pickle.dump(data, f)
    
    print_kg_statistics(data, dataset_name)
    return data, True


def download_and_process_fb15k237(kg_dir):
    """Download and process FB15k237 dataset"""
    # Try to use PyTorch Geometric's dataset
    try:
        from torch_geometric.datasets import FB15k_237
        dataset = FB15k_237(root=kg_dir)
        
        # Process train split
        train_data = dataset[0]
        
        # Extract entities and relations
        entities = set()
        relations = set()
        
        for split_data in dataset:
            if hasattr(split_data, 'edge_index'):
                entities.update(split_data.edge_index[0].tolist())
                entities.update(split_data.edge_index[1].tolist())
            if hasattr(split_data, 'edge_type'):
                relations.update(split_data.edge_type.tolist())
        
        num_entities = len(entities)
        num_relations = len(set(train_data.edge_type.tolist())) if hasattr(train_data, 'edge_type') else max(relations) + 1
        
        data = {
            'num_entities': num_entities,
            'num_relations': num_relations,
            'train': {
                'edge_index': train_data.edge_index,
                'edge_type': train_data.edge_type if hasattr(train_data, 'edge_type') else torch.zeros(train_data.edge_index.size(1), dtype=torch.long),
                'num_edges': train_data.edge_index.size(1)
            },
            'entity_ids': torch.arange(num_entities)
        }
        
        return data
        
    except Exception as e:
        print(f"Warning: Could not load FB15k237 using PyG: {e}")
        print("Creating synthetic FB15k237-like dataset for testing...")
        return create_synthetic_kg('FB15k237', num_entities=14541, num_relations=237)


def download_and_process_wn18rr(kg_dir):
    """Download and process WN18RR dataset"""
    try:
        from torch_geometric.datasets import WordNet18RR
        dataset = WordNet18RR(root=kg_dir)
        
        train_data = dataset[0]
        
        entities = set()
        relations = set()
        
        for split_data in dataset:
            if hasattr(split_data, 'edge_index'):
                entities.update(split_data.edge_index[0].tolist())
                entities.update(split_data.edge_index[1].tolist())
            if hasattr(split_data, 'edge_type'):
                relations.update(split_data.edge_type.tolist())
        
        num_entities = len(entities)
        num_relations = len(set(train_data.edge_type.tolist())) if hasattr(train_data, 'edge_type') else max(relations) + 1
        
        data = {
            'num_entities': num_entities,
            'num_relations': num_relations,
            'train': {
                'edge_index': train_data.edge_index,
                'edge_type': train_data.edge_type if hasattr(train_data, 'edge_type') else torch.zeros(train_data.edge_index.size(1), dtype=torch.long),
                'num_edges': train_data.edge_index.size(1)
            },
            'entity_ids': torch.arange(num_entities)
        }
        
        return data
        
    except Exception as e:
        print(f"Warning: Could not load WN18RR using PyG: {e}")
        print("Creating synthetic WN18RR-like dataset for testing...")
        return create_synthetic_kg('WN18RR', num_entities=40943, num_relations=11)


def create_synthetic_kg(name, num_entities, num_relations):
    """Create a synthetic knowledge graph for testing"""
    print(f"Creating synthetic {name} with {num_entities} entities and {num_relations} relations")
    
    # Generate random triples
    num_edges = min(num_entities * 20, 100000)  # Reasonable number of edges
    
    heads = torch.randint(0, num_entities, (num_edges,))
    tails = torch.randint(0, num_entities, (num_edges,))
    relations = torch.randint(0, num_relations, (num_edges,))
    
    edge_index = torch.stack([heads, tails], dim=0)
    
    data = {
        'num_entities': num_entities,
        'num_relations': num_relations,
        'train': {
            'edge_index': edge_index,
            'edge_type': relations,
            'num_edges': num_edges
        },
        'entity_ids': torch.arange(num_entities)
    }
    
    return data


def print_kg_statistics(data, dataset_name):
    """Print statistics for knowledge graph"""
    print(f"\n{'='*60}")
    print(f"Knowledge Graph: {dataset_name}")
    print(f"{'='*60}")
    print(f"Number of entities: {data['num_entities']}")
    print(f"Number of relations: {data['num_relations']}")
    print(f"Number of training triples: {data['train']['num_edges']}")
    print(f"Average edges per entity: {data['train']['num_edges'] / data['num_entities']:.2f}")
    print(f"{'='*60}\n")


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



