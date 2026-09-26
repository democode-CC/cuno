import os
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import networkx as nx
from copy import deepcopy

from learning import create_model, train_homogeneous, train_knowledge_graph
from data import get_forget_retain_split

from curriculum import CurriculumDesigner, ComplexityCalculator
from utils import set_unlearn_seed

def retrain_baseline(args, data, is_kg, retain_mask):
    """
    Baseline 1: Retrain from scratch on retain set only
    
    Args:
        args: Arguments
        data: Graph data
        is_kg: Whether it's a knowledge graph
        retain_mask: Mask for retain set
        
    Returns:
        model: Retrained model
    """
    # Set seed for reproducibility
    set_unlearn_seed(getattr(args, 'seed', 42))
    
    print("\n" + "="*60)
    print("Baseline: Retrain from Scratch")
    print("="*60)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Create new model
    model = create_model(args, data, is_kg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_lr, weight_decay=args.weight_decay)
    
    # Temporarily modify training mask to only include retain set
    if not is_kg:
        original_train_mask = data.train_mask.clone()
        data.train_mask = retain_mask
        data = data.to(device)
    
    # Train on retain set only
    pbar = tqdm(range(args.learning_epochs), desc="Retraining")
    for epoch in pbar:
        if is_kg:
            # Filter to retain edges only
            retain_data = {
                'num_entities': data['num_entities'],
                'num_relations': data['num_relations'],
                'train': {
                    'edge_index': data['train']['edge_index'][:, retain_mask],
                    'edge_type': data['train']['edge_type'][retain_mask],
                    'num_edges': retain_mask.sum().item()
                },
                'entity_ids': data['entity_ids']
            }
            loss = train_knowledge_graph(model, retain_data, optimizer, device)
        else:
            loss = train_homogeneous(model, data, optimizer, device)
        
        pbar.set_postfix({'Loss': f'{loss:.4f}'})
    
    # Restore original mask
    if not is_kg:
        data.train_mask = original_train_mask
    
    print("Retrain completed!")
    return model