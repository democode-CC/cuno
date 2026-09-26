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


def gradient_ascent_baseline(args, model, data, is_kg, forget_mask, retain_mask):
    """
    Baseline 2: Gradient Ascent
    
    Args:
        args: Arguments
        model: Trained model
        data: Graph data
        is_kg: Whether it's a knowledge graph
        forget_mask: Mask for forget set
        retain_mask: Mask for retain set
        
    Returns:
        model: Unlearned model
    """
    # Set seed for reproducibility
    set_unlearn_seed(getattr(args, 'seed', 42))
    
    print("\n" + "="*60)
    print("Baseline: Gradient Ascent")
    print("="*60)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if not is_kg:
        data = data.to(device)
        forget_mask = forget_mask.to(device)
        retain_mask = retain_mask.to(device)
    

    # ------------------------------------------------------------
    # Stabilization for very small forget sets (identical across
    # all iterative methods in this repo: gradient_ascent,
    # curriculum_gradient_ascent, npo_ga, full_method_kl, full_method).
    # When |S_d| < 10 labelled nodes, iterative unlearning under the
    # shared learning rate can diverge on tiny subsets. We therefore
    # scale the unlearning learning rate by |S_d|/10 and raise the
    # retain-loss weight to 0.5. This rule fires only at the smallest
    # deletion rates (e.g. gamma <= 5% on Cora, gamma <= 10% on PubMed)
    # and does not fire at gamma >= 20% on any of the three datasets
    # reported in the paper, so all mass-deletion results are obtained
    # without stabilization.
    # ------------------------------------------------------------
    # Dynamically lower the learning rate for smaller forget sets
    forget_count = forget_mask.sum().item()
    MIN_STABLE_FORGET_SIZE = 10  # Empirical threshold: at least 10 nodes are needed for stable training
    
    if forget_count < MIN_STABLE_FORGET_SIZE:
        # Reduce learning rate to avoid catastrophic model collapse
        lr_scale = forget_count / MIN_STABLE_FORGET_SIZE
        adjusted_lr = args.unlearn_lr * lr_scale
        print(f"⚠️  Small forget set ({forget_count} nodes), reducing LR: {args.unlearn_lr} -> {adjusted_lr:.6f}")
    else:
        adjusted_lr = args.unlearn_lr
    
    optimizer = optim.Adam(model.parameters(), lr=adjusted_lr)
    
    # Add a retain-set regularization weight to preserve retained performance
    retain_weight = getattr(args, 'retain_weight', 0.1)  # Default retain regularization weight
    if forget_count < MIN_STABLE_FORGET_SIZE:
        retain_weight = 0.5  # Increase regularization for small forget sets
        print(f"  Using stronger retain regularization: {retain_weight}")
    
    pbar = tqdm(range(args.unlearn_epochs), desc="Gradient Ascent")
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        
        if is_kg:
            # Unlearn forget edges
            edge_index = data['train']['edge_index'].to(device)
            edge_type = data['train']['edge_type'].to(device)
            entity_ids = data['entity_ids'].to(device)
            
            forget_indices = forget_mask.nonzero(as_tuple=True)[0]
            if len(forget_indices) > 0:
                batch_size = min(len(forget_indices), 512)
                sample_idx = forget_indices[torch.randperm(len(forget_indices))[:batch_size]]
                
                head_idx = edge_index[0, sample_idx]
                tail_idx = edge_index[1, sample_idx]
                rel_idx = edge_type[sample_idx]
                
                # Gradient ascent: maximize loss (minimize negative loss)
                scores = model.predict_link(entity_ids, edge_index, edge_type, head_idx, tail_idx, rel_idx)
                loss = -scores.mean()  # Negative to maximize
                
                loss.backward()
                optimizer.step()
                
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        else:
            # Unlearn forget nodes
            out = model(data.x, data.edge_index)
            
            # Gradient ascent: maximize loss on forget set
            forget_loss = -F.cross_entropy(out[forget_mask], data.y[forget_mask])
            
            # Add retain-loss regularization to preserve performance on the retain set
            retain_loss = F.cross_entropy(out[retain_mask], data.y[retain_mask])
            
            # Total loss = forget_loss (negated, drives forgetting) + retain_weight * retain_loss (preserves utility)
            loss = forget_loss + retain_weight * retain_loss
            
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({
                'Forget': f'{forget_loss.item():.4f}',
                'Retain': f'{retain_loss.item():.4f}'
            })
    
    print("Gradient Ascent completed!")
    return model
