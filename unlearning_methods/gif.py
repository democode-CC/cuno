"""
GIF: Graph-oriented Influence Function for Graph Unlearning
Based on: "GIF: A General Graph Unlearning Strategy via Influence Function"
Paper: https://arxiv.org/abs/2304.02835
Code: https://github.com/wujcan/GIF-torch
"""

import os
import random
import torch
import torch.nn.functional as F
from torch.autograd import grad
from tqdm import tqdm
import numpy as np
from copy import deepcopy

from utils import set_unlearn_seed


def gif_unlearn(args, model, data, is_kg, forget_mask, retain_mask):
    """
    GIF: Graph-oriented Influence Function for Graph Unlearning
    
    Uses influence functions to estimate parameter changes in response to data deletions.
    For graphs, it accounts for influenced neighbors due to structural dependencies.
    
    Args:
        args: Arguments containing GIF hyperparameters
        model: Trained model
        data: Graph data
        is_kg: Whether it's a knowledge graph
        forget_mask: Mask for forget set (nodes or edges)
        retain_mask: Mask for retain set (nodes or edges)
        
    Returns:
        model: Unlearned model
    """
    # Set seed for reproducibility
    set_unlearn_seed(getattr(args, 'seed', 42))
    
    print("\n" + "="*60)
    print("GIF: Graph-oriented Influence Function")
    print(f"Iterations: {args.gif_iteration}, Scale: {args.gif_scale}, Damp: {args.gif_damp}")
    print("="*60)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if not is_kg:
        data = data.to(device)
        forget_mask = forget_mask.to(device)
        retain_mask = retain_mask.to(device)
    
    # Find influenced neighbors (k-hop neighbors of forget set)
    if is_kg:
        # For KG, find entities connected to forget edges
        influenced_mask = find_influenced_entities_kg(data, forget_mask, k_hops=2)
    else:
        # For homogeneous graphs, find k-hop neighbors of forget nodes
        influenced_mask = find_influenced_nodes(data, forget_mask, k_hops=2)
    influenced_mask = influenced_mask.to(device)
    
    # Compute gradients needed for GIF
    print("Computing gradients for influence function...")
    grad_all, grad1, grad2 = compute_gif_gradients(
        model, data, is_kg, forget_mask, retain_mask, influenced_mask, device
    )
    
    # Apply GIF approximation to estimate parameter changes
    print("Applying GIF approximation...")
    model = apply_gif_approximation(
        model, grad_all, grad1, grad2, 
        iteration=args.gif_iteration,
        scale=args.gif_scale,
        damp=args.gif_damp,
        device=device
    )
    
    print("GIF unlearning completed!")
    return model


def find_influenced_nodes(data, forget_mask, k_hops=2):
    """
    Find k-hop neighbors of forget nodes (influenced nodes)
    
    Args:
        data: Graph data
        forget_mask: Boolean mask for forget nodes
        k_hops: Number of hops to consider
        
    Returns:
        influenced_mask: Boolean mask for influenced nodes
    """
    edge_index = data.edge_index.cpu().numpy()
    forget_nodes = forget_mask.cpu().numpy().nonzero()[0]
    
    influenced_nodes = set(forget_nodes)
    
    # Find k-hop neighbors
    for _ in range(k_hops):
        # Find edges where source is in influenced_nodes
        source_mask = np.isin(edge_index[0], list(influenced_nodes))
        neighbor_nodes = edge_index[1, source_mask]
        influenced_nodes.update(neighbor_nodes)
    
    # Create mask
    influenced_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    influenced_mask[list(influenced_nodes)] = True
    
    return influenced_mask


def find_influenced_entities_kg(data, forget_mask, k_hops=2):
    """
    Find entities influenced by forget edges in knowledge graph
    
    Args:
        data: Knowledge graph data
        forget_mask: Boolean mask for forget edges
        k_hops: Number of hops to consider
        
    Returns:
        influenced_mask: Boolean mask for influenced edges (edges connected to influenced entities)
    """
    edge_index = data['train']['edge_index'].cpu().numpy()
    forget_edges = forget_mask.cpu().numpy().nonzero()[0]
    
    # Get entities involved in forget edges
    forget_entities = set()
    for edge_idx in forget_edges:
        forget_entities.add(int(edge_index[0, edge_idx]))
        forget_entities.add(int(edge_index[1, edge_idx]))
    
    influenced_entities = set(forget_entities)
    
    # Find k-hop neighbors
    for _ in range(k_hops):
        # Find edges where source is in influenced_entities
        source_mask = np.isin(edge_index[0], list(influenced_entities))
        neighbor_entities = edge_index[1, source_mask]
        influenced_entities.update([int(e) for e in neighbor_entities])
    
    # Create mask (for edges, we mark edges connected to influenced entities)
    influenced_mask = torch.zeros(data['train']['edge_index'].size(1), dtype=torch.bool)
    influenced_entities_set = influenced_entities
    for i in range(edge_index.shape[1]):
        if int(edge_index[0, i]) in influenced_entities_set or int(edge_index[1, i]) in influenced_entities_set:
            influenced_mask[i] = True
    
    return influenced_mask


def compute_gif_gradients(model, data, is_kg, forget_mask, retain_mask, influenced_mask, device):
    """
    Compute gradients needed for GIF:
    - grad_all: Gradient on all training data
    - grad1: Gradient on original graph with forget set
    - grad2: Gradient on modified graph (after deletion) with forget set
    
    Args:
        model: The model
        data: Graph data
        is_kg: Whether it's a knowledge graph
        forget_mask: Mask for forget set
        retain_mask: Mask for retain set
        influenced_mask: Mask for influenced nodes/edges
        device: Device
        
    Returns:
        grad_all, grad1, grad2: Three gradient tuples
    """
    model.train()
    model_params = [p for p in model.parameters() if p.requires_grad]
    
    if is_kg:
        return compute_gif_gradients_kg(
            model, data, forget_mask, retain_mask, influenced_mask, model_params, device
        )
    else:
        return compute_gif_gradients_homogeneous(
            model, data, forget_mask, retain_mask, influenced_mask, model_params, device
        )


def compute_gif_gradients_homogeneous(model, data, forget_mask, retain_mask, influenced_mask, model_params, device):
    """Compute GIF gradients for homogeneous graphs"""
    
    # Create modified graph (remove edges connected to forget nodes)
    edge_mask = ~forget_mask[data.edge_index[0]] & ~forget_mask[data.edge_index[1]]
    modified_edge_index = data.edge_index[:, edge_mask]
    
    # Forward pass on original graph
    out1 = model(data.x, data.edge_index)
    
    # Forward pass on modified graph
    out2 = model(data.x, modified_edge_index)
    
    # Loss on all training data (retain set)
    loss_all = F.cross_entropy(out1[retain_mask], data.y[retain_mask], reduction='sum')
    
    # Loss1: Original graph, forget set + influenced nodes
    mask1 = forget_mask | influenced_mask
    if mask1.sum() > 0:
        loss1 = F.cross_entropy(out1[mask1], data.y[mask1], reduction='sum')
    else:
        loss1 = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Loss2: Modified graph, forget set + influenced nodes (after deletion)
    mask2 = forget_mask | influenced_mask
    if mask2.sum() > 0:
        loss2 = F.cross_entropy(out2[mask2], data.y[mask2], reduction='sum')
    else:
        loss2 = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Compute gradients (allow_unused=True: some params may not affect loss1/loss2)
    grad_all = grad(loss_all, model_params, retain_graph=True, create_graph=True, allow_unused=True)
    grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True, allow_unused=True)
    grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True, allow_unused=True)
    # Replace None with zeros so downstream zip/subtract works
    grad_all = tuple(g if g is not None else torch.zeros_like(p) for g, p in zip(grad_all, model_params))
    grad1 = tuple(g if g is not None else torch.zeros_like(p) for g, p in zip(grad1, model_params))
    grad2 = tuple(g if g is not None else torch.zeros_like(p) for g, p in zip(grad2, model_params))
    
    return grad_all, grad1, grad2


def compute_gif_gradients_kg(model, data, forget_mask, retain_mask, influenced_mask, model_params, device):
    """Compute GIF gradients for knowledge graphs"""
    
    edge_index = data['train']['edge_index'].to(device)
    edge_type = data['train']['edge_type'].to(device)
    entity_ids = data['entity_ids'].to(device)
    
    # Create modified graph (remove forget edges)
    retain_edge_mask = retain_mask.to(device)
    modified_edge_index = edge_index[:, retain_edge_mask]
    modified_edge_type = edge_type[retain_edge_mask]
    
    # Sample edges for gradient computation
    num_sample = min(512, retain_mask.sum().item())
    retain_indices = retain_mask.nonzero(as_tuple=True)[0]
    if len(retain_indices) > 0:
        sample_retain = retain_indices[torch.randperm(len(retain_indices))[:num_sample]]
    else:
        sample_retain = torch.tensor([], dtype=torch.long, device=device)
    
    forget_indices = forget_mask.nonzero(as_tuple=True)[0]
    if len(forget_indices) > 0:
        num_forget_sample = min(512, len(forget_indices))
        sample_forget = forget_indices[torch.randperm(len(forget_indices))[:num_forget_sample]]
    else:
        sample_forget = torch.tensor([], dtype=torch.long, device=device)
    
    # Loss on all training data (retain set)
    if len(sample_retain) > 0:
        head_idx = edge_index[0, sample_retain]
        tail_idx = edge_index[1, sample_retain]
        rel_idx = edge_type[sample_retain]
        
        pos_scores = model.predict_link(entity_ids, edge_index, edge_type, head_idx, tail_idx, rel_idx)
        neg_tail = torch.randint(0, data['num_entities'], (len(sample_retain),), device=device)
        neg_scores = model.predict_link(entity_ids, edge_index, edge_type, head_idx, neg_tail, rel_idx)
        
        loss_all = F.margin_ranking_loss(
            pos_scores, neg_scores,
            torch.ones(len(sample_retain), device=device),
            margin=1.0,
            reduction='sum'
        )
    else:
        loss_all = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Loss1: Original graph, forget edges
    if len(sample_forget) > 0:
        head_idx = edge_index[0, sample_forget]
        tail_idx = edge_index[1, sample_forget]
        rel_idx = edge_type[sample_forget]
        
        pos_scores = model.predict_link(entity_ids, edge_index, edge_type, head_idx, tail_idx, rel_idx)
        neg_tail = torch.randint(0, data['num_entities'], (len(sample_forget),), device=device)
        neg_scores = model.predict_link(entity_ids, edge_index, edge_type, head_idx, neg_tail, rel_idx)
        
        loss1 = F.margin_ranking_loss(
            pos_scores, neg_scores,
            torch.ones(len(sample_forget), device=device),
            margin=1.0,
            reduction='sum'
        )
    else:
        loss1 = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Loss2: Modified graph (after deletion), influenced edges
    # Note: forget edges are removed in modified graph, so we use influenced edges that remain
    if modified_edge_index.size(1) > 0:
        # Get influenced edges that are still in the modified graph (i.e., in retain set)
        influenced_retain_mask = influenced_mask & retain_mask
        influenced_indices = influenced_retain_mask.nonzero(as_tuple=True)[0]
        
        if len(influenced_indices) > 0:
            num_influenced_sample = min(512, len(influenced_indices))
            sample_influenced = influenced_indices[torch.randperm(len(influenced_indices))[:num_influenced_sample]]
            
            # Map to modified graph indices
            # Find which edges in modified graph correspond to these influenced edges
            retain_edge_indices = retain_mask.nonzero(as_tuple=True)[0]
            # Create mapping from original edge index to modified edge index
            edge_mapping = {int(orig_idx): i for i, orig_idx in enumerate(retain_edge_indices)}
            
            # Get valid sample indices that exist in modified graph
            valid_samples = []
            for orig_idx in sample_influenced:
                if int(orig_idx) in edge_mapping:
                    valid_samples.append(edge_mapping[int(orig_idx)])
            
            if len(valid_samples) > 0:
                valid_samples = torch.tensor(valid_samples, device=device)
                head_idx = modified_edge_index[0, valid_samples]
                tail_idx = modified_edge_index[1, valid_samples]
                rel_idx = modified_edge_type[valid_samples]
                
                pos_scores = model.predict_link(entity_ids, modified_edge_index, modified_edge_type, head_idx, tail_idx, rel_idx)
                neg_tail = torch.randint(0, data['num_entities'], (len(valid_samples),), device=device)
                neg_scores = model.predict_link(entity_ids, modified_edge_index, modified_edge_type, head_idx, neg_tail, rel_idx)
                
                loss2 = F.margin_ranking_loss(
                    pos_scores, neg_scores,
                    torch.ones(len(valid_samples), device=device),
                    margin=1.0,
                    reduction='sum'
                )
            else:
                loss2 = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            loss2 = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        loss2 = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Compute gradients (allow_unused=True: some params may not affect loss)
    grad_all = grad(loss_all, model_params, retain_graph=True, create_graph=True, allow_unused=True)
    grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True, allow_unused=True)
    grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True, allow_unused=True)
    grad_all = tuple(g if g is not None else torch.zeros_like(p) for g, p in zip(grad_all, model_params))
    grad1 = tuple(g if g is not None else torch.zeros_like(p) for g, p in zip(grad1, model_params))
    grad2 = tuple(g if g is not None else torch.zeros_like(p) for g, p in zip(grad2, model_params))
    
    return grad_all, grad1, grad2


def apply_gif_approximation(model, grad_all, grad1, grad2, iteration, scale, damp, device):
    """
    Apply GIF approximation using Hessian-vector product (HVP) iteration
    
    Args:
        model: The model
        grad_all: Gradient on all training data
        grad1: Gradient on original graph with forget set
        grad2: Gradient on modified graph with forget set
        iteration: Number of HVP iterations
        scale: Scaling factor
        damp: Damping factor
        device: Device
        
    Returns:
        model: Model with updated parameters
    """
    model_params = [p for p in model.parameters() if p.requires_grad]
    
    # GIF: v = grad1 - grad2 (difference accounts for structural dependencies)
    # IF: v = grad1 (standard influence function)
    v = tuple(g1 - g2 for g1, g2 in zip(grad1, grad2))
    
    # Initialize h_estimate
    h_estimate = tuple(g1 - g2 for g1, g2 in zip(grad1, grad2))
    
    # Iterative HVP approximation
    pbar = tqdm(range(iteration), desc="GIF HVP Iteration")
    for _ in pbar:
        # Compute Hessian-vector product
        hv = compute_hvp(grad_all, model_params, h_estimate)
        
        # Update h_estimate: h = v + (1-damp)*h - hv/scale
        with torch.no_grad():
            h_estimate = tuple(
                v1 + (1 - damp) * h1 - hv1 / scale
                for v1, h1, hv1 in zip(v, h_estimate, hv)
            )
    
    # Estimate parameter change
    params_change = tuple(h_est / scale for h_est in h_estimate)
    
    # Apply parameter changes
    with torch.no_grad():
        for param, change in zip(model_params, params_change):
            param.data = param.data + change
    
    return model


def compute_hvp(grad_all, model_params, h_estimate):
    """
    Compute Hessian-vector product: H * h_estimate
    where H is the Hessian of the loss on all training data
    
    Args:
        grad_all: Gradient on all training data
        model_params: Model parameters
        h_estimate: Vector to multiply with Hessian
        
    Returns:
        hv: Hessian-vector product
    """
    # Compute element-wise product: sum(grad_all * h_estimate)
    element_product = 0
    for grad_elem, h_elem in zip(grad_all, h_estimate):
        element_product += torch.sum(grad_elem * h_elem)
    
    # Compute gradient of element_product w.r.t. model_params
    # This gives us H * h_estimate (allow_unused=True for params not in graph)
    hv = grad(element_product, model_params, create_graph=True, retain_graph=True, allow_unused=True)
    hv = tuple(h if h is not None else torch.zeros_like(p) for h, p in zip(hv, model_params))
    
    return hv
