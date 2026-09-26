"""
Evaluation utilities for graph unlearning
Provides functions to calculate Forget Effect (FE) and Model Utility (MU)
"""

import torch
import numpy as np
from tqdm import tqdm
from fe_evaluation_methods import (
    fe_standard,
    fe_mia,
    fe_backdoor,
    fe_embedding,
    calculate_forget_effect_multi_method
)

def calculate_forget_effect(model, data, is_kg, forget_mask, retain_mask, device, 
                           method='standard', train_mask=None, test_mask=None,
                           use_multi_method=False, methods=['standard'],
                           mia_evaluator=None, backdoor_evaluator=None,
                           verbose=False):
    """
    Calculate Forget Effect (FE) using one or multiple evaluation methods
    Lower performance on forget set indicates better unlearning
    
    IMPORTANT: Evaluation is performed WITHOUT using forget set structure
    to avoid graph structure leakage.
    
    Args:
        model: The GNN model to evaluate
        data: Graph data
        is_kg: Whether it's a knowledge graph
        forget_mask: Mask for forget set
        retain_mask: Mask for retain set
        device: Device to run on
        method: Single method to use. Options:
                - 'standard': Standard accuracy/MRR-based FE (default)
                - 'mia': Membership Inference Attack-based FE
                - 'backdoor': Backdoor-based FE
                - 'embedding': Embedding distance-based FE
        train_mask: Training mask (required for MIA)
        test_mask: Test mask (required for MIA)
        use_multi_method: If True, evaluate using multiple methods
        methods: List of methods to use (when use_multi_method=True)
        mia_evaluator: Pre-trained MIA evaluator (optional, for efficiency)
        backdoor_evaluator: Pre-initialized backdoor evaluator (optional)
        verbose: Whether to print detailed information
    
    Returns:
        If use_multi_method=False:
            fe_score: Single forget effect score (0-1, higher = better)
        If use_multi_method=True:
            results: Dictionary with FE scores from each method
            evaluators: Dictionary with reusable evaluators
    
    For homogeneous graphs: FE = 1 - accuracy on forget set (standard method)
    For knowledge graphs: FE = 1 - MRR on forget set (standard method)
    
    Higher FE is better (closer to 1 means better forgetting)
    """
    
    # Multi-method evaluation
    if use_multi_method:
        return calculate_forget_effect_multi_method(
            model=model,
            data=data,
            is_kg=is_kg,
            forget_mask=forget_mask,
            retain_mask=retain_mask,
            device=device,
            methods=methods,
            train_mask=train_mask,
            test_mask=test_mask,
            mia_evaluator=mia_evaluator,
            backdoor_evaluator=backdoor_evaluator,
            verbose=verbose
        )
    
    # Single method evaluation
    if method == 'standard':
        return fe_standard(model, data, is_kg, forget_mask, retain_mask, device)
    
    elif method == 'mia':
        if train_mask is None or test_mask is None:
            raise ValueError("train_mask and test_mask are required for MIA-based evaluation")
        fe_score, _ = fe_mia(
            model, data, is_kg, forget_mask, retain_mask,
            train_mask, test_mask, device, mia_evaluator
        )
        return fe_score
    
    elif method == 'backdoor':
        fe_score, _ = fe_backdoor(
            model, data, is_kg, forget_mask, retain_mask,
            device, backdoor_evaluator
        )
        return fe_score
    
    elif method == 'embedding':
        fe_score, _ = fe_embedding(
            model, data, is_kg, forget_mask, retain_mask, device
        )
        return fe_score
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'standard', 'mia', 'backdoor', or 'embedding'")


def calculate_model_utility(model, data, is_kg, retain_mask, test_mask, forget_mask, device, 
                            use_original_graph=False):
    """
    Calculate Model Utility (MU)
    Performance on retain/test set
    
    Args:
        model: The GNN model
        data: Graph data
        is_kg: Whether it's a knowledge graph
        retain_mask: Mask for retain set
        test_mask: Mask for test set
        forget_mask: Mask for forget set
        device: Device to run on
        use_original_graph: If True, use original graph structure for evaluation
                           If False (default), remove forget nodes' edges
    
    For homogeneous graphs: MU = accuracy on test set
    For knowledge graphs: MU = MRR on retain set
    
    Higher MU is better
    
    NOTE: When use_original_graph=False, edges connected to forget nodes are removed.
          This can sometimes lead to counter-intuitive results where larger unlearn_rate
          gives higher MU (due to graph sparsification reducing noise propagation).
          Use use_original_graph=True to compare models on identical graph structures.
    """
    model.eval()
    
    if is_kg:
        # For KG, use retain set and ONLY retain edges for encoding
        mu = evaluate_knowledge_graph_subset(model, data, device, retain_mask, retain_mask, num_samples=1000)
    else:
        if use_original_graph:
            # Use original graph structure (no edge removal)
            mu = evaluate_homogeneous_original(model, data, device, test_mask)
        else:
            # For homogeneous graphs, use test set but remove forget nodes' edges
            mu = evaluate_homogeneous_filtered(model, data, device, test_mask, forget_mask)
    
    return mu


def evaluate_homogeneous_original(model, data, device, eval_mask):
    """
    Evaluate model on homogeneous graph using ORIGINAL graph structure
    
    This provides a consistent evaluation baseline across different unlearn_rates.
    
    Args:
        model: The GNN model
        data: Graph data
        device: Device to run on
        eval_mask: Mask for nodes to evaluate on (e.g., test_mask)
    
    Returns:
        Accuracy on eval_mask using original graph structure
    """
    model.eval()
    
    with torch.no_grad():
        data = data.to(device)
        eval_mask = eval_mask.to(device)
        
        # Use original graph structure (no edge filtering)
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        if eval_mask.sum() == 0:
            return 0.0
        
        correct = (pred[eval_mask] == data.y[eval_mask]).sum()
        acc = correct.float() / eval_mask.sum()
    
    return acc.item()


def evaluate_knowledge_graph_subset(model, data, device, eval_mask, encoding_mask, num_samples=1000):
    """
    Evaluate knowledge graph on a subset of edges
    
    Args:
        model: The GNN model
        data: Graph data
        device: Device to run on
        eval_mask: Mask for edges to evaluate on
        encoding_mask: Mask for edges to use for entity encoding (should exclude forget edges)
        num_samples: Number of samples to evaluate
    
    Returns:
        MRR score
    """
    model.eval()
    
    with torch.no_grad():
        edge_index = data['train']['edge_index'].to(device)
        edge_type = data['train']['edge_type'].to(device)
        entity_ids = data['entity_ids'].to(device)
        eval_mask = eval_mask.to(device)  # Ensure on same device
        encoding_mask = encoding_mask.to(device)  # Ensure on same device
        
        # Get edges for evaluation
        eval_indices = eval_mask.nonzero(as_tuple=True)[0]
        if len(eval_indices) == 0:
            return 0.0
        
        # ✅ Use only encoding_mask edges for entity representation
        encoding_edge_index = edge_index[:, encoding_mask]
        encoding_edge_type = edge_type[encoding_mask]
        
        num_samples = min(len(eval_indices), num_samples)
        sample_idx = eval_indices[torch.randperm(len(eval_indices))[:num_samples]]
        
        ranks = []
        for i in tqdm(range(num_samples), desc="Evaluating Knowledge Graph"):
            idx = sample_idx[i]
            h = edge_index[0, idx:idx+1]
            t = edge_index[1, idx:idx+1]
            r = edge_type[idx:idx+1]
            
            all_entities = torch.arange(data['num_entities'], device=device)
            # Use encoding graph (without forget edges) for prediction
            scores = model.predict_link(entity_ids, encoding_edge_index, encoding_edge_type,
                                      h.repeat(data['num_entities']), all_entities,
                                      r.repeat(data['num_entities']))
            
            sorted_indices = torch.argsort(scores, descending=True)
            rank = (sorted_indices == t).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)
        
        mrr = np.mean([1.0 / r for r in ranks])
    
    return mrr


def evaluate_homogeneous_filtered(model, data, device, eval_mask, forget_mask):
    """
    Evaluate model on homogeneous graph WITHOUT using forget set structure
    
    Args:
        model: The GNN model
        data: Graph data
        device: Device to run on
        eval_mask: Mask for nodes to evaluate on (e.g., test_mask)
        forget_mask: Mask for forget nodes (to remove their edges)
    
    Returns:
        Accuracy on eval_mask
    
    Note: For GNNDelete-style wrappers (have .deletion and .affected_mask), we pass
    the ORIGINAL graph so the wrapper can apply its deletion operator correctly
    (it was trained to map base(original) -> base(modified) on affected nodes).
    Passing filtered graph would feed base(modified) into deletion (out-of-distribution).
    """
    model.eval()
    
    with torch.no_grad():
        data = data.to(device)
        forget_mask = forget_mask.to(device)  # Ensure on same device
        eval_mask = eval_mask.to(device)  # Ensure on same device
        
        # GNNDelete wrapper: trained to output "as if modified" when given ORIGINAL graph.
        # So use original graph here; do not filter (wrapper internally simulates removal).
        if hasattr(model, 'deletion') and hasattr(model, 'affected_mask'):
            edge_index_to_use = data.edge_index
        else:
            edge_mask = ~forget_mask[data.edge_index[0]] & ~forget_mask[data.edge_index[1]]
            edge_index_to_use = data.edge_index[:, edge_mask]
        
        out = model(data.x, edge_index_to_use)
        pred = out.argmax(dim=1)
        
        if eval_mask.sum() == 0:
            return 0.0
        
        correct = (pred[eval_mask] == data.y[eval_mask]).sum()
        acc = correct.float() / eval_mask.sum()
    
    return acc.item()

