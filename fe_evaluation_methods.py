"""
Forget Effect Evaluation Methods

This module provides four different methods to evaluate forget effect:
1. Standard FE: Based on accuracy/MRR on forget set
2. MIA-based FE: Based on Membership Inference Attack
3. Backdoor-based FE: Based on backdoor trigger removal
4. Embedding-based FE: Based on embedding distance analysis

All methods return scores such that higher is better.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


# ============================================================================
# Method 1: Standard Forget Effect (Accuracy/MRR-based)
# ============================================================================

def fe_standard(model, data, is_kg, forget_mask, retain_mask, device):
    """
    Method 1: Standard Forget Effect
    
    Measures prediction accuracy/MRR on forget set.
    Lower accuracy = Better forgetting
    
    For homogeneous graphs: FE = 1 - accuracy on forget set
    For knowledge graphs: FE = 1 - MRR on forget set
    
    Args:
        model: The GNN model to evaluate
        data: Graph data
        is_kg: Whether it's a knowledge graph
        forget_mask: Mask for forget set
        retain_mask: Mask for retain set
        device: Device to run on
    
    Returns:
        fe_score: Forget effect score (0-1, higher = better)
    """
    model.eval()
    
    with torch.no_grad():
        if is_kg:
            # For KG, evaluate on forget edges using ONLY retain edges for encoding
            edge_index = data['train']['edge_index'].to(device)
            edge_type = data['train']['edge_type'].to(device)
            entity_ids = data['entity_ids'].to(device)
            forget_mask = forget_mask.to(device)
            retain_mask = retain_mask.to(device)
            
            forget_indices = forget_mask.nonzero(as_tuple=True)[0]
            if len(forget_indices) == 0:
                return 1.0
            
            # Use ONLY retain edges to encode entities (no forget edge information)
            retain_edge_index = edge_index[:, retain_mask]
            retain_edge_type = edge_type[retain_mask]
            
            # Sample for efficiency
            num_samples = min(len(forget_indices), 1000)
            sample_idx = forget_indices[torch.randperm(len(forget_indices))[:num_samples]]
            
            ranks = []
            
            for i in tqdm(range(num_samples), desc="[Standard FE] Evaluating Forget Set", leave=False):
                idx = sample_idx[i]
                h = edge_index[0, idx:idx+1]
                t = edge_index[1, idx:idx+1]
                r = edge_type[idx:idx+1]
                
                # Score against all entities using ONLY retain graph
                all_entities = torch.arange(data['num_entities'], device=device)
                scores = model.predict_link(entity_ids, retain_edge_index, retain_edge_type,
                                          h.repeat(data['num_entities']), all_entities, 
                                          r.repeat(data['num_entities']))
                
                sorted_indices = torch.argsort(scores, descending=True)
                rank = (sorted_indices == t).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(rank)
            
            mrr = np.mean([1.0 / r for r in ranks])
            fe = 1.0 - mrr  # Higher is better
            
        else:
            # For homogeneous graphs, evaluate on forget nodes
            # Remove edges connected to forget nodes (no forget structure leakage)
            data = data.to(device)
            forget_mask = forget_mask.to(device)
            
            # Create edge mask: keep only edges where BOTH endpoints are NOT in forget set
            edge_mask = ~forget_mask[data.edge_index[0]] & ~forget_mask[data.edge_index[1]]
            filtered_edge_index = data.edge_index[:, edge_mask]
            
            # Evaluate using filtered graph (without forget nodes' edges)
            out = model(data.x, filtered_edge_index)
            pred = out.argmax(dim=1)
            
            if forget_mask.sum() == 0:
                return 1.0
            
            correct = (pred[forget_mask] == data.y[forget_mask]).sum()
            acc = correct.float() / forget_mask.sum()
            fe = 1.0 - acc.item()  # Higher is better
    
    return fe


# ============================================================================
# Method 2: MIA-based Forget Effect
# ============================================================================

class MIAEvaluator:
    """Helper class for MIA evaluation (reusable attack model)"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.attack_model = None
        
    def extract_features(self, model, data, is_kg, node_mask):
        """Extract features for MIA attack"""
        model.eval()
        
        with torch.no_grad():
            if is_kg:
                # Simplified for KG (not fully implemented in this version)
                return np.array([[0.5, 0.5, 0.5, 0.5]])
            else:
                # For homogeneous graphs: use node-level features
                data = data.to(self.device)
                if node_mask is not None:
                    node_mask = node_mask.to(self.device)
                
                out = model(data.x, data.edge_index)
                probs = F.softmax(out, dim=1)
                
                # Feature 1: Maximum confidence
                max_conf, _ = probs.max(dim=1)
                
                # Feature 2: Entropy
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                
                # Feature 3: Modified loss (cross-entropy with true label)
                loss = F.cross_entropy(out, data.y, reduction='none')
                
                # Feature 4: Top-2 confidence gap
                top2_conf, _ = torch.topk(probs, k=2, dim=1)
                conf_gap = top2_conf[:, 0] - top2_conf[:, 1]
                
                # Combine features
                features = torch.stack([max_conf, entropy, loss, conf_gap], dim=1)
                
                if node_mask is not None:
                    features = features[node_mask]
                
                return features.cpu().numpy()
    
    def train_attack(self, model, data, is_kg, train_mask, non_train_mask):
        """Train MIA attack model"""
        # Extract features for training data (members)
        member_features = self.extract_features(model, data, is_kg, train_mask)
        member_labels = np.ones(len(member_features))
        
        # Extract features for non-training data (non-members)
        non_member_features = self.extract_features(model, data, is_kg, non_train_mask)
        non_member_labels = np.zeros(len(non_member_features))
        
        # Combine
        X = np.vstack([member_features, non_member_features])
        y = np.concatenate([member_labels, non_member_labels])
        
        # Train attack model (Random Forest)
        self.attack_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        self.attack_model.fit(X, y)
        
        return self.attack_model


def fe_mia(model, data, is_kg, forget_mask, retain_mask, train_mask, test_mask, device, 
           mia_evaluator=None):
    """
    Method 2: MIA-based Forget Effect
    
    Uses Membership Inference Attack to measure if forget set can be
    distinguished from non-training data.
    
    Lower MIA accuracy on forget set = Better forgetting
    
    Args:
        model: The GNN model to evaluate
        data: Graph data
        is_kg: Whether it's a knowledge graph
        forget_mask: Mask for forget set
        retain_mask: Mask for retain set
        train_mask: Mask for training set (for MIA training)
        test_mask: Mask for test set (for MIA training)
        device: Device to run on
        mia_evaluator: Pre-trained MIA evaluator (optional, for efficiency)
    
    Returns:
        fe_score: Forget effect score (0-1, higher = better)
        mia_evaluator: The MIA evaluator (can be reused)
    """
    if is_kg:
        # MIA not fully implemented for KG yet
        print("  ⚠️  MIA-based FE not fully implemented for knowledge graphs")
        return 0.5, None
    
    # Initialize or use provided MIA evaluator
    if mia_evaluator is None:
        mia_evaluator = MIAEvaluator(device=device)
        # Train attack model
        mia_evaluator.train_attack(model, data, is_kg, train_mask, test_mask)
    
    # Extract features for forget set
    forget_features = mia_evaluator.extract_features(model, data, is_kg, forget_mask)
    
    # Predict membership (should be 0 if successfully forgotten)
    forget_pred = mia_evaluator.attack_model.predict(forget_features)
    
    # Calculate MIA accuracy
    true_labels = np.ones(len(forget_features))  # They were members before
    mia_accuracy = accuracy_score(true_labels, forget_pred)
    
    # Unlearning score: lower MIA accuracy = better
    # Normalize to 0-1 range where 0.5 (random) maps to 0.5
    # Perfect forgetting (MIA acc = 0.5) → fe_score = 0.5
    # No forgetting (MIA acc = 1.0) → fe_score = 0.0
    # Over-forgetting (MIA acc = 0.0) → fe_score = 1.0
    fe_score = 1.0 - mia_accuracy
    
    return fe_score, mia_evaluator


# ============================================================================
# Method 3: Backdoor-based Forget Effect
# ============================================================================

class BackdoorEvaluator:
    """Helper class for backdoor evaluation"""
    
    def __init__(self, device='cuda', trigger_size=5):
        self.device = device
        self.trigger_size = trigger_size
        self.trigger_pattern = torch.ones(trigger_size, device=device)
        self.target_class = 0
    
    def apply_trigger(self, data, node_indices):
        """Apply backdoor trigger to specified nodes"""
        triggered_data = data.clone() if hasattr(data, 'clone') else data
        triggered_data = triggered_data.to(self.device)
        
        for idx in node_indices:
            triggered_data.x[idx, -self.trigger_size:] = self.trigger_pattern
        
        return triggered_data


def fe_backdoor(model, data, is_kg, forget_mask, retain_mask, device, 
                backdoor_evaluator=None, target_class=0):
    """
    Method 3: Backdoor-based Forget Effect
    
    Tests if backdoor trigger planted in forget set has been removed.
    Lower backdoor success rate = Better forgetting
    
    Note: This assumes a backdoor was injected during training.
    
    Args:
        model: The GNN model to evaluate
        data: Graph data
        is_kg: Whether it's a knowledge graph
        forget_mask: Mask for forget set
        retain_mask: Mask for retain set
        device: Device to run on
        backdoor_evaluator: Pre-initialized backdoor evaluator (optional)
        target_class: Target class for backdoor
    
    Returns:
        fe_score: Forget effect score (0-1, higher = better)
        backdoor_evaluator: The backdoor evaluator (can be reused)
    """
    if is_kg:
        # Backdoor not implemented for KG yet
        print("  ⚠️  Backdoor-based FE not implemented for knowledge graphs")
        return 0.5, None
    
    model.eval()
    
    # Initialize backdoor evaluator if not provided
    if backdoor_evaluator is None:
        backdoor_evaluator = BackdoorEvaluator(device=device)
        backdoor_evaluator.target_class = target_class
    
    forget_mask = forget_mask.to(device)
    forget_indices = forget_mask.nonzero(as_tuple=True)[0]
    
    if len(forget_indices) == 0:
        return 1.0, backdoor_evaluator
    
    # Apply trigger to forget set
    triggered_data = backdoor_evaluator.apply_trigger(data, forget_indices)
    
    # Predict on triggered data
    with torch.no_grad():
        out = model(triggered_data.x, triggered_data.edge_index)
        pred = out.argmax(dim=1)
    
    # Calculate success rate (how many are classified as target class)
    triggered_pred = pred[forget_indices]
    backdoor_success = (triggered_pred == target_class).sum().item()
    backdoor_success_rate = backdoor_success / len(forget_indices)
    
    # Unlearning score: lower backdoor success = better
    fe_score = 1.0 - backdoor_success_rate
    
    return fe_score, backdoor_evaluator


# ============================================================================
# Method 4: Embedding-based Forget Effect
# ============================================================================

def fe_embedding(model, data, is_kg, forget_mask, retain_mask, device):
    """
    Method 4: Embedding Distance-based Forget Effect
    
    Analyzes embedding distances to measure forgetting.
    Better forgetting means:
    - Higher distance between forget and retain embeddings
    - Lower distance between forget and random embeddings
    
    Args:
        model: The GNN model to evaluate
        data: Graph data
        is_kg: Whether it's a knowledge graph
        forget_mask: Mask for forget set
        retain_mask: Mask for retain set
        device: Device to run on
    
    Returns:
        fe_score: Forget effect score (0-1, higher = better)
        details: Dictionary with detailed metrics
    """
    if is_kg:
        # Embedding analysis not fully implemented for KG yet
        print("  ⚠️  Embedding-based FE not fully implemented for knowledge graphs")
        return 0.5, {}
    
    model.eval()
    
    data = data.to(device)
    forget_mask = forget_mask.to(device)
    retain_mask = retain_mask.to(device)
    
    with torch.no_grad():
        # Get embeddings
        out = model(data.x, data.edge_index)
        
        # If model has get_embedding method, use it
        if hasattr(model, 'get_embedding'):
            embeddings = model.get_embedding(data.x, data.edge_index)
        else:
            # Use output as embeddings
            embeddings = out
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Get forget and retain embeddings
        forget_emb = embeddings[forget_mask]
        retain_emb = embeddings[retain_mask]
        
        if len(forget_emb) == 0 or len(retain_emb) == 0:
            return 0.5, {}
        
        # Sample for efficiency (if too many nodes)
        max_samples = 500
        if len(forget_emb) > max_samples:
            forget_indices = torch.randperm(len(forget_emb))[:max_samples]
            forget_emb = forget_emb[forget_indices]
        if len(retain_emb) > max_samples:
            retain_indices = torch.randperm(len(retain_emb))[:max_samples]
            retain_emb = retain_emb[retain_indices]
        
        # Metric 1: Average distance from forget to retain
        forget_retain_distances = torch.cdist(forget_emb, retain_emb, p=2)
        avg_forget_retain_dist = forget_retain_distances.mean().item()
        
        # Metric 2: Similarity to random embeddings
        random_emb = torch.randn_like(forget_emb)
        random_emb = F.normalize(random_emb, p=2, dim=1)
        forget_random_distances = torch.cdist(forget_emb, random_emb, p=2)
        avg_forget_random_dist = forget_random_distances.mean().item()
        
        # Metric 3: Concentration (how spread out are forget embeddings)
        forget_center = forget_emb.mean(dim=0, keepdim=True)
        concentration = torch.cdist(forget_emb, forget_center, p=2).mean().item()
        
    # Combine metrics into single FE score
    # Higher forget-retain distance = better (normalize by typical distance ~2.0)
    # Lower forget-random distance = better (normalize by typical distance ~1.4)
    # Higher concentration = better (embeddings collapsing, normalize by ~1.0)
    
    # Weighted combination (can be tuned)
    separation_score = min(avg_forget_retain_dist / 2.0, 1.0)  # 0-1
    randomness_score = max(1.0 - avg_forget_random_dist / 1.4, 0.0)  # 0-1
    concentration_score = min(concentration / 1.0, 1.0)  # 0-1
    
    # Combine: prioritize separation and randomness
    fe_score = 0.4 * separation_score + 0.4 * randomness_score + 0.2 * concentration_score
    
    details = {
        'forget_retain_distance': avg_forget_retain_dist,
        'forget_random_distance': avg_forget_random_dist,
        'concentration': concentration,
        'separation_score': separation_score,
        'randomness_score': randomness_score,
        'concentration_score': concentration_score
    }
    
    return fe_score, details


# ============================================================================
# Unified Interface
# ============================================================================

def calculate_forget_effect_multi_method(
    model, 
    data, 
    is_kg, 
    forget_mask, 
    retain_mask, 
    device,
    methods=['standard'],
    train_mask=None,
    test_mask=None,
    mia_evaluator=None,
    backdoor_evaluator=None,
    verbose=True
):
    """
    Calculate forget effect using multiple methods
    
    Args:
        model: The GNN model to evaluate
        data: Graph data
        is_kg: Whether it's a knowledge graph
        forget_mask: Mask for forget set
        retain_mask: Mask for retain set
        device: Device to run on
        methods: List of methods to use. Options:
                 - 'standard': Standard accuracy/MRR-based FE
                 - 'mia': Membership Inference Attack-based FE
                 - 'backdoor': Backdoor-based FE
                 - 'embedding': Embedding distance-based FE
                 - 'all': All methods
        train_mask: Training mask (required for MIA)
        test_mask: Test mask (required for MIA)
        mia_evaluator: Pre-trained MIA evaluator (optional)
        backdoor_evaluator: Pre-initialized backdoor evaluator (optional)
        verbose: Whether to print progress
    
    Returns:
        results: Dictionary with FE scores from each method
        evaluators: Dictionary with reusable evaluators
    """
    if 'all' in methods:
        methods = ['standard', 'mia', 'backdoor', 'embedding']
    
    results = {}
    evaluators = {
        'mia': mia_evaluator,
        'backdoor': backdoor_evaluator
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Calculating Forget Effect using {len(methods)} method(s)")
        print(f"{'='*70}")
    
    # Method 1: Standard FE
    if 'standard' in methods:
        if verbose:
            print(f"\n[Method 1/4] Standard FE (Accuracy/MRR-based)...")
        fe_std = fe_standard(model, data, is_kg, forget_mask, retain_mask, device)
        results['fe_standard'] = fe_std
        if verbose:
            print(f"  ✓ Standard FE: {fe_std:.4f}")
    
    # Method 2: MIA-based FE
    if 'mia' in methods:
        if verbose:
            print(f"\n[Method 2/4] MIA-based FE...")
        if train_mask is None or test_mask is None:
            if verbose:
                print(f"  ⚠️  Skipping MIA: train_mask and test_mask required")
            results['fe_mia'] = None
        else:
            try:
                fe_mia_score, mia_eval = fe_mia(
                    model, data, is_kg, forget_mask, retain_mask, 
                    train_mask, test_mask, device, evaluators['mia']
                )
                results['fe_mia'] = fe_mia_score
                evaluators['mia'] = mia_eval
                if verbose:
                    print(f"  ✓ MIA-based FE: {fe_mia_score:.4f}")
            except Exception as e:
                if verbose:
                    print(f"  ⚠️  MIA evaluation failed: {e}")
                results['fe_mia'] = None
    
    # Method 3: Backdoor-based FE
    if 'backdoor' in methods:
        if verbose:
            print(f"\n[Method 3/4] Backdoor-based FE...")
        try:
            fe_bd_score, bd_eval = fe_backdoor(
                model, data, is_kg, forget_mask, retain_mask, 
                device, evaluators['backdoor']
            )
            results['fe_backdoor'] = fe_bd_score
            evaluators['backdoor'] = bd_eval
            if verbose:
                print(f"  ✓ Backdoor-based FE: {fe_bd_score:.4f}")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Backdoor evaluation failed: {e}")
            results['fe_backdoor'] = None
    
    # Method 4: Embedding-based FE
    if 'embedding' in methods:
        if verbose:
            print(f"\n[Method 4/4] Embedding-based FE...")
        try:
            fe_emb_score, emb_details = fe_embedding(
                model, data, is_kg, forget_mask, retain_mask, device
            )
            results['fe_embedding'] = fe_emb_score
            results['embedding_details'] = emb_details
            if verbose:
                print(f"  ✓ Embedding-based FE: {fe_emb_score:.4f}")
                if emb_details:
                    print(f"    - Forget-Retain Distance: {emb_details.get('forget_retain_distance', 0):.4f}")
                    print(f"    - Forget-Random Distance: {emb_details.get('forget_random_distance', 0):.4f}")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Embedding evaluation failed: {e}")
            results['fe_embedding'] = None
            results['embedding_details'] = {}
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Forget Effect Evaluation Complete")
        print(f"{'='*70}\n")
    
    return results, evaluators

