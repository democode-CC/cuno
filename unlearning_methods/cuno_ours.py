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



def curriculum_gradient_ascent(args, model, data, is_kg, forget_mask, retain_mask):
    """
    Variant 1: Gradient Ascent + Curriculum Unlearning (Step 1) (a.k.a. without NPO)
    
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
    print("Curriculum Gradient Ascent (curriculum_ga)")
    print(f"Curricula: {args.num_curricula}, Metric: {args.complexity_metric}, Mode: {args.curriculum_mode}")
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
    # Check forget-set size and adjust curriculum parameters dynamically
    forget_count = forget_mask.sum().item()
    MIN_STABLE_FORGET_SIZE = 10
    
    # Reduce the number of curricula if the forget set is too small
    effective_num_curricula = args.num_curricula
    if forget_count < effective_num_curricula * 2:  # Ensure at least 2 nodes per curriculum stage
        effective_num_curricula = max(1, forget_count // 2)
        print(f"⚠️  Small forget set ({forget_count} nodes), reducing curricula: {args.num_curricula} -> {effective_num_curricula}")
    
    # Adjust the learning rate dynamically
    if forget_count < MIN_STABLE_FORGET_SIZE:
        lr_scale = forget_count / MIN_STABLE_FORGET_SIZE
        adjusted_lr = args.unlearn_lr * lr_scale
        print(f"⚠️  Reducing learning rate: {args.unlearn_lr} -> {adjusted_lr:.6f}")
    else:
        adjusted_lr = args.unlearn_lr
    
    # Set the retain regularization weight
    retain_weight = getattr(args, 'retain_weight', 0.1)
    if forget_count < MIN_STABLE_FORGET_SIZE:
        retain_weight = 0.5
        print(f"  Using stronger retain regularization: {retain_weight}")
    
    # Design curricula
    designer = CurriculumDesigner(
        forget_mask=forget_mask,
        data=data,
        is_kg=is_kg,
        complexity_metric=args.complexity_metric,
        num_curricula=effective_num_curricula,
        mode=args.curriculum_mode,
        overlap_ratio=args.overlap_ratio,
        model=model,
        retain_mask=retain_mask,
        device=str(device),
        num_layers=getattr(args, 'num_layers', 2),
        hop_decay=getattr(args, 'hop_decay', 0.5),
        curriculum_order=getattr(args, 'curriculum_order', 'easy_to_hard'),
    )
    curricula = designer.design_curricula()
    
    print(f"Designed {len(curricula)} curricula")
    for i, curriculum in enumerate(curricula):
        print(f"  Curriculum {i+1}: {curriculum.sum().item()} nodes/edges")
    
    # Unlearn curriculum by curriculum
    epochs_per_curriculum = args.unlearn_epochs // len(curricula)
    
    for curriculum_idx, curriculum_mask in enumerate(curricula):
        print(f"\nUnlearning Curriculum {curriculum_idx + 1}/{len(curricula)}...")
        
        # Skip empty curriculum stages
        curriculum_count = curriculum_mask.sum().item()
        if curriculum_count == 0:
            print(f"  Skipping empty curriculum")
            continue
        
        optimizer = optim.Adam(model.parameters(), lr=adjusted_lr)
        
        pbar = tqdm(range(epochs_per_curriculum), desc=f"Curriculum {curriculum_idx+1}")
        for epoch in pbar:
            model.train()
            optimizer.zero_grad()
            
            if is_kg:
                edge_index = data['train']['edge_index'].to(device)
                edge_type = data['train']['edge_type'].to(device)
                entity_ids = data['entity_ids'].to(device)
                
                curriculum_indices = curriculum_mask.nonzero(as_tuple=True)[0]
                if len(curriculum_indices) > 0:
                    batch_size = min(len(curriculum_indices), 512)
                    sample_idx = curriculum_indices[torch.randperm(len(curriculum_indices))[:batch_size]]
                    
                    head_idx = edge_index[0, sample_idx]
                    tail_idx = edge_index[1, sample_idx]
                    rel_idx = edge_type[sample_idx]
                    
                    scores = model.predict_link(entity_ids, edge_index, edge_type, head_idx, tail_idx, rel_idx)
                    loss = -scores.mean()
                    
                    loss.backward()
                    optimizer.step()
                    
                    pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            else:
                out = model(data.x, data.edge_index)
                
                # Forget loss + Retain regularization
                forget_loss = -F.cross_entropy(out[curriculum_mask], data.y[curriculum_mask])
                retain_loss = F.cross_entropy(out[retain_mask], data.y[retain_mask])
                loss = forget_loss + retain_weight * retain_loss
                
                loss.backward()
                optimizer.step()
                
                pbar.set_postfix({
                    'Forget': f'{forget_loss.item():.4f}',
                    'Retain': f'{retain_loss.item():.4f}'
                })
    
    print("Curriculum Gradient Ascent completed!")
    return model


def npo_gradient_ascent(args, model, data, is_kg, forget_mask, retain_mask):
    """
    Variant 2: Gradient Ascent + NPO (Step 2) (a.k.a. without Curriculum)
    NPO (Negative Preference Optimization) modifies the loss function
    
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
    print("NPO Gradient Ascent (npo_ga)")
    print(f"Beta: {args.npo_beta}, Lambda: {args.npo_lambda}, Temperature: {args.npo_temperature}")
    print("="*60)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if not is_kg:
        data = data.to(device)
    
    # Clone model for reference (frozen)
    reference_model = deepcopy(model)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(model.parameters(), lr=args.unlearn_lr)
    
    pbar = tqdm(range(args.unlearn_epochs), desc="NPO")
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        
        if is_kg:
            edge_index = data['train']['edge_index'].to(device)
            edge_type = data['train']['edge_type'].to(device)
            entity_ids = data['entity_ids'].to(device)
            
            # Forget loss (NPO-modified)
            forget_indices = forget_mask.nonzero(as_tuple=True)[0]
            if len(forget_indices) > 0:
                batch_size = min(len(forget_indices), 256)
                forget_sample = forget_indices[torch.randperm(len(forget_indices))[:batch_size]]
                
                head_idx = edge_index[0, forget_sample]
                tail_idx = edge_index[1, forget_sample]
                rel_idx = edge_type[forget_sample]
                
                # Current model scores
                scores_current = model.predict_link(entity_ids, edge_index, edge_type, head_idx, tail_idx, rel_idx)
                
                # Reference model scores
                with torch.no_grad():
                    scores_ref = reference_model.predict_link(entity_ids, edge_index, edge_type, head_idx, tail_idx, rel_idx)
                
                # NPO loss: encourage current model to deviate from reference on forget set
                forget_loss = -torch.mean(
                    F.logsigmoid(args.npo_beta * (scores_ref - scores_current) / args.npo_temperature)
                )
            else:
                forget_loss = 0.0
            
            # Retain loss (preserve utility)
            retain_indices = retain_mask.nonzero(as_tuple=True)[0]
            if len(retain_indices) > 0:
                batch_size = min(len(retain_indices), 256)
                retain_sample = retain_indices[torch.randperm(len(retain_indices))[:batch_size]]
                
                head_idx = edge_index[0, retain_sample]
                tail_idx = edge_index[1, retain_sample]
                rel_idx = edge_type[retain_sample]
                
                scores_current = model.predict_link(entity_ids, edge_index, edge_type, head_idx, tail_idx, rel_idx)
                
                # Generate negative samples
                neg_tail = torch.randint(0, data['num_entities'], (batch_size,), device=device)
                neg_scores = model.predict_link(entity_ids, edge_index, edge_type, head_idx, neg_tail, rel_idx)
                
                retain_loss = F.margin_ranking_loss(
                    scores_current, neg_scores,
                    torch.ones(batch_size, device=device),
                    margin=1.0
                )
            else:
                retain_loss = 0.0
            
            # Combined loss
            loss = args.npo_lambda * forget_loss + (1 - args.npo_lambda) * retain_loss
            
        else:
            out = model(data.x, data.edge_index)
            
            # Forget loss (NPO-modified)
            if forget_mask.sum() > 0:
                # Current model predictions
                logits_current = out[forget_mask]
                
                # Reference model predictions
                with torch.no_grad():
                    logits_ref = reference_model(data.x, data.edge_index)[forget_mask]
                
                # NPO loss: maximize difference from reference
                log_probs_current = F.log_softmax(logits_current / args.npo_temperature, dim=-1)
                log_probs_ref = F.log_softmax(logits_ref / args.npo_temperature, dim=-1)
                
                # KL divergence from reference (we want to maximize this)
                forget_loss = -args.npo_beta * F.kl_div(log_probs_current, log_probs_ref.detach(), 
                                                       reduction='batchmean', log_target=True)
            else:
                forget_loss = 0.0
            
            # Retain loss (preserve utility)
            if retain_mask.sum() > 0:
                retain_loss = F.cross_entropy(out[retain_mask], data.y[retain_mask])
            else:
                retain_loss = 0.0
            
            # Combined loss
            loss = args.npo_lambda * forget_loss + (1 - args.npo_lambda) * retain_loss
        
        if torch.is_tensor(loss):
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        else:
            pbar.set_postfix({'Loss': f'{loss:.4f}'})
    
    print("NPO Gradient Ascent completed!")
    return model


def _optimal_npo_loss(p_cur, p_ref, inf_weights, beta):
    """
    Optimal NPO forget loss (node classification).

    L = (2/β) · Σ_c p_ref(c) · log(1 + (p_cur(c) / p_ref(c))^β)

    Design rationale:
      - p_ref weighting: concentrates gradient on classes the reference model
        was confident about (reverse-KL philosophy), so forgetting pressure is
        targeted rather than diffuse across all C classes.
      - log(1 + r^β) form: self-regularising — when p_cur(c) → 0 (already
        forgotten), the ratio r → 0 and the loss → 0 automatically, preventing
        unnecessary perturbation of a model that has already unlearned.
      - Bounded gradient: ∂L/∂log_p_cur(c) ∈ [0, 2·p_ref(c)], unlike plain
        KL whose gradient can diverge.

    Args:
        p_cur:       [N, C] current model softmax probabilities
        p_ref:       [N, C] reference model softmax probabilities (detached)
        inf_weights: [N]    per-node influence / complexity scores (or None)
        beta:        float  sharpness parameter (larger → harder pressure)

    Returns:
        scalar loss (to be minimised)
    """
    # log(p_cur / p_ref) per class, clamped for numerical stability
    log_ratio = (torch.log(p_cur + 1e-8)
                 - torch.log(p_ref + 1e-8)).clamp(-30.0, 30.0)  # [N, C]
    ratio_beta = torch.exp(beta * log_ratio)                      # (p_cur/p_ref)^β

    # INPO-style ratio loss per class
    per_class = (2.0 / beta) * torch.log(1.0 + ratio_beta)       # [N, C]

    # p_ref-weighted sum over classes
    per_node = (p_ref * per_class).sum(dim=-1)                    # [N]

    if inf_weights is not None:
        per_node = per_node * inf_weights

    return per_node.mean()


def _optimal_npo_loss_kg(scores_cur, scores_ref, inf_weights, beta):
    """
    Optimal NPO forget loss (knowledge graph link prediction).

    KG analogue of _optimal_npo_loss using sigmoid-transformed link scores:

    L = (2/β) · mean_w [ log(1 + (σ(s_cur) / σ(s_ref))^β) ]

    Args:
        scores_cur:  [N] current model link scores (raw, before sigmoid)
        scores_ref:  [N] reference model link scores (detached)
        inf_weights: [N] per-edge influence scores (or None)
        beta:        float

    Returns:
        scalar loss
    """
    p_cur = scores_cur.sigmoid()
    p_ref = scores_ref.sigmoid().detach()

    log_ratio = (torch.log(p_cur + 1e-8)
                 - torch.log(p_ref + 1e-8)).clamp(-30.0, 30.0)
    ratio_beta = torch.exp(beta * log_ratio)
    per_edge = (2.0 / beta) * torch.log(1.0 + ratio_beta)        # [N]

    if inf_weights is not None:
        per_edge = per_edge * inf_weights

    return per_edge.mean()


def full_method_kl(args, model, data, is_kg, forget_mask, retain_mask):
    """
    Ablation: Full Method with KL-based NPO (Curriculum + KL forget objective)
    
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
    print("Curriculum + NPO (full_method / CUNO)")
    print(f"Curricula: {args.num_curricula}, Metric: {args.complexity_metric}")
    print(f"Beta: {args.npo_beta}, Lambda: {args.npo_lambda}")
    print("="*60)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if not is_kg:
        data = data.to(device)
    
    # Clone reference model
    reference_model = deepcopy(model)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    
    # Design curricula
    designer = CurriculumDesigner(
        forget_mask=forget_mask,
        data=data,
        is_kg=is_kg,
        complexity_metric=args.complexity_metric,
        num_curricula=args.num_curricula,
        mode=args.curriculum_mode,
        overlap_ratio=args.overlap_ratio,
        model=reference_model,
        retain_mask=retain_mask,
        device=str(device),
        num_layers=getattr(args, 'num_layers', 2),
        hop_decay=getattr(args, 'hop_decay', 0.5),
        curriculum_order=getattr(args, 'curriculum_order', 'easy_to_hard'),
    )
    curricula = designer.design_curricula()
    
    print(f"Designed {len(curricula)} curricula")
    
    # Unlearn curriculum by curriculum with NPO
    epochs_per_curriculum = args.unlearn_epochs // len(curricula)
    
    for curriculum_idx, curriculum_mask in enumerate(curricula):
        print(f"\nUnlearning Curriculum {curriculum_idx + 1}/{len(curricula)} with NPO...")
        
        optimizer = optim.Adam(model.parameters(), lr=args.unlearn_lr)
        
        pbar = tqdm(range(epochs_per_curriculum), desc=f"Curriculum {curriculum_idx+1}")
        for epoch in pbar:
            model.train()
            optimizer.zero_grad()
            
            if is_kg:
                edge_index = data['train']['edge_index'].to(device)
                edge_type = data['train']['edge_type'].to(device)
                entity_ids = data['entity_ids'].to(device)
                
                # NPO forget loss on current curriculum
                curriculum_indices = curriculum_mask.nonzero(as_tuple=True)[0]
                if len(curriculum_indices) > 0:
                    batch_size = min(len(curriculum_indices), 256)
                    sample_idx = curriculum_indices[torch.randperm(len(curriculum_indices))[:batch_size]]
                    
                    head_idx = edge_index[0, sample_idx]
                    tail_idx = edge_index[1, sample_idx]
                    rel_idx = edge_type[sample_idx]
                    
                    scores_current = model.predict_link(entity_ids, edge_index, edge_type, head_idx, tail_idx, rel_idx)
                    
                    with torch.no_grad():
                        scores_ref = reference_model.predict_link(entity_ids, edge_index, edge_type, head_idx, tail_idx, rel_idx)
                    
                    forget_loss = -torch.mean(
                        F.logsigmoid(args.npo_beta * (scores_ref - scores_current) / args.npo_temperature)
                    )
                else:
                    forget_loss = 0.0
                
                # Retain loss
                retain_indices = retain_mask.nonzero(as_tuple=True)[0]
                if len(retain_indices) > 0:
                    batch_size = min(len(retain_indices), 256)
                    retain_sample = retain_indices[torch.randperm(len(retain_indices))[:batch_size]]
                    
                    head_idx = edge_index[0, retain_sample]
                    tail_idx = edge_index[1, retain_sample]
                    rel_idx = edge_type[retain_sample]
                    
                    scores_current = model.predict_link(entity_ids, edge_index, edge_type, head_idx, tail_idx, rel_idx)
                    neg_tail = torch.randint(0, data['num_entities'], (batch_size,), device=device)
                    neg_scores = model.predict_link(entity_ids, edge_index, edge_type, head_idx, neg_tail, rel_idx)
                    
                    retain_loss = F.margin_ranking_loss(scores_current, neg_scores,
                                                       torch.ones(batch_size, device=device), margin=1.0)
                else:
                    retain_loss = 0.0
                
                loss = args.npo_lambda * forget_loss + (1 - args.npo_lambda) * retain_loss
                
            else:
                out = model(data.x, data.edge_index)
                
                # NPO forget loss on current curriculum
                if curriculum_mask.sum() > 0:
                    logits_current = out[curriculum_mask]
                    
                    with torch.no_grad():
                        logits_ref = reference_model(data.x, data.edge_index)[curriculum_mask]
                    
                    log_probs_current = F.log_softmax(logits_current / args.npo_temperature, dim=-1)
                    log_probs_ref = F.log_softmax(logits_ref / args.npo_temperature, dim=-1)
                    
                    forget_loss = -args.npo_beta * F.kl_div(log_probs_current, log_probs_ref.detach(),
                                                           reduction='batchmean', log_target=True)
                else:
                    forget_loss = 0.0
                
                # Retain loss
                if retain_mask.sum() > 0:
                    retain_loss = F.cross_entropy(out[retain_mask], data.y[retain_mask])
                else:
                    retain_loss = 0.0
                
                loss = args.npo_lambda * forget_loss + (1 - args.npo_lambda) * retain_loss
            
            if torch.is_tensor(loss):
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            else:
                pbar.set_postfix({'Loss': f'{loss:.4f}'})
    
    print("Full Method completed!")
    return model


def full_method(args, model, data, is_kg, forget_mask, retain_mask):
    """
    Full Method (CUNO): Curriculum scheduling + p_ref-weighted NPO (default).
    Curriculum scheduling + p_ref-weighted INPO ratio loss.

    Forget objective per stage k:
        L_forget = (2/β|C_k|) Σ_{v∈C_k} w_v Σ_c p_ref(c;τ)·log(1+(p_v(c;τ)/p_ref(c;τ))^β)

    Retain objective (same as full_method):
        L_retain = CE(model(retain), y_retain)

    Stage loss:
        L^(k) = λ·L_forget + (1-λ)·L_retain
    """
    set_unlearn_seed(getattr(args, 'seed', 42))

    print("\n" + "="*60)
    print("CUNO: Curriculum + distribution-level NPO (full_method)")
    print(f"Curricula: {args.num_curricula}, Metric: {args.complexity_metric}")
    print(f"Beta: {args.npo_beta}, Lambda: {args.npo_lambda}, Tau: {args.npo_temperature}")
    print("="*60)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not is_kg:
        data = data.to(device)

    # Frozen reference model
    reference_model = deepcopy(model)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    # Design curricula (use reference model for model-aware metrics)
    designer = CurriculumDesigner(
        forget_mask=forget_mask,
        data=data,
        is_kg=is_kg,
        complexity_metric=args.complexity_metric,
        num_curricula=args.num_curricula,
        mode=args.curriculum_mode,
        overlap_ratio=args.overlap_ratio,
        model=reference_model,
        retain_mask=retain_mask,
        device=str(device),
        num_layers=getattr(args, 'num_layers', 2),
        hop_decay=getattr(args, 'hop_decay', 0.5),
        curriculum_order=getattr(args, 'curriculum_order', 'easy_to_hard'),
    )
    curricula = designer.design_curricula()

    print(f"Designed {len(curricula)} curricula")

    beta = args.npo_beta
    tau  = args.npo_temperature
    lam  = args.npo_lambda

    epochs_per_curriculum = args.unlearn_epochs // len(curricula)

    for curriculum_idx, curriculum_mask in enumerate(curricula):
        print(f"\nUnlearning Curriculum {curriculum_idx + 1}/{len(curricula)} "
              f"with Optimal NPO...")

        if curriculum_mask.sum() == 0:
            print("  Skipping empty curriculum")
            continue

        optimizer = optim.Adam(model.parameters(), lr=args.unlearn_lr)

        pbar = tqdm(range(epochs_per_curriculum),
                    desc=f"Curriculum {curriculum_idx+1}")
        for epoch in pbar:
            model.train()
            optimizer.zero_grad()

            if is_kg:
                edge_index = data['train']['edge_index'].to(device)
                edge_type  = data['train']['edge_type'].to(device)
                entity_ids = data['entity_ids'].to(device)

                # ---- Forget loss (optimal NPO, KG) ----
                curriculum_indices = curriculum_mask.nonzero(as_tuple=True)[0]
                batch_size  = min(len(curriculum_indices), 256)
                sample_idx  = curriculum_indices[
                    torch.randperm(len(curriculum_indices))[:batch_size]
                ]
                h_idx = edge_index[0, sample_idx]
                t_idx = edge_index[1, sample_idx]
                r_idx = edge_type[sample_idx]

                scores_cur = model.predict_link(
                    entity_ids, edge_index, edge_type, h_idx, t_idx, r_idx
                )
                with torch.no_grad():
                    scores_ref = reference_model.predict_link(
                        entity_ids, edge_index, edge_type, h_idx, t_idx, r_idx
                    )

                forget_loss = _optimal_npo_loss_kg(
                    scores_cur, scores_ref, inf_weights=None, beta=beta
                )

                # ---- Retain loss ----
                retain_indices = retain_mask.nonzero(as_tuple=True)[0]
                retain_batch   = min(len(retain_indices), 256)
                retain_sample  = retain_indices[
                    torch.randperm(len(retain_indices))[:retain_batch]
                ]
                h_r = edge_index[0, retain_sample]
                t_r = edge_index[1, retain_sample]
                r_r = edge_type[retain_sample]
                scores_retain = model.predict_link(
                    entity_ids, edge_index, edge_type, h_r, t_r, r_r
                )
                neg_tail  = torch.randint(
                    0, data['num_entities'], (retain_batch,), device=device
                )
                neg_scores = model.predict_link(
                    entity_ids, edge_index, edge_type, h_r, neg_tail, r_r
                )
                retain_loss = F.margin_ranking_loss(
                    scores_retain, neg_scores,
                    torch.ones(retain_batch, device=device), margin=1.0
                )

            else:
                out = model(data.x, data.edge_index)

                # ---- Forget loss (optimal NPO, node classification) ----
                logits_cur = out[curriculum_mask]
                with torch.no_grad():
                    logits_ref = reference_model(
                        data.x, data.edge_index
                    )[curriculum_mask]

                p_cur = F.softmax(logits_cur / tau, dim=-1)
                p_ref = F.softmax(logits_ref / tau, dim=-1).detach()

                forget_loss = _optimal_npo_loss(
                    p_cur, p_ref, inf_weights=None, beta=beta
                )

                # ---- Retain loss ----
                retain_loss = F.cross_entropy(
                    out[retain_mask], data.y[retain_mask]
                )

            if getattr(args, 'npo_loss_mode', 'zero_sum') == 'decoupled':
                loss = forget_loss + lam * retain_loss
            else:
                loss = lam * forget_loss + (1 - lam) * retain_loss

            if torch.is_tensor(loss):
                loss.backward()
                optimizer.step()
                pbar.set_postfix({
                    'Forget': f'{forget_loss.item():.4f}',
                    'Retain': f'{retain_loss.item():.4f}'
                })

    print("Full Method (Optimal NPO) completed!")
    return model


def npo_ga(args, model, data, is_kg, forget_mask, retain_mask):
    """
    NPO without Curriculum (GA + p_ref-weighted NPO, default npo_ga).

    Same as npo_gradient_ascent but replaces the KL-based forget loss with
    the p_ref-weighted INPO ratio loss:

        L_forget = (2/β|B_f|) Σ_{v∈B_f} Σ_c p_ref(c;τ)·log(1+(p_v(c;τ)/p_ref(c;τ))^β)
    """
    set_unlearn_seed(getattr(args, 'seed', 42))

    print("\n" + "="*60)
    print("NPO Gradient Ascent (npo_ga)")
    print(f"Beta: {args.npo_beta}, Lambda: {args.npo_lambda}, Tau: {args.npo_temperature}")
    print("="*60)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not is_kg:
        data = data.to(device)

    # Frozen reference model
    reference_model = deepcopy(model)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    beta = args.npo_beta
    tau  = args.npo_temperature
    lam  = args.npo_lambda

    optimizer = optim.Adam(model.parameters(), lr=args.unlearn_lr)

    pbar = tqdm(range(args.unlearn_epochs), desc="Optimal NPO")
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        if is_kg:
            edge_index = data['train']['edge_index'].to(device)
            edge_type  = data['train']['edge_type'].to(device)
            entity_ids = data['entity_ids'].to(device)

            # ---- Forget loss (optimal NPO, KG) ----
            forget_indices = forget_mask.nonzero(as_tuple=True)[0]
            if len(forget_indices) > 0:
                batch_size  = min(len(forget_indices), 256)
                sample_idx  = forget_indices[
                    torch.randperm(len(forget_indices))[:batch_size]
                ]
                h_idx = edge_index[0, sample_idx]
                t_idx = edge_index[1, sample_idx]
                r_idx = edge_type[sample_idx]

                scores_cur = model.predict_link(
                    entity_ids, edge_index, edge_type, h_idx, t_idx, r_idx
                )
                with torch.no_grad():
                    scores_ref = reference_model.predict_link(
                        entity_ids, edge_index, edge_type, h_idx, t_idx, r_idx
                    )
                forget_loss = _optimal_npo_loss_kg(
                    scores_cur, scores_ref, inf_weights=None, beta=beta
                )
            else:
                forget_loss = torch.tensor(0.0, device=device)

            # ---- Retain loss ----
            retain_indices = retain_mask.nonzero(as_tuple=True)[0]
            if len(retain_indices) > 0:
                batch_size   = min(len(retain_indices), 256)
                retain_sample = retain_indices[
                    torch.randperm(len(retain_indices))[:batch_size]
                ]
                h_r = edge_index[0, retain_sample]
                t_r = edge_index[1, retain_sample]
                r_r = edge_type[retain_sample]
                scores_retain = model.predict_link(
                    entity_ids, edge_index, edge_type, h_r, t_r, r_r
                )
                neg_tail  = torch.randint(
                    0, data['num_entities'], (batch_size,), device=device
                )
                neg_scores = model.predict_link(
                    entity_ids, edge_index, edge_type, h_r, neg_tail, r_r
                )
                retain_loss = F.margin_ranking_loss(
                    scores_retain, neg_scores,
                    torch.ones(batch_size, device=device), margin=1.0
                )
            else:
                retain_loss = torch.tensor(0.0, device=device)

        else:
            out = model(data.x, data.edge_index)

            # ---- Forget loss (optimal NPO, node classification) ----
            if forget_mask.sum() > 0:
                logits_cur = out[forget_mask]
                with torch.no_grad():
                    logits_ref = reference_model(data.x, data.edge_index)[forget_mask]

                p_cur = F.softmax(logits_cur / tau, dim=-1)
                p_ref = F.softmax(logits_ref / tau, dim=-1).detach()
                forget_loss = _optimal_npo_loss(
                    p_cur, p_ref, inf_weights=None, beta=beta
                )
            else:
                forget_loss = torch.tensor(0.0, device=device)

            # ---- Retain loss ----
            if retain_mask.sum() > 0:
                retain_loss = F.cross_entropy(out[retain_mask], data.y[retain_mask])
            else:
                retain_loss = torch.tensor(0.0, device=device)

        if getattr(args, 'npo_loss_mode', 'zero_sum') == 'decoupled':
            loss = forget_loss + lam * retain_loss
        else:
            loss = lam * forget_loss + (1 - lam) * retain_loss

        if torch.is_tensor(loss) and loss.requires_grad:
            loss.backward()
            optimizer.step()
        pbar.set_postfix({
            'Forget': f'{forget_loss.item():.4f}',
            'Retain': f'{retain_loss.item():.4f}'
        })

    print("GA + Optimal NPO completed!")
    return model
