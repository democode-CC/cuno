"""
INPO: Influence-aware Negative Preference Optimization for Graph Unlearning

Simplified implementation adapted for our codebase (node classification + KG link prediction).

Paper: https://dl.acm.org/doi/pdf/10.1145/3746027.3754941
Original repo: https://github.com/sh-qiangchen/INPO

Three loss components:
  1. NPO loss (influence-weighted): push forget-set predictions away from reference model
  2. Retain CE loss: preserve model utility on retain set
  3. TE (local structure alignment): keep forget-node embeddings aligned with reference

Simplified NORA (no DGL required):
  Approximates per-node removal influence via input-gradient norm × degree normalization.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from copy import deepcopy

from utils import set_unlearn_seed


# ---------------------------------------------------------------------------
# Simplified NORA: Node Removal Influence Analysis (PyG-native, no DGL)
# ---------------------------------------------------------------------------

def compute_influence_scores_homo(model, data, device):
    """
    Approximate per-node influence scores for homogeneous graphs.

    Based on NORA (Node Removal Influence Analysis), simplified to:
      influence[v] = ||∂L/∂x_v||_2 · deg(v) / (deg(v) + self_buff)

    where the gradient is computed via a Grad-CAM-style backward pass on the
    original model output.

    Returns:
        influence: Tensor [num_nodes] in [0, 1]
    """
    model.eval()
    x = data.x.detach().clone().requires_grad_(True)
    edge_index = data.edge_index

    out = model(x, edge_index)
    out_soft = F.softmax(out.detach(), dim=1)
    # Backward: gradient proportional to output (Grad-CAM style)
    out.backward(gradient=out_soft)

    grad_norm = x.grad.detach().norm(p=2, dim=1)   # [N]

    # Degree of each node
    num_nodes = x.size(0)
    src, dst = edge_index[0], edge_index[1]
    degree = torch.zeros(num_nodes, device=device)
    degree.scatter_add_(0, src, torch.ones(src.size(0), device=device))
    degree.scatter_add_(0, dst, torch.ones(dst.size(0), device=device))
    degree = degree.clamp(min=1.0)

    self_buff = 8.0  # from original NORA
    influence = grad_norm * degree / (degree + self_buff)

    # Min-max normalize to [0, 1]
    inf_min, inf_max = influence.min(), influence.max()
    if inf_max > inf_min:
        influence = (influence - inf_min) / (inf_max - inf_min)
    else:
        influence = torch.ones_like(influence)

    model.train()
    return influence.detach()


def compute_influence_scores_kg(data, device):
    """
    Degree-based influence scores for knowledge graph entities.

    Since KG models require more complex gradient routing, we use entity
    degree as a structural proxy for removal influence.

    Returns:
        influence: Tensor [num_entities] in [0, 1]
    """
    num_entities = data['num_entities']
    edge_index = data['train']['edge_index']

    degree = torch.zeros(num_entities, device=device)
    src = edge_index[0].to(device)
    dst = edge_index[1].to(device)
    degree.scatter_add_(0, src, torch.ones(src.size(0), device=device))
    degree.scatter_add_(0, dst, torch.ones(dst.size(0), device=device))

    deg_max = degree.max()
    influence = degree / (deg_max + 1e-8)
    return influence.detach()


# ---------------------------------------------------------------------------
# Per-step loss helpers
# ---------------------------------------------------------------------------

def _npo_loss(p_cur, p_ref, inf_weights, beta):
    """
    INPO/NPO loss: 2/β · mean(inf_w · log(1 + (p_θ / p_ref)^β))

    Computed in log-space to avoid overflow when beta is large:
      ratio^beta = exp(beta * log(ratio))

    Args:
        p_cur:       [N, C] or [N] current model probabilities
        p_ref:       [N, C] or [N] reference model probabilities (detached)
        inf_weights: [N] or [N, 1] per-sample influence scores
        beta:        float, temperature parameter

    Returns:
        scalar loss
    """
    # Log-space computation: beta * log(p_cur/p_ref), clamped to prevent overflow.
    # Clamp log_ratio so that beta*log_ratio ∈ [-20, 20], keeping exp() in [~2e-9, ~5e8].
    # Using a fixed clamp of ±30 with beta=10 gives exp(300) = inf → NaN cascade.
    log_ratio = torch.log(p_cur + 1e-8) - torch.log(p_ref + 1e-8)
    safe_limit = 20.0 / max(float(beta), 1.0)
    log_ratio = log_ratio.clamp(min=-safe_limit, max=safe_limit)
    ratio_pow_beta = torch.exp(beta * log_ratio)
    per_node = torch.log(1.0 + ratio_pow_beta)
    if per_node.dim() > 1:
        per_node = per_node.sum(dim=-1)           # [N]
    if inf_weights is not None:
        per_node = per_node * inf_weights.squeeze(-1)
    return (2.0 / beta) * per_node.mean()


def _homo_step(model, reference_model, data, forget_mask, retain_mask,
               influence_scores, ref_embeds, beta, npo_lambda, te_weight, device):
    """One training step for homogeneous node classification."""
    out = model(data.x, data.edge_index)

    # ---- 1. NPO loss on forget nodes (influence-weighted) ----
    if forget_mask.sum() > 0:
        logits_cur = out[forget_mask]
        with torch.no_grad():
            logits_ref = reference_model(data.x, data.edge_index)[forget_mask]

        p_cur = torch.softmax(logits_cur, dim=-1)
        p_ref = torch.softmax(logits_ref, dim=-1)
        inf_w = influence_scores[forget_mask].unsqueeze(-1)   # [N_f, 1]
        loss_npo = _npo_loss(p_cur, p_ref, inf_w, beta)
    else:
        loss_npo = torch.tensor(0.0, device=device)

    # ---- 2. Retain CE loss ----
    if retain_mask.sum() > 0:
        retain_loss = F.cross_entropy(out[retain_mask], data.y[retain_mask])
    else:
        retain_loss = torch.tensor(0.0, device=device)

    # ---- 3. Local structure alignment (TE) ----
    # Cross-entropy between reference and current embeddings on forget nodes.
    # Raw hidden vectors have arbitrary scale → L2-normalize first so that
    # softmax inputs are bounded in [-1, 1], preventing log_softmax → -inf → NaN.
    if ref_embeds is not None and forget_mask.sum() > 0:
        cur_embeds = model.get_embeddings(data.x, data.edge_index)
        ref_norm = F.normalize(ref_embeds[forget_mask], p=2, dim=-1).detach()
        cur_norm = F.normalize(cur_embeds[forget_mask], p=2, dim=-1)
        p_ref_emb    = F.softmax(ref_norm, dim=-1)
        log_p_cur_emb = F.log_softmax(cur_norm, dim=-1)
        te_loss = -torch.mean(torch.sum(p_ref_emb * log_p_cur_emb, dim=-1))
    else:
        te_loss = torch.tensor(0.0, device=device)

    return loss_npo + npo_lambda * retain_loss + te_weight * te_loss, loss_npo, retain_loss, te_loss


def _kg_step(model, reference_model, data, forget_mask, retain_mask,
             influence_scores, beta, npo_lambda, device):
    """One training step for knowledge graph link prediction."""
    edge_index = data['train']['edge_index'].to(device)
    edge_type  = data['train']['edge_type'].to(device)
    entity_ids = data['entity_ids'].to(device)

    # ---- 1. NPO loss on forget edges (influence-weighted) ----
    forget_indices = forget_mask.nonzero(as_tuple=True)[0]
    if len(forget_indices) > 0:
        batch_size = min(len(forget_indices), 512)
        sample_idx = forget_indices[torch.randperm(len(forget_indices))[:batch_size]]

        h_idx = edge_index[0, sample_idx]
        t_idx = edge_index[1, sample_idx]
        r_idx = edge_type[sample_idx]

        scores_cur = model.predict_link(
            entity_ids, edge_index, edge_type, h_idx, t_idx, r_idx
        ).sigmoid()
        with torch.no_grad():
            scores_ref = reference_model.predict_link(
                entity_ids, edge_index, edge_type, h_idx, t_idx, r_idx
            ).sigmoid()

        # Edge influence = avg of head and tail influence
        inf_w = (influence_scores[h_idx] + influence_scores[t_idx]) / 2.0
        loss_npo = _npo_loss(scores_cur, scores_ref, inf_w, beta)
    else:
        loss_npo = torch.tensor(0.0, device=device)

    # ---- 2. Retain margin ranking loss ----
    retain_indices = retain_mask.nonzero(as_tuple=True)[0]
    if len(retain_indices) > 0:
        batch_size = min(len(retain_indices), 512)
        retain_sample = retain_indices[torch.randperm(len(retain_indices))[:batch_size]]

        h_idx = edge_index[0, retain_sample]
        t_idx = edge_index[1, retain_sample]
        r_idx = edge_type[retain_sample]

        scores_cur = model.predict_link(entity_ids, edge_index, edge_type, h_idx, t_idx, r_idx)
        neg_tail   = torch.randint(0, data['num_entities'], (batch_size,), device=device)
        neg_scores = model.predict_link(entity_ids, edge_index, edge_type, h_idx, neg_tail, r_idx)

        retain_loss = F.margin_ranking_loss(
            scores_cur, neg_scores, torch.ones(batch_size, device=device), margin=1.0
        )
    else:
        retain_loss = torch.tensor(0.0, device=device)

    total = loss_npo + npo_lambda * retain_loss
    return total, loss_npo, retain_loss


# ---------------------------------------------------------------------------
# Main INPO entry point
# ---------------------------------------------------------------------------

def inpo(args, model, data, is_kg, forget_mask, retain_mask):
    """
    INPO: Influence-aware Negative Preference Optimization for Graph Unlearning.

    Algorithm:
      1. Clone a frozen reference model.
      2. Compute per-node/edge influence scores via simplified NORA.
      3. Iterate:
         a. NPO loss (influence-weighted) on forget set  →  unlearn
         b. CE / margin-ranking loss on retain set       →  preserve utility
         c. TE alignment loss on forget embeddings       →  structural stability

    Args:
        args:         Namespace with npo_beta, npo_lambda, inpo_te_weight,
                      unlearn_lr, unlearn_epochs, device, seed
        model:        Trained GNN model (modified in-place)
        data:         Graph data (PyG Data or KG dict)
        is_kg:        True for knowledge graphs
        forget_mask:  Boolean mask [num_nodes/edges] selecting the forget set
        retain_mask:  Boolean mask [num_nodes/edges] selecting the retain set

    Returns:
        model: Unlearned model
    """
    set_unlearn_seed(getattr(args, 'seed', 42))

    beta      = 15
    lam       = 0.00001
    te_weight = 0.1

    print("\n" + "=" * 60)
    print("INPO: Influence-aware Negative Preference Optimization")
    print(f"  beta={beta}, lambda={lam}, te_weight={te_weight}")
    print("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model  = model.to(device)

    if not is_kg:
        data        = data.to(device)
        forget_mask = forget_mask.to(device)
        retain_mask = retain_mask.to(device)

    # --- Step 1: freeze reference model ---
    reference_model = deepcopy(model)
    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad = False

    # --- Step 2: compute influence scores ---
    print("Computing influence scores (simplified NORA)...")
    try:
        if is_kg:
            influence_scores = compute_influence_scores_kg(data, device)
        else:
            influence_scores = compute_influence_scores_homo(model, data, device)
        print(f"  range: [{influence_scores.min():.4f}, {influence_scores.max():.4f}]")
    except Exception as e:
        print(f"  Warning: influence computation failed ({e}). Using uniform scores.")
        n = data['num_entities'] if is_kg else data.x.size(0)
        influence_scores = torch.ones(n, device=device)

    # --- Step 3: cache reference embeddings for TE loss (homogeneous only) ---
    ref_embeds = None
    if not is_kg:
        with torch.no_grad():
            ref_embeds = reference_model.get_embeddings(data.x, data.edge_index).detach()

    # --- Step 4: unlearning loop ---
    optimizer = optim.Adam(model.parameters(), lr=args.unlearn_lr)

    pbar = tqdm(range(args.unlearn_epochs), desc="INPO")
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        if is_kg:
            loss, loss_npo, retain_loss = _kg_step(
                model, reference_model, data,
                forget_mask, retain_mask,
                influence_scores, beta, lam, device
            )
            pbar.set_postfix({
                'NPO': f'{loss_npo.item():.4f}',
                'Retain': f'{retain_loss.item():.4f}',
            })
        else:
            loss, loss_npo, retain_loss, te_loss = _homo_step(
                model, reference_model, data,
                forget_mask, retain_mask,
                influence_scores, ref_embeds, beta, lam, te_weight, device
            )
            pbar.set_postfix({
                'NPO': f'{loss_npo.item():.4f}',
                'Retain': f'{retain_loss.item():.4f}',
                'TE': f'{te_loss.item():.4f}',
            })

        if loss.requires_grad and not torch.isnan(loss):
            loss.backward()
            # Gradient clipping: prevents parameter explosion that causes NaN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        elif torch.isnan(loss):
            print(f"\n  [Warning] NaN loss at epoch {epoch}, skipping update.")

    print("INPO completed!")
    return model
