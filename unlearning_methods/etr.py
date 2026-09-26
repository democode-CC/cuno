"""
ETR: Erase then Rectify — Training-Free Parameter Editing for Graph Unlearning

Paper: https://arxiv.org/pdf/2409.16684 (AAAI 2025)
Original repo: https://github.com/AllminerLab/ETR

Two-stage, training-free algorithm:

  Stage 1 — Erase (Neighborhood-Aware Parameter Editing):
    Compute diagonal FIM for the full training set (F_D), forget set (F_Df),
    and k-hop retain-neighbors of the forget set (F_Dk).
    Scale down parameters that are disproportionately important to D_f:

      b1 = F_D / F_Df                    (Condition 1: forget-specific params)
      b2 = F_D² / (F_Df · F_Dk)         (Condition 2: neighborhood-affected)

    Parameters in the bottom erase_ratio quantile of b1 are scaled by b1/γ.
    Parameters in the bottom erase_ratio quantile of b2 (not in Cond.1)
    are scaled by b2/η.  All others are left unchanged.

  Stage 2 — Rectify (Gradient Approximation):
    Approximate ∇L_{D_r} without accessing D_r directly (Eq. 9):
      grad_r = (|D|·g_D − |D_f|·g_f − |D_k|·g_k + |D_k|·g_k') / |D_r|
    Apply one-step update: ω' = ω̂ − λ · grad_r

No iterative optimization — the entire unlearning is parameter arithmetic
plus two gradient evaluations.

Adaptation notes:
  - Homogeneous: full ETR (Erase + Rectify) on node subsets
  - KG: full ETR using edge-level subsets as proxy for D_f and D_k
"""

import torch
import torch.nn.functional as F
from collections import defaultdict

from utils import set_unlearn_seed


# ---------------------------------------------------------------------------
# Utilities: k-hop BFS and gradient / FIM computation
# ---------------------------------------------------------------------------

def _get_k_hop_neighbors(seed_nodes, edge_index, k, exclude=None):
    """
    BFS from seed_nodes for k hops; returns the set of reached nodes
    (excluding seed_nodes and any nodes in exclude).

    Args:
        seed_nodes : set[int]
        edge_index : LongTensor [2, E] (undirected)
        k          : int, number of hops
        exclude    : set[int] | None — additional nodes to strip from result

    Returns:
        set[int]
    """
    adj = defaultdict(set)
    for s, d in zip(edge_index[0].cpu().tolist(), edge_index[1].cpu().tolist()):
        adj[s].add(d)
        adj[d].add(s)

    visited  = set(seed_nodes)
    frontier = set(seed_nodes)
    for _ in range(k):
        nxt = set()
        for v in frontier:
            nxt |= adj[v] - visited
        visited  |= nxt
        frontier  = nxt

    result = visited - set(seed_nodes)
    if exclude:
        result -= exclude
    return result


def _grad_and_fim_homo(model, data, node_list, device, per_sample_fim=False):
    """
    Forward + backward on a subset of training nodes (homogeneous).
    Returns (grads, fim, n).

    grads[name] = gradient of the average loss over node_list  (for Rectify)
    fim[name]   = diagonal FIM proxy  (for Erase)

    When per_sample_fim=True (recommended for small forget sets):
      fim[name] = (1/n) Σ_i (∂l_i/∂ω)²   — proper per-sample estimate
    When per_sample_fim=False (fast, for large sets):
      fim[name] = (∂L_avg/∂ω)²             — batch proxy (may underestimate
                                               due to gradient cancellation)
    """
    zero = lambda: {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    if not node_list:
        return zero(), zero(), 0

    # ── batch gradient (used for Rectify) ────────────────────────────────
    model.zero_grad()
    idx  = torch.tensor(node_list, dtype=torch.long, device=device)
    out  = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[idx], data.y[idx])
    loss.backward()

    grads = {}
    for name, p in model.named_parameters():
        grads[name] = p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
    model.zero_grad()

    # ── FIM diagonal ─────────────────────────────────────────────────────
    if per_sample_fim:
        # Accumulate squared per-sample gradients: proper FIM approximation.
        # Feasible when node_list is small (forget set).
        fim = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        # Cache full-graph output to avoid repeated message-passing
        with torch.no_grad():
            out_all = model(data.x, data.edge_index)
        for node_idx in node_list:
            model.zero_grad()
            idx_i = torch.tensor([node_idx], dtype=torch.long, device=device)
            # Re-use cached output but need grad through it: recompute for this node
            out_i = model(data.x, data.edge_index)
            loss_i = F.cross_entropy(out_i[idx_i], data.y[idx_i])
            loss_i.backward()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    fim[name] += p.grad.detach().pow(2)
        model.zero_grad()
        n = len(node_list)
        for name in fim:
            fim[name] /= n
    else:
        fim = {name: grads[name].pow(2) for name in grads}

    return grads, fim, len(node_list)


def _grad_and_fim_kg(model, data, edge_list, device, per_sample_fim=False):
    """
    Forward + backward on a subset of KG training edges.
    Returns (grads, fim, n).
    per_sample_fim=True uses proper per-edge FIM (recommended for forget set).
    """
    zero = lambda: {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    if not edge_list:
        return zero(), zero(), 0

    ei         = data['train']['edge_index'].to(device)
    et         = data['train']['edge_type'].to(device)
    entity_ids = data['entity_ids'].to(device)

    idx   = torch.tensor(edge_list, dtype=torch.long, device=device)
    h_idx = ei[0, idx]
    t_idx = ei[1, idx]
    r_idx = et[idx]

    # batch gradient (for Rectify)
    model.zero_grad()
    scores = model.predict_link(entity_ids, ei, et, h_idx, t_idx, r_idx)
    loss   = -scores.mean()
    loss.backward()

    grads = {}
    for name, p in model.named_parameters():
        grads[name] = p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
    model.zero_grad()

    # FIM diagonal
    if per_sample_fim:
        fim = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        for i in range(len(edge_list)):
            model.zero_grad()
            s = model.predict_link(
                entity_ids, ei, et,
                h_idx[i:i+1], t_idx[i:i+1], r_idx[i:i+1]
            )
            (-s).backward()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    fim[name] += p.grad.detach().pow(2)
        model.zero_grad()
        n = len(edge_list)
        for name in fim:
            fim[name] /= n
    else:
        fim = {name: grads[name].pow(2) for name in grads}
    model.zero_grad()
    return grads, fim, len(edge_list)


# ---------------------------------------------------------------------------
# Stage 1: Erase
# ---------------------------------------------------------------------------

def _erase(model, fim_D, fim_Df, fim_Dk, erase_ratio, erase_strength=0.0):
    """
    Edit parameters in-place using FIM ratios (paper Eq. 5).

    b1 = F_D / F_Df  — small → parameter is disproportionately forget-specific
    b2 = F_D² / (F_Df·F_Dk) — small → parameter is neighborhood-affected

    Bottom erase_ratio quantile of b1 → scale down [Condition 1]
    Bottom erase_ratio quantile of b2 (excluding Cond.1) → scale down [Condition 2]

    erase_strength: 0 = soft scaling only (p *= b/γ); 1 = full zero; in (0,1) = blend.
    Scale = (1 - erase_strength) * (b/γ), so 0.5 gives a milder shrink toward zero.
    """
    eps = 1e-8
    coef = 1.0 - erase_strength  # 0 → scale as-is; 1 → scale 0 (full zero)
    with torch.no_grad():
        for name, p in model.named_parameters():
            f_D  = fim_D.get(name,  torch.zeros_like(p)).clamp(min=eps)
            f_Df = fim_Df.get(name, torch.zeros_like(p)).clamp(min=eps)
            f_Dk = fim_Dk.get(name, torch.zeros_like(p)).clamp(min=eps)

            b1 = (f_D / f_Df).clamp(max=1.0)
            b2 = (f_D.pow(2) / (f_Df * f_Dk)).clamp(max=1.0)

            gamma = torch.quantile(b1.flatten().float(), erase_ratio)
            eta   = torch.quantile(b2.flatten().float(), erase_ratio)

            loc1 = b1 <= gamma
            loc2 = (b2 <= eta) & ~loc1

            # Scale: (1 - erase_strength) * (b/γ); erase_strength=0 → original, 1 → zero
            if loc1.any():
                scale1 = coef * (b1[loc1] / gamma.clamp(min=eps))
                p[loc1] = p[loc1] * scale1
            if loc2.any():
                scale2 = coef * (b2[loc2] / eta.clamp(min=eps))
                p[loc2] = p[loc2] * scale2


# ---------------------------------------------------------------------------
# Stage 2: Rectify
# ---------------------------------------------------------------------------

def _rectify(model, g_D, g_Df, g_Dk, g_Dk_new, n_D, n_Df, n_Dk, lam):
    """
    One-step gradient update using the approximate retain-set gradient (Eq. 9–10).

    grad_r = ( |D|·g_D − |D_f|·g_f − |D_k|·g_k + |D_k|·g_k' ) / |D_r|
    ω'     = ω̂ − λ · grad_r
    """
    n_r = n_D - n_Df
    if n_r <= 0:
        print("  [Rectify] |D_r| = 0, skipping.")
        return

    with torch.no_grad():
        for name, p in model.named_parameters():
            g_d   = g_D.get(name,      torch.zeros_like(p))
            g_f   = g_Df.get(name,     torch.zeros_like(p))
            g_k   = g_Dk.get(name,     torch.zeros_like(p))
            g_k_n = g_Dk_new.get(name, torch.zeros_like(p))

            grad_r = (n_D * g_d - n_Df * g_f - n_Dk * g_k + n_Dk * g_k_n) / n_r
            p.sub_(lam * grad_r)


# ---------------------------------------------------------------------------
# Homogeneous: full ETR
# ---------------------------------------------------------------------------

def _etr_homo(model, data, forget_mask, retain_mask,
              erase_ratio, erase_strength, lam, k_hops, device):
    forget_nodes = set(forget_mask.nonzero(as_tuple=True)[0].cpu().tolist())
    retain_nodes = set(retain_mask.nonzero(as_tuple=True)[0].cpu().tolist())
    all_train    = list(forget_nodes | retain_nodes)
    n_D          = len(all_train)

    # k-hop retain neighbors of the forget set
    print("  Computing k-hop neighborhood...")
    k_hop_nodes = (
        _get_k_hop_neighbors(forget_nodes, data.edge_index, k_hops,
                             exclude=forget_nodes)
        & retain_nodes
    )
    print(f"  |D|={n_D}, |D_f|={len(forget_nodes)}, |D_k|={len(k_hop_nodes)}")

    # Compute gradients and FIM under the original model.
    # Use per-sample FIM for D_f (small set, critical for discrimination).
    # Use fast batch-proxy FIM for D and D_k (large sets).
    print("  Computing FIM (original model)...")
    g_D,  fim_D,  _    = _grad_and_fim_homo(model, data, all_train,          device, per_sample_fim=False)
    g_Df, fim_Df, n_Df = _grad_and_fim_homo(model, data, list(forget_nodes), device, per_sample_fim=True)
    g_Dk, fim_Dk, n_Dk = _grad_and_fim_homo(model, data, list(k_hop_nodes),  device, per_sample_fim=False)

    # Stage 1: Erase
    print("  Stage 1: Erase...")
    _erase(model, fim_D, fim_Df, fim_Dk, erase_ratio, erase_strength)

    # Stage 2: Rectify — re-compute g_Dk under edited model ω̂
    print("  Stage 2: Rectify...")
    g_Dk_new, _, _ = _grad_and_fim_homo(model, data, list(k_hop_nodes), device)
    _rectify(model, g_D, g_Df, g_Dk, g_Dk_new, n_D, n_Df, n_Dk, lam)


# ---------------------------------------------------------------------------
# KG: ETR on edge-level subsets
# ---------------------------------------------------------------------------

def _etr_kg(model, data, forget_mask, retain_mask,
            erase_ratio, erase_strength, lam, k_hops, device):
    ei           = data['train']['edge_index'].to(device)
    forget_edges = set(forget_mask.nonzero(as_tuple=True)[0].cpu().tolist())
    retain_edges = set(retain_mask.nonzero(as_tuple=True)[0].cpu().tolist())
    all_edges    = list(range(forget_mask.numel()))
    n_D          = len(all_edges)

    # D_f entities = endpoints of forget edges
    forget_ents = set(
        ei[0, list(forget_edges)].cpu().tolist() +
        ei[1, list(forget_edges)].cpu().tolist()
    )

    # D_k entities = k-hop entity neighbors of D_f in the retain-edge subgraph
    print("  Computing k-hop entity neighborhood (KG)...")
    k_hop_ents = _get_k_hop_neighbors(forget_ents, ei.cpu(), k_hops,
                                       exclude=forget_ents)

    # Map D_k entities → retain edge indices (edges whose endpoints are in D_k)
    h_all = ei[0].cpu()
    t_all = ei[1].cpu()
    k_hop_edges = {
        idx for idx in retain_edges
        if h_all[idx].item() in k_hop_ents or t_all[idx].item() in k_hop_ents
    }
    print(f"  |D| edges={n_D}, |D_f| edges={len(forget_edges)}, |D_k| edges={len(k_hop_edges)}")

    # Compute gradients and FIM
    print("  Computing FIM (original model, KG)...")
    g_D,  fim_D,  _    = _grad_and_fim_kg(model, data, all_edges,           device, per_sample_fim=False)
    g_Df, fim_Df, n_Df = _grad_and_fim_kg(model, data, list(forget_edges),  device, per_sample_fim=True)
    g_Dk, fim_Dk, n_Dk = _grad_and_fim_kg(model, data, list(k_hop_edges),   device, per_sample_fim=False)

    # Stage 1: Erase
    print("  Stage 1: Erase (KG)...")
    _erase(model, fim_D, fim_Df, fim_Dk, erase_ratio, erase_strength)

    # Stage 2: Rectify
    print("  Stage 2: Rectify (KG)...")
    g_Dk_new, _, _ = _grad_and_fim_kg(model, data, list(k_hop_edges), device)
    _rectify(model, g_D, g_Df, g_Dk, g_Dk_new, n_D, n_Df, n_Dk, lam)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def etr(args, model, data, is_kg, forget_mask, retain_mask):
    """
    ETR: Erase then Rectify for graph unlearning.

    Args:
        args:        Namespace with etr_erase_ratio, etr_lambda,
                     num_layers, device, seed
        model:       Trained GNN model (modified in-place)
        data:        Graph data (PyG Data for homogeneous, dict for KG)
        is_kg:       True for knowledge graphs
        forget_mask: Boolean mask over nodes (homo) or edges (KG)
        retain_mask: Boolean mask over nodes (homo) or edges (KG)

    Returns:
        model: Unlearned model
    """
    set_unlearn_seed(getattr(args, 'seed', 42))

    erase_ratio     = getattr(args, 'etr_erase_ratio', 0.01)
    erase_strength  = getattr(args, 'etr_erase_strength', 0.0)
    lam             = getattr(args, 'etr_lambda', 0.3)
    k_hops          = getattr(args, 'num_layers', 2)

    print("\n" + "=" * 60)
    print("ETR: Erase then Rectify")
    print(f"  erase_ratio={erase_ratio}, erase_strength={erase_strength}, lambda={lam}, k_hops={k_hops}")
    print("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model  = model.to(device)

    # Enable gradients on all params for FIM computation
    was_training    = model.training
    grad_states     = {p: p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()

    with torch.enable_grad():
        if is_kg:
            _etr_kg(model, data, forget_mask, retain_mask,
                    erase_ratio, erase_strength, lam, k_hops, device)
        else:
            data        = data.to(device)
            forget_mask = forget_mask.to(device)
            retain_mask = retain_mask.to(device)
            _etr_homo(model, data, forget_mask, retain_mask,
                      erase_ratio, erase_strength, lam, k_hops, device)

    # Restore original gradient state and training mode
    for p, state in grad_states.items():
        p.requires_grad_(state)
    if not was_training:
        model.eval()

    print("ETR completed!")
    return model
