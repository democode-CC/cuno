"""
GNNDelete: A General Unlearning Strategy for Graph Neural Networks
Based on: "GNNDelete: A General Unlearning Strategy for Graph Neural Networks" (ICLR 2023)
Paper: https://openreview.net/forum?id=X9yCkmT5Qrl
Code: https://github.com/mims-harvard/GNNDelete

Two core objectives (paper):
1. Deleted Edge Consistency: Remove influence of deleted elements from model and neighbors.
   → deletion(base(original))[affected] ≈ base(modified)[affected]
2. Neighborhood Influence: Preserve remaining model knowledge.
   → Retain set predictions kept correct (CE on retain).

Only the deletion operator is trained; the base GNN is frozen.
Simplified: single output-layer deletion (linear map on logits), no identity/entropy extras.
"""

# Bump this when you change implementation/loss so cache is invalidated and new runs are used.
GNNDELETE_IMPL_VERSION = 4

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from utils import set_unlearn_seed


def _get_affected_nodes_homogeneous(data, forget_mask, k_hops=2):
    """S_Df: k-hop neighborhood of forget set (paper)."""
    edge_index = data.edge_index.cpu().numpy()
    forget_nodes = forget_mask.cpu().numpy().nonzero()[0]
    affected = set(forget_nodes)
    for _ in range(k_hops):
        source_mask = np.isin(edge_index[0], list(affected))
        neighbors = edge_index[1, source_mask]
        affected.update(neighbors)
    mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=forget_mask.device)
    mask[list(affected)] = True
    return mask


def _get_modified_edge_index_homogeneous(data, forget_mask, device):
    """Graph with edges incident to forget set removed (Df deleted)."""
    ei = data.edge_index.to(device)
    keep = ~(forget_mask[ei[0]] | forget_mask[ei[1]])
    return ei[:, keep]


class DeletionOperator(nn.Module):
    """Single linear map on logits (paper: layer-wise matrix deletion). Simplified to one layer at output."""

    def __init__(self, num_classes, init_near_identity=True):
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(num_classes, num_classes, bias=False)
        if init_near_identity:
            with torch.no_grad():
                # Small perturbation from identity so gradients can flow immediately
                noise = torch.randn(num_classes, num_classes) * 0.01
                self.linear.weight.copy_(torch.eye(num_classes) + noise)

    def forward(self, x):
        return self.linear(x)


class GNNDeleteWrappedModel(nn.Module):
    """Wrapper: base GNN + deletion applied only to affected nodes."""

    def __init__(self, base_model, deletion, affected_mask):
        super().__init__()
        self.base_model = base_model
        self.deletion = deletion
        self.affected_mask = affected_mask

    def forward(self, x, edge_index):
        logits_orig = self.base_model(x, edge_index)
        if not self.affected_mask.any():
            return logits_orig
        logits_affected = self.deletion(logits_orig[self.affected_mask])
        out = logits_orig.clone()
        out[self.affected_mask] = logits_affected
        return out


def _run_gnndelete_homogeneous(args, model, data, forget_mask, retain_mask, device):
    """GNNDelete for homogeneous graphs: freeze base, train deletion with Consistency + Retain only."""
    num_classes = int(data.y.max().item()) + 1
    if hasattr(data, 'num_classes'):
        num_classes = getattr(data, 'num_classes', num_classes)

    affected_mask = _get_affected_nodes_homogeneous(data, forget_mask, k_hops=2)
    affected_mask = affected_mask.to(device)
    modified_edge_index = _get_modified_edge_index_homogeneous(data, forget_mask, device)

    for p in model.parameters():
        p.requires_grad = False

    deletion = DeletionOperator(num_classes).to(device)
    wrapper = GNNDeleteWrappedModel(model, deletion, affected_mask).to(device)

    optimizer = optim.Adam(deletion.parameters(), lr=getattr(args, 'gnndelete_lr', 2e-2))
    alpha = getattr(args, 'gnndelete_alpha', 0.1)      # retain (Neighborhood Influence)
    gamma = getattr(args, 'gnndelete_consistency_weight', 4.0)  # Deleted Edge Consistency
    forget_weight = getattr(args, 'gnndelete_forget_weight', 3.0)  # extra push on forget nodes
    epochs = getattr(args, 'gnndelete_epochs', None) or max(100, getattr(args, 'unlearn_epochs', 100))

    model.eval()
    pbar = tqdm(range(epochs), desc="GNNDelete")
    for _ in pbar:
        deletion.train()
        optimizer.zero_grad()

        logits_orig = wrapper.base_model(data.x, data.edge_index)
        logits_affected = deletion(logits_orig[affected_mask])
        with torch.no_grad():
            logits_ref = wrapper.base_model(data.x, modified_edge_index)
        ref_affected = logits_ref[affected_mask].detach()

        # Precompute index mappings (global node id → local index in logits_affected)
        affected_idx = affected_mask.nonzero(as_tuple=True)[0]
        forget_in_affected = forget_mask & affected_mask
        retain_in_affected = retain_mask & affected_mask
        retain_not_affected = retain_mask & ~affected_mask

        def global_to_local(global_idx):
            return (affected_idx.unsqueeze(0) == global_idx.unsqueeze(1)).nonzero()[:, 1]

        # 1) Gradient ascent on forget nodes: push predictions AWAY from correct labels
        #    No consistency loss here — base(modified)[forget] is still confident (features intact),
        #    so consistency would conflict with forgetting.
        forget_loss = torch.tensor(0.0, device=device)
        if forget_in_affected.sum() > 0:
            forget_global = forget_in_affected.nonzero(as_tuple=True)[0]
            local_forget = global_to_local(forget_global)
            forget_logits = logits_affected[local_forget]
            forget_loss = -F.cross_entropy(forget_logits, data.y[forget_in_affected])

        # 2) Consistency ONLY for retain nodes in affected area (no conflict here)
        consistency_loss = torch.tensor(0.0, device=device)
        if retain_in_affected.sum() > 0:
            retain_aff_global = retain_in_affected.nonzero(as_tuple=True)[0]
            local_retain_aff = global_to_local(retain_aff_global)
            consistency_loss = F.mse_loss(
                logits_affected[local_retain_aff], ref_affected[local_retain_aff]
            )

        # 3) Retain CE: preserve correct predictions on retain nodes in affected area
        retain_loss = torch.tensor(0.0, device=device)
        if retain_in_affected.sum() > 0:
            retain_aff_global = retain_in_affected.nonzero(as_tuple=True)[0]
            local_retain_aff = global_to_local(retain_aff_global)
            retain_loss = retain_loss + F.cross_entropy(
                logits_affected[local_retain_aff], data.y[retain_in_affected]
            )

        # 4) Gradient ascent on non-affected retain nodes: push AWAY from correct labels
        #    Extends forgetting pressure beyond the affected neighborhood.
        ga_retain_loss = torch.tensor(0.0, device=device)
        if retain_not_affected.sum() > 0:
            ga_retain_loss = -F.cross_entropy(
                logits_orig[retain_not_affected], data.y[retain_not_affected]
            )

        ga_retain_weight = getattr(args, 'gnndelete_ga_retain_weight', 1.0)
        loss = forget_weight * forget_loss + gamma * consistency_loss + alpha * retain_loss + ga_retain_weight * ga_retain_loss
        loss.backward()
        optimizer.step()
        pbar.set_postfix({'GA': f'{forget_loss.item():.4f}', 'Cons': f'{consistency_loss.item():.4f}', 'Ret': f'{retain_loss.item():.4f}', 'GA_ret': f'{ga_retain_loss.item():.4f}'})

    return wrapper


def load_gnndelete_wrapper(checkpoint, args, data, is_kg, forget_mask, device):
    """Load from checkpoint: only DeletionOperator format (deletion.linear.weight)."""
    from learning import create_model
    state = checkpoint['model_state_dict']
    if 'deletion.linear.weight' not in state:
        raise ValueError(
            "GNNDelete checkpoint is legacy format (no deletion.linear.weight). "
            "Re-run unlearning to save in current format, or delete the old cache file."
        )
    num_classes = state['deletion.linear.weight'].shape[0]
    base_model = create_model(args, data, is_kg)
    deletion = DeletionOperator(num_classes, init_near_identity=False)
    affected_mask = _get_affected_nodes_homogeneous(data, forget_mask, k_hops=2).to(device)
    wrapper = GNNDeleteWrappedModel(base_model, deletion, affected_mask)
    wrapper.load_state_dict(state, strict=True)
    return wrapper.to(device)


def _run_gnndelete_kg(args, model, data, forget_mask, retain_mask, device):
    """KG: fallback to gradient ascent."""
    from unlearning_methods.gradient_ascent import gradient_ascent_baseline
    return gradient_ascent_baseline(args, model, data, True, forget_mask, retain_mask)


def gnndelete_unlearn(args, model, data, is_kg, forget_mask, retain_mask):
    """GNNDelete unlearning: Deleted Edge Consistency + Neighborhood Influence."""
    set_unlearn_seed(getattr(args, 'seed', 42))
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not is_kg:
        data = data.to(device)
        forget_mask = forget_mask.to(device)
        retain_mask = retain_mask.to(device)
        model = _run_gnndelete_homogeneous(args, model, data, forget_mask, retain_mask, device)
    else:
        forget_mask = forget_mask.to(device)
        retain_mask = retain_mask.to(device)
        model = _run_gnndelete_kg(args, model, data, forget_mask, retain_mask, device)
    return model
