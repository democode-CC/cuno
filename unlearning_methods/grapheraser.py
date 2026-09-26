"""
GraphEraser: Graph Erasing Method
Based on sharded retraining: partition graph with constrained K-means on embeddings,
train one model per shard on modified graph (edges incident to forget set removed),
aggregate by cluster assignment (each node predicted by its shard).

Reference: Graph Unlearning (e.g. MinChen00/Graph-Unlearning style partitioning).
Adapted from SUMMIT-TIFS/framework/trainer/graph_eraser.py for node classification.
"""

import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from tqdm import tqdm, trange

from learning import create_model, train_homogeneous
from utils import set_unlearn_seed


# ---------- Constrained K-means (from Graph-Unlearning / SUMMIT) ----------
class ConstrainedKmeans:
    """
    Constrained K-means for graph partitioning: balance cluster sizes with node_threshold.
    """

    def __init__(self, data_feat, num_clusters, node_threshold, terminate_delta, max_iteration=20):
        self.data_feat = data_feat
        self.num_clusters = num_clusters
        self.node_threshold = node_threshold
        self.terminate_delta = terminate_delta
        self.max_iteration = max_iteration

    def initialization(self):
        n = self.data_feat.shape[0]
        k = min(self.num_clusters, n)
        centroids = np.random.choice(np.arange(n), size=k, replace=False)
        self.centroid = {}
        for i in range(k):
            self.centroid[i] = self.data_feat[centroids[i]].copy()
        for i in range(k, self.num_clusters):
            self.centroid[i] = self.data_feat[centroids[0]].copy()

    def clustering(self):
        centroid = copy.deepcopy(self.centroid)
        km_delta = []
        for i in trange(self.max_iteration, desc='Graph partition'):
            self._node_reassignment()
            self._centroid_updating()
            delta = self._centroid_delta(centroid, self.centroid)
            km_delta.append(delta)
            centroid = copy.deepcopy(self.centroid)
            if delta <= self.terminate_delta:
                break
        return self.clusters, km_delta

    def _node_reassignment(self):
        self.clusters = {i: np.zeros(0, dtype=np.int64) for i in range(self.num_clusters)}
        n = self.data_feat.shape[0]
        distance = np.zeros((self.num_clusters, n))
        for i in range(self.num_clusters):
            distance[i] = np.sum((self.data_feat - self.centroid[i]) ** 2, axis=1)
        sort_indices = np.unravel_index(np.argsort(distance, axis=None), distance.shape)
        clusters = sort_indices[0]
        users = sort_indices[1]
        selected_nodes = np.zeros(0, dtype=np.int64)
        counter = 0
        while len(selected_nodes) < n and counter < clusters.size:
            cluster = int(clusters[counter])
            user = int(users[counter])
            if self.clusters[cluster].size < self.node_threshold:
                self.clusters[cluster] = np.append(self.clusters[cluster], user)
                selected_nodes = np.append(selected_nodes, user)
                user_indices = np.where(users == user)[0]
                a = np.arange(users.size)
                b = user_indices[user_indices > counter]
                remain_indices = a[np.where(np.logical_not(np.isin(a, b)))[0]]
                if remain_indices.size > 0:
                    clusters = clusters[remain_indices]
                    users = users[remain_indices]
                counter = 0
                continue
            counter += 1
        # Assign any remaining nodes to nearest cluster
        selected_set = set(selected_nodes.tolist())
        for node in range(n):
            if node not in selected_set:
                d = np.sum((self.data_feat[node] - np.array([self.centroid[c] for c in range(self.num_clusters)])) ** 2, axis=1)
                c = np.argmin(d)
                self.clusters[c] = np.append(self.clusters[c], node)

    def _centroid_updating(self):
        for i in range(self.num_clusters):
            if self.clusters[i].size > 0:
                self.centroid[i] = np.mean(self.data_feat[self.clusters[i].astype(int)], axis=0)

    def _centroid_delta(self, centroid_pre, centroid_cur):
        delta = 0.0
        for i in range(len(centroid_cur)):
            delta += np.sum(np.abs(centroid_cur[i] - centroid_pre[i]))
        return delta


def _get_modified_edge_index(data, forget_mask, device):
    """Remove edges incident to forget set (graph as if forget nodes were removed from structure)."""
    ei = data.edge_index.to(device)
    keep = ~(forget_mask[ei[0]] | forget_mask[ei[1]])
    return ei[:, keep]


def _get_embeddings(model, x, edge_index, device):
    """Get node embeddings; use get_embeddings if available else logits."""
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'get_embeddings'):
            z = model.get_embeddings(x.to(device), edge_index.to(device))
        else:
            z = model(x.to(device), edge_index.to(device))
    return z.cpu().numpy()


def _run_grapheraser_homogeneous(args, model, data, forget_mask, retain_mask, device):
    """GraphEraser for homogeneous graphs: partition with K-means, sharded retrain, wrapper for inference."""
    num_nodes = data.num_nodes
    num_classes = int(data.y.max().item()) + 1
    if hasattr(data, 'num_classes'):
        num_classes = getattr(data, 'num_classes', num_classes)

    num_clusters = getattr(args, 'grapheraser_num_clusters', 4)
    kmeans_iters = getattr(args, 'grapheraser_kmeans_iters', 20)
    terminate_delta = getattr(args, 'grapheraser_terminate_delta', 1e-4)
    shard_delta = getattr(args, 'grapheraser_shard_delta', 0.1)
    shard_epochs = getattr(args, 'grapheraser_epochs', None) or getattr(args, 'learning_epochs', 200)

    modified_edge_index = _get_modified_edge_index(data, forget_mask, device)
    # Embeddings on modified graph (no forget edges) for partitioning
    data_mod = Data(x=data.x, edge_index=modified_edge_index, y=data.y)
    data_mod = data_mod.to(device)
    z = _get_embeddings(model, data.x, modified_edge_index, device)
    if z.ndim > 2:
        z = z.reshape(z.shape[0], -1)
    if np.any(np.isnan(z)) or np.any(np.isinf(z)):
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    node_threshold = math.ceil(
        num_nodes / num_clusters + shard_delta * (num_nodes - num_nodes / num_clusters)
    )
    cluster = ConstrainedKmeans(
        z, num_clusters, node_threshold, terminate_delta, max_iteration=kmeans_iters
    )
    cluster.initialization()
    community, _ = cluster.clustering()
    community_to_node = {i: np.array(community[i].astype(int)) for i in range(num_clusters)}

    entropy_weight = getattr(args, 'grapheraser_entropy_weight', 0.0)
    shard_models = []
    for shard_id in trange(num_clusters, desc='GraphEraser sharded retraining'):
        subset = torch.from_numpy(community_to_node[shard_id]).long().to(device)
        edge_index_shard, _ = subgraph(
            subset, modified_edge_index, relabel_nodes=True, num_nodes=num_nodes
        )
        x_shard = data.x[subset.cpu()].to(device)
        y_shard = data.y[subset.cpu()].to(device)
        # Train mask: retain nodes in this shard
        train_mask_shard = retain_mask[subset.cpu()].to(device)
        if train_mask_shard.sum() == 0:
            # No retain nodes in this shard: still create a model, train minimally or skip loss
            train_mask_shard = torch.ones(y_shard.size(0), dtype=torch.bool, device=device)
        # Forget nodes in this shard (local indices: 0..len(subset)-1)
        forget_in_shard = forget_mask[subset.cpu()].to(device)
        data_shard = Data(
            x=x_shard,
            edge_index=edge_index_shard,
            y=y_shard,
            train_mask=train_mask_shard,
        )
        model_shard = create_model(args, data, is_kg=False).to(device)
        optimizer = optim.Adam(model_shard.parameters(), lr=args.learning_lr, weight_decay=args.weight_decay)
        for _ in range(shard_epochs):
            loss = _train_shard(
                model_shard, data_shard, optimizer, device,
                forget_mask_local=forget_in_shard if entropy_weight > 0 else None,
                entropy_weight=entropy_weight,
            )
        shard_models.append(model_shard)

    return GraphEraserWrapper(shard_models, community_to_node, num_classes, device)


def _train_shard(model_shard, data_shard, optimizer, device, forget_mask_local=None, entropy_weight=0.0):
    """Train shard: retain CE loss + optional entropy maximization on forget nodes (local indices)."""
    model_shard.train()
    optimizer.zero_grad()
    out = model_shard(data_shard.x, data_shard.edge_index)
    loss = F.cross_entropy(out[data_shard.train_mask], data_shard.y[data_shard.train_mask])
    if entropy_weight > 0 and forget_mask_local is not None and forget_mask_local.sum() > 0:
        p = F.softmax(out[forget_mask_local], dim=-1)
        log_p = torch.log(p.clamp(min=1e-10))
        forget_entropy = -(p * log_p).sum(dim=-1).mean()
        loss = loss - entropy_weight * forget_entropy
    loss.backward()
    optimizer.step()
    return loss.item()


class GraphEraserWrapper(torch.nn.Module):
    """
    Wrapper that holds one model per shard; forward runs the correct shard per node and merges logits.
    """

    def __init__(self, shard_models, community_to_node, num_classes, device):
        super().__init__()
        self.shard_models = torch.nn.ModuleList(shard_models)
        self.community_to_node = community_to_node
        self.num_clusters = len(shard_models)
        self.num_classes = num_classes
        self._device = device

    def forward(self, x, edge_index):
        device = x.device
        num_nodes = x.size(0)
        out = torch.zeros(num_nodes, self.num_classes, device=device, dtype=x.dtype)
        for shard_id in range(self.num_clusters):
            nodes = self.community_to_node[shard_id]
            if len(nodes) == 0:
                continue
            subset = torch.as_tensor(nodes, device=device, dtype=torch.long)
            edge_index_s, _ = subgraph(subset, edge_index, relabel_nodes=True, num_nodes=num_nodes)
            x_s = x[subset]
            logits_s = self.shard_models[shard_id](x_s, edge_index_s)
            out[subset] = logits_s
        return out


def load_grapheraser_wrapper(checkpoint, args, data, device):
    """
    Rebuild GraphEraser wrapper from a saved checkpoint for evaluation cache loading.
    Checkpoint must contain: model_state_dict, grapheraser_community_to_node, grapheraser_num_classes.
    """
    community_to_node = checkpoint['grapheraser_community_to_node']
    num_classes = int(checkpoint['grapheraser_num_classes'])
    num_clusters = len(community_to_node)
    shard_models = [create_model(args, data, is_kg=False) for _ in range(num_clusters)]
    wrapper = GraphEraserWrapper(shard_models, community_to_node, num_classes, device)
    wrapper.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return wrapper.to(device)


def grapheraser_unlearn(args, model, data, is_kg, forget_mask, retain_mask):
    """
    GraphEraser: Graph Erasing Method (sharded retraining with constrained K-means partition).

    Args:
        args: Arguments (optional: grapheraser_num_clusters, grapheraser_kmeans_iters, ...)
        model: Trained model (used for embeddings only; shards are retrained)
        data: Graph data
        is_kg: Whether it's a knowledge graph
        forget_mask: Mask for forget set (nodes)
        retain_mask: Mask for retain set (nodes)

    Returns:
        model: Unlearned model (GraphEraserWrapper for homogeneous; gradient_ascent for KG)
    """
    set_unlearn_seed(getattr(args, 'seed', 42))

    print("\n" + "=" * 60)
    print("GraphEraser: Graph Erasing Method (Sharded Retraining)")
    print("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if is_kg:
        from unlearning_methods.gradient_ascent import gradient_ascent_baseline
        print("  GraphEraser for KG not implemented; using Gradient Ascent fallback.")
        forget_mask = forget_mask.to(device)
        retain_mask = retain_mask.to(device)
        return gradient_ascent_baseline(args, model, data, True, forget_mask, retain_mask)

    data = data.to(device)
    forget_mask = forget_mask.to(device)
    retain_mask = retain_mask.to(device)
    model = _run_grapheraser_homogeneous(args, model, data, forget_mask, retain_mask, device)
    print("GraphEraser unlearning completed!")
    return model
