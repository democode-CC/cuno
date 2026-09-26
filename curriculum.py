import os
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import networkx as nx
from copy import deepcopy

class ComplexityCalculator:
    """Calculate graph complexity metrics for curriculum design"""

    def __init__(self, data, is_kg=False, model=None, retain_mask=None, device='cpu'):
        self.data = data
        self.is_kg = is_kg
        self.model = model
        self.retain_mask = retain_mask
        self.device = device

    def calculate_complexity(self, nodes, metric='degree'):
        """
        Calculate complexity scores for nodes

        Args:
            nodes: List of node indices
            metric: Complexity metric (degree, betweenness, pagerank, clustering,
                    eigenvector, prediction_confidence, retain_coupling, gradient_norm)

        Returns:
            complexity_scores: Dictionary mapping node to complexity score
        """
        if metric == 'degree':
            return self._calculate_degree(nodes)
        elif metric == 'betweenness':
            return self._calculate_betweenness(nodes)
        elif metric == 'pagerank':
            return self._calculate_pagerank(nodes)
        elif metric == 'clustering':
            return self._calculate_clustering(nodes)
        elif metric == 'eigenvector':
            return self._calculate_eigenvector(nodes)
        elif metric == 'prediction_confidence':
            return self._calculate_prediction_confidence(nodes)
        elif metric == 'retain_coupling':
            return self._calculate_retain_coupling(nodes)
        elif metric == 'gradient_norm':
            return self._calculate_gradient_norm(nodes)
        elif metric == 'multihop_retain_coverage':
            return self._calculate_multihop_retain_coverage(nodes)
        elif metric == 'retain_betweenness':
            return self._calculate_retain_betweenness(nodes)
        elif metric == 'class_boundary':
            return self._calculate_class_boundary(nodes)
        else:
            raise ValueError(f"Unknown complexity metric: {metric}")
    
    def _calculate_degree(self, nodes):
        """Calculate node degree"""
        if self.is_kg:
            edge_index = self.data['train']['edge_index']
        else:
            edge_index = self.data.edge_index
        
        degrees = {}
        for node in nodes:
            node_item = node.item() if torch.is_tensor(node) else node
            # Count both in and out degrees
            degree = ((edge_index[0] == node_item).sum() + (edge_index[1] == node_item).sum()).item()
            degrees[node_item] = degree
        
        return degrees
    
    def _calculate_betweenness(self, nodes):
        """Calculate betweenness centrality"""
        G = self._to_networkx()
        betweenness = nx.betweenness_centrality(G)
        
        result = {}
        for node in nodes:
            node_item = node.item() if torch.is_tensor(node) else node
            result[node_item] = betweenness.get(node_item, 0.0)
        
        return result
    
    def _calculate_pagerank(self, nodes):
        """Calculate PageRank"""
        G = self._to_networkx()
        pagerank = nx.pagerank(G)
        
        result = {}
        for node in nodes:
            node_item = node.item() if torch.is_tensor(node) else node
            result[node_item] = pagerank.get(node_item, 0.0)
        
        return result
    
    def _calculate_clustering(self, nodes):
        """Calculate clustering coefficient"""
        G = self._to_networkx()
        clustering = nx.clustering(G)
        
        result = {}
        for node in nodes:
            node_item = node.item() if torch.is_tensor(node) else node
            result[node_item] = clustering.get(node_item, 0.0)
        
        return result
    
    def _calculate_eigenvector(self, nodes):
        """Calculate eigenvector centrality"""
        G = self._to_networkx()
        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            # If doesn't converge, use degree as fallback
            eigenvector = {node: G.degree(node) for node in G.nodes()}
        
        result = {}
        for node in nodes:
            node_item = node.item() if torch.is_tensor(node) else node
            result[node_item] = eigenvector.get(node_item, 0.0)
        
        return result
    
    # ------------------------------------------------------------------
    # Advanced model-aware complexity metrics
    # ------------------------------------------------------------------

    def _calculate_prediction_confidence(self, nodes):
        """
        Metric: Prediction Confidence (negative entropy of model output).

        Nodes where the model is very confident (low entropy / high softmax
        peak) are more deeply memorised and harder to unlearn.

        complexity(v) = -H(p(v)) = sum_c p_c(v) * log(p_c(v))   (always <= 0;
        less negative = more confident = higher complexity)

        For KGs we use the link-prediction score of the forget triple as a
        proxy (higher score = model is more "certain" about this fact).
        """
        if self.model is None:
            raise ValueError("prediction_confidence requires a model. "
                             "Pass model= to ComplexityCalculator.")

        result = {}
        self.model.eval()

        with torch.no_grad():
            if self.is_kg:
                edge_index = self.data['train']['edge_index'].to(self.device)
                edge_type  = self.data['train']['edge_type'].to(self.device)
                entity_ids = self.data['entity_ids'].to(self.device)

                for node in nodes:
                    idx = node.item() if torch.is_tensor(node) else node
                    head_idx = edge_index[0, idx].unsqueeze(0)
                    tail_idx = edge_index[1, idx].unsqueeze(0)
                    rel_idx  = edge_type[idx].unsqueeze(0)
                    score = self.model.predict_link(
                        entity_ids, edge_index, edge_type,
                        head_idx, tail_idx, rel_idx
                    )
                    result[idx] = score.item()
            else:
                data = self.data.to(self.device)
                out  = self.model(data.x, data.edge_index)          # [N, C]
                probs = F.softmax(out, dim=-1)                       # [N, C]
                # negative entropy: high = confident = complex
                neg_entropy = (probs * torch.log(probs + 1e-8)).sum(dim=-1)  # [N]

                for node in nodes:
                    idx = node.item() if torch.is_tensor(node) else node
                    result[idx] = neg_entropy[idx].item()

        return result

    def _calculate_retain_coupling(self, nodes):
        """
        Metric: Retain-Coupling Similarity.

        Measures how semantically entangled a forget node is with its
        retain-set neighbours in the model's embedding space.

        complexity(v) = mean_{u in N(v) ∩ retain} cosine_sim(h_v, h_u)

        High coupling => forgetting v risks disrupting retain representations
        => harder to unlearn safely.

        For KGs we use head/tail entity embedding similarity as a proxy.
        """
        if self.model is None:
            raise ValueError("retain_coupling requires a model. "
                             "Pass model= to ComplexityCalculator.")
        if self.retain_mask is None:
            raise ValueError("retain_coupling requires retain_mask. "
                             "Pass retain_mask= to ComplexityCalculator.")

        self.model.eval()
        result = {}

        with torch.no_grad():
            if self.is_kg:
                edge_index  = self.data['train']['edge_index'].to(self.device)
                edge_type   = self.data['train']['edge_type'].to(self.device)
                entity_ids  = self.data['entity_ids'].to(self.device)

                # Get entity embeddings
                emb = self.model.entity_embedding(entity_ids)        # [E, d]
                emb = F.normalize(emb, dim=-1)

                retain_edges = self.retain_mask.nonzero(as_tuple=True)[0]
                retain_heads = set(edge_index[0, retain_edges].cpu().tolist())
                retain_tails = set(edge_index[1, retain_edges].cpu().tolist())
                retain_entities = retain_heads | retain_tails

                for node in nodes:
                    idx = node.item() if torch.is_tensor(node) else node
                    h = edge_index[0, idx].item()
                    t = edge_index[1, idx].item()

                    sims = []
                    for e in [h, t]:
                        # Neighbours of this entity in retain set
                        mask_h = (edge_index[0] == e) | (edge_index[1] == e)
                        nbr_edges = mask_h.nonzero(as_tuple=True)[0]
                        nbr_entities = set(
                            edge_index[0, nbr_edges].cpu().tolist() +
                            edge_index[1, nbr_edges].cpu().tolist()
                        ) & retain_entities - {e}
                        if nbr_entities:
                            e_emb  = emb[e].unsqueeze(0)             # [1, d]
                            nbr_id = list(nbr_entities)
                            n_emb  = emb[nbr_id]                     # [k, d]
                            sim    = (e_emb * n_emb).sum(dim=-1).mean().item()
                            sims.append(sim)
                    result[idx] = float(np.mean(sims)) if sims else 0.0

            else:
                data = self.data.to(self.device)
                # Use the penultimate layer as the embedding
                # Works for GCN/GAT/GraphSAGE via standard forward
                emb = self.model(data.x, data.edge_index)            # [N, C]
                emb = F.normalize(emb, dim=-1)

                edge_index = data.edge_index
                retain_node_set = set(
                    self.retain_mask.nonzero(as_tuple=True)[0].cpu().tolist()
                )

                for node in nodes:
                    idx = node.item() if torch.is_tensor(node) else node
                    # All neighbours of this node
                    nbr_mask  = (edge_index[0] == idx) | (edge_index[1] == idx)
                    nbr_edges = nbr_mask.nonzero(as_tuple=True)[0]
                    nbrs = set(
                        edge_index[0, nbr_edges].cpu().tolist() +
                        edge_index[1, nbr_edges].cpu().tolist()
                    ) & retain_node_set - {idx}

                    if nbrs:
                        v_emb  = emb[idx].unsqueeze(0)               # [1, d]
                        nbr_id = list(nbrs)
                        n_emb  = emb[nbr_id]                         # [k, d]
                        sim    = (v_emb * n_emb).sum(dim=-1).mean().item()
                        result[idx] = sim
                    else:
                        result[idx] = 0.0

        return result

    def _calculate_gradient_norm(self, nodes):
        """
        Metric: Per-node Gradient Norm (influence magnitude).

        For each forget node v, compute ||∇_θ L_CE(θ; v)||₂ — the L2 norm
        of the gradient of the cross-entropy loss for that single node.

        Larger gradient norm => the model's weights are more strongly shaped
        by this node => harder to erase without large weight perturbation.

        This is the first-order approximation of the influence function and
        is the most principled metric for measuring unlearning difficulty.

        For KGs we use the triple-level loss gradient norm instead.

        NOTE: This is O(|forget_set| × |params|) — for large graphs consider
        computing on a subset or using param groups.
        """
        if self.model is None:
            raise ValueError("gradient_norm requires a model. "
                             "Pass model= to ComplexityCalculator.")

        result = {}

        # Ensure gradients flow regardless of outer torch.no_grad() context,
        # model training mode, or per-parameter requires_grad=False.
        was_training = self.model.training
        self.model.train()

        # Save and temporarily enable requires_grad on all parameters
        # (reference_model in full_method has requires_grad=False on all params)
        param_grad_states = {p: p.requires_grad for p in self.model.parameters()}
        for p in self.model.parameters():
            p.requires_grad_(True)

        with torch.enable_grad():
            if self.is_kg:
                edge_index = self.data['train']['edge_index'].to(self.device)
                edge_type  = self.data['train']['edge_type'].to(self.device)
                entity_ids = self.data['entity_ids'].to(self.device)

                for node in nodes:
                    idx = node.item() if torch.is_tensor(node) else node
                    self.model.zero_grad()

                    head_idx = edge_index[0, idx].unsqueeze(0)
                    tail_idx = edge_index[1, idx].unsqueeze(0)
                    rel_idx  = edge_type[idx].unsqueeze(0)

                    score = self.model.predict_link(
                        entity_ids, edge_index, edge_type,
                        head_idx, tail_idx, rel_idx
                    )
                    loss = -score.mean()
                    loss.backward()

                    grad_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.detach().norm(2).item() ** 2
                    result[idx] = grad_norm ** 0.5

                    self.model.zero_grad()
            else:
                data = self.data.to(self.device)

                for node in nodes:
                    idx = node.item() if torch.is_tensor(node) else node
                    self.model.zero_grad()

                    out  = self.model(data.x, data.edge_index)
                    loss = F.cross_entropy(out[idx].unsqueeze(0),
                                           data.y[idx].unsqueeze(0))
                    loss.backward()

                    grad_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.detach().norm(2).item() ** 2
                    result[idx] = grad_norm ** 0.5

                    self.model.zero_grad()

        # Restore original parameter requires_grad states and training mode
        for p, grad_state in param_grad_states.items():
            p.requires_grad_(grad_state)
        if not was_training:
            self.model.eval()

        return result

    # ------------------------------------------------------------------
    # Graph-structure × Task-aware complexity metrics
    # ------------------------------------------------------------------

    def _calculate_multihop_retain_coverage(self, nodes):
        """
        Metric: Multi-hop Retain Contamination Coverage (MRC).

        In an L-layer GNN, message passing propagates a node's information
        exactly L hops.  During training, forget node v's features/labels have
        been aggregated by every retain node within its L-hop neighbourhood.
        The more retain nodes fall inside this receptive field — and the closer
        they are — the harder it is to fully erase v's influence.

        complexity(v) = Σ_{k=1}^{L}  α^{k-1} · |N^k(v) ∩ V_r|

        where N^k(v) is the set of nodes EXACTLY k hops from v (excluding
        nearer hops), α ∈ (0,1) is the hop-decay factor (default 0.5 mirrors
        the 1/deg normalisation of mean-aggregation GNNs), and L = num_layers.

        This metric is the only one that directly encodes the GNN architecture
        (L) into the complexity score.  For KGs we treat the bipartite
        head–tail entity graph as undirected and apply the same BFS procedure.
        """
        L      = getattr(self, 'num_layers', 2)
        alpha  = getattr(self, 'hop_decay',  0.5)

        if self.is_kg:
            edge_index = self.data['train']['edge_index'].cpu()
            # retain_mask indexes into the EDGE list; convert to entity IDs
            if self.retain_mask is not None:
                retain_edge_idx = self.retain_mask.nonzero(as_tuple=True)[0].cpu()
                retain_set = set(
                    edge_index[0, retain_edge_idx].tolist() +
                    edge_index[1, retain_edge_idx].tolist()
                )
            else:
                retain_set = set(
                    edge_index[0].tolist() + edge_index[1].tolist()
                )

            # Build entity-level adjacency list
            from collections import defaultdict
            adj = defaultdict(set)
            ei = edge_index.numpy()
            for h, t in zip(ei[0], ei[1]):
                adj[h].add(t)
                adj[t].add(h)

            result = {}
            for node in nodes:
                idx = node.item() if torch.is_tensor(node) else node
                # Seed BFS from both endpoints of the forget triple
                head = edge_index[0, idx].item()
                tail = edge_index[1, idx].item()
                seeds = {head, tail}

                visited  = set(seeds)
                frontier = set(seeds)
                score    = 0.0
                for k in range(1, L + 1):
                    next_frontier = set()
                    for ent in frontier:
                        next_frontier |= (adj[ent] - visited)
                    visited |= next_frontier
                    score += (alpha ** (k - 1)) * len(next_frontier & retain_set)
                    frontier = next_frontier
                result[idx] = score
        else:
            edge_index = self.data.edge_index.cpu()
            retain_set = set(
                self.retain_mask.nonzero(as_tuple=True)[0].cpu().tolist()
            )

            # Build adjacency list once for all nodes
            from collections import defaultdict
            adj = defaultdict(set)
            ei = edge_index.numpy()
            for src, dst in zip(ei[0], ei[1]):
                adj[src].add(dst)
                adj[dst].add(src)

            result = {}
            for node in nodes:
                idx      = node.item() if torch.is_tensor(node) else node
                visited  = {idx}
                frontier = {idx}
                score    = 0.0
                for k in range(1, L + 1):
                    next_frontier = set()
                    for v in frontier:
                        next_frontier |= (adj[v] - visited)
                    visited |= next_frontier
                    score += (alpha ** (k - 1)) * len(next_frontier & retain_set)
                    frontier = next_frontier
                result[idx] = score

        return result

    def _calculate_retain_betweenness(self, nodes, max_retain_sample=500):
        """
        Metric: Retain-Conditioned Betweenness Centrality (RCBC).

        Standard betweenness centrality counts shortest paths between ALL
        node pairs.  For graph unlearning, the only paths that matter are
        those connecting pairs of RETAIN nodes — because it is the retain
        subgraph's connectivity that must be preserved after forgetting.

        complexity(v) = Σ_{s,t ∈ V_r} σ_r(s,t|v) / σ_r(s,t)
                        ──────────────────────────────────────
                                  C(|V_r|, 2)

        where σ_r(s,t) is the total number of shortest paths between retain
        nodes s and t, and σ_r(s,t|v) is the number of those paths passing
        through forget node v.

        A forget node with high RCBC is a structural "bridge" for the retain
        subgraph: removing it would force retain–retain communication to reroute
        through longer, less direct paths — degrading the retain subgraph's
        effective connectivity and making the GNN's retain-side representations
        harder to preserve.

        Uses NetworkX betweenness_centrality_subset (O(|V_r|·|E|)) with an
        optional retain-node sample cap for large graphs.
        """
        G = self._to_networkx()

        if self.retain_mask is not None:
            if self.is_kg:
                # retain_mask indexes EDGES; convert to entity IDs for NetworkX
                edge_index = self.data['train']['edge_index'].cpu()
                retain_edge_idx = self.retain_mask.nonzero(as_tuple=True)[0].cpu()
                retain_nodes_all = list(set(
                    edge_index[0, retain_edge_idx].tolist() +
                    edge_index[1, retain_edge_idx].tolist()
                ))
            else:
                retain_nodes_all = self.retain_mask.nonzero(as_tuple=True)[0].cpu().tolist()
        else:
            retain_nodes_all = list(G.nodes())

        # Sub-sample retain nodes for large graphs to keep runtime tractable
        if len(retain_nodes_all) > max_retain_sample:
            rng = np.random.default_rng(seed=42)
            retain_nodes = rng.choice(
                retain_nodes_all, size=max_retain_sample, replace=False
            ).tolist()
        else:
            retain_nodes = retain_nodes_all

        # betweenness_centrality_subset efficiently computes paths only between
        # the specified source–target subsets
        betweenness = nx.betweenness_centrality_subset(
            G,
            sources=retain_nodes,
            targets=retain_nodes,
            normalized=True,
        )

        result = {}
        for node in nodes:
            idx = node.item() if torch.is_tensor(node) else node
            result[idx] = betweenness.get(idx, 0.0)

        return result

    def _calculate_class_boundary(self, nodes):
        """
        Metric: Class-Boundary Heterophily (CBH).

        Exploits the label structure of node classification tasks and the
        graph's homophily property.  In homophilic graphs (Cora, CiteSeer,
        PubMed all have homophily > 0.7), same-class nodes tend to be
        connected.  A forget node v that sits at a CLASS BOUNDARY — i.e., most
        of its retain-set neighbours belong to a different class — encodes
        cross-class discriminative information:

          • The model had to "fight against" the neighbourhood signal to
            correctly classify v.
          • v's embedding carries strong class-boundary information that
            does not exist in interior nodes.
          • Unlearning v via gradient ascent in a smooth direction is harder
            because the loss landscape is non-smooth at class boundaries.

        We define complexity using ONLY the retain-side neighbours (not other
        forget nodes) because after unlearning, only the retain subgraph
        remains and must stay intact:

        complexity(v) = 1 − |{u ∈ N(v) ∩ V_r : y_u = y_v}|
                            ──────────────────────────────
                                    |N(v) ∩ V_r|

        High score (≈1) = v is surrounded by retain nodes of different classes
                         = class-boundary node = harder to unlearn.
        Low score (≈0)  = v is in a homophilic region = easier to unlearn.

        For knowledge graphs (no node labels), we use the RELATION-TYPE
        ENTROPY of the forget triple's endpoint entities as a proxy:
        entities that participate in many diverse relation types occupy a
        semantically richer, more "boundary" position in the KG.
        """
        if self.is_kg:
            edge_index = self.data['train']['edge_index']
            edge_type  = self.data['train']['edge_type']

            result = {}
            for node in nodes:
                idx = node.item() if torch.is_tensor(node) else node
                h   = edge_index[0, idx].item()
                t   = edge_index[1, idx].item()

                entropies = []
                for ent in [h, t]:
                    mask    = (edge_index[0] == ent) | (edge_index[1] == ent)
                    rel_ids = edge_type[mask].cpu().numpy()
                    if len(rel_ids) == 0:
                        entropies.append(0.0)
                        continue
                    counts = np.bincount(rel_ids)
                    probs  = counts / counts.sum()
                    entropy = -np.sum(probs * np.log(probs + 1e-8))
                    entropies.append(entropy)

                result[idx] = float(np.mean(entropies))
        else:
            edge_index   = self.data.edge_index
            labels       = self.data.y
            retain_set   = set(
                self.retain_mask.nonzero(as_tuple=True)[0].cpu().tolist()
                if self.retain_mask is not None
                else range(self.data.num_nodes)
            )

            result = {}
            for node in nodes:
                idx  = node.item() if torch.is_tensor(node) else node
                y_v  = labels[idx].item()

                # Neighbours that belong to the retain set
                nbr_mask  = (edge_index[0] == idx) | (edge_index[1] == idx)
                nbr_edges = nbr_mask.nonzero(as_tuple=True)[0]
                nbrs = (
                    set(edge_index[0, nbr_edges].cpu().tolist() +
                        edge_index[1, nbr_edges].cpu().tolist())
                    & retain_set
                ) - {idx}

                if nbrs:
                    same = sum(1 for u in nbrs if labels[u].item() == y_v)
                    # heterophily = 1 − homophily
                    result[idx] = 1.0 - same / len(nbrs)
                else:
                    result[idx] = 0.0

        return result

    # ------------------------------------------------------------------

    def _to_networkx(self):
        """Convert PyG data to NetworkX graph"""
        if self.is_kg:
            edge_index = self.data['train']['edge_index'].cpu().numpy()
        else:
            edge_index = self.data.edge_index.cpu().numpy()

        G = nx.Graph()
        edges = list(zip(edge_index[0], edge_index[1]))
        G.add_edges_from(edges)

        return G


class CurriculumDesigner:
    """Design curriculum for unlearning"""

    def __init__(self, forget_mask, data, is_kg, complexity_metric='degree',
                 num_curricula=4, mode='overlapping', overlap_ratio=0.2,
                 model=None, retain_mask=None, device='cpu',
                 num_layers=2, hop_decay=0.5, curriculum_order='hard_to_easy'):
        self.forget_mask = forget_mask
        self.data = data
        self.is_kg = is_kg
        self.complexity_metric = complexity_metric
        self.num_curricula = num_curricula
        self.mode = mode
        self.overlap_ratio = overlap_ratio
        self.curriculum_order = curriculum_order

        # Get forget nodes/edges
        self.forget_indices = forget_mask.nonzero(as_tuple=True)[0].cpu().numpy()

        # Calculate complexity
        self.calculator = ComplexityCalculator(
            data, is_kg,
            model=model,
            retain_mask=retain_mask,
            device=device,
        )
        # Pass GNN architecture parameters for multihop_retain_coverage
        self.calculator.num_layers = num_layers
        self.calculator.hop_decay  = hop_decay

        self.complexity_scores = self.calculator.calculate_complexity(
            self.forget_indices, metric=complexity_metric
        )
    
    def design_curricula(self):
        """
        Design curricula from simple to complex
        
        Returns:
            curricula: List of curricula, each containing node/edge indices
        """
        # Sort nodes by complexity
        reverse = (self.curriculum_order == 'hard_to_easy')
        sorted_nodes = sorted(self.complexity_scores.items(), key=lambda x: x[1], reverse=reverse)
        sorted_indices = [node for node, _ in sorted_nodes]
        
        if self.mode == 'non_overlapping':
            return self._non_overlapping_split(sorted_indices)
        else:
            return self._overlapping_split(sorted_indices)
    
    def _non_overlapping_split(self, sorted_indices):
        """Split into non-overlapping curricula"""
        n = len(sorted_indices)
        chunk_size = n // self.num_curricula
        
        curricula = []
        for i in range(self.num_curricula):
            start = i * chunk_size
            end = start + chunk_size if i < self.num_curricula - 1 else n
            
            curriculum_nodes = sorted_indices[start:end]
            
            # Create mask
            mask = torch.zeros_like(self.forget_mask)
            mask[curriculum_nodes] = True
            
            curricula.append(mask)
        
        return curricula
    
    def _overlapping_split(self, sorted_indices):
        """Split into overlapping curricula"""
        n = len(sorted_indices)
        
        # Calculate step size with overlap
        step_size = int(n / (self.num_curricula * (1 - self.overlap_ratio) + self.overlap_ratio))
        chunk_size = int(step_size / (1 - self.overlap_ratio))
        
        curricula = []
        for i in range(self.num_curricula):
            start = int(i * step_size)
            end = min(start + chunk_size, n)
            
            if start >= n:
                break
            
            curriculum_nodes = sorted_indices[start:end]
            
            # Create mask
            mask = torch.zeros_like(self.forget_mask)
            mask[curriculum_nodes] = True
            
            curricula.append(mask)
        
        return curricula