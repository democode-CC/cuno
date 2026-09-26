"""
GNN Models for Knowledge Graphs
Includes: RGCN, CompGCN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RGCN(nn.Module):
    """Relational Graph Convolutional Network for Knowledge Graphs"""
    
    def __init__(self, num_entities, num_relations, hidden_dim, num_layers=2, 
                 dropout=0.5, num_bases=None):
        super(RGCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_entities = num_entities
        self.num_relations = num_relations
        
        # Entity embeddings
        self.entity_embedding = nn.Embedding(num_entities, hidden_dim)
        
        # Use basis decomposition to reduce parameters
        if num_bases is None:
            num_bases = min(num_relations, 30)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, 
                                      num_relations=num_relations,
                                      num_bases=num_bases))
        
        # Output layer for link prediction
        self.w_relation = nn.Parameter(torch.Tensor(num_relations, hidden_dim))
        nn.init.xavier_uniform_(self.w_relation)
        
        nn.init.xavier_uniform_(self.entity_embedding.weight)
    
    def forward(self, entity_ids, edge_index, edge_type):
        """
        Forward pass for node representation
        Args:
            entity_ids: Entity IDs (for node classification)
            edge_index: Edge index [2, num_edges]
            edge_type: Edge types [num_edges]
        """
        # Get initial embeddings
        x = self.entity_embedding(entity_ids)
        
        # Apply R-GCN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def predict_link(self, entity_ids, edge_index, edge_type, head_idx, tail_idx, rel_idx):
        """
        Predict link scores using DistMult scoring
        Args:
            head_idx: Head entity indices
            tail_idx: Tail entity indices
            rel_idx: Relation indices
        Returns:
            scores: Link prediction scores
        """
        # Get entity representations
        x = self.forward(entity_ids, edge_index, edge_type)
        
        # Get head, tail, and relation embeddings
        head_emb = x[head_idx]
        tail_emb = x[tail_idx]
        rel_emb = self.w_relation[rel_idx]
        
        # DistMult scoring: <h, r, t>
        scores = torch.sum(head_emb * rel_emb * tail_emb, dim=1)
        return scores
    
    def get_embeddings(self, entity_ids, edge_index, edge_type):
        """Get entity embeddings"""
        return self.forward(entity_ids, edge_index, edge_type)


class CompGCN(nn.Module):
    """Composition-based GCN for Knowledge Graphs"""
    
    def __init__(self, num_entities, num_relations, hidden_dim, num_layers=2,
                 dropout=0.5, comp_fn='mult'):
        super(CompGCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.comp_fn = comp_fn
        self.num_entities = num_entities
        self.num_relations = num_relations
        
        # Entity and relation embeddings
        self.entity_embedding = nn.Embedding(num_entities, hidden_dim)
        # Each relation has forward and backward
        self.relation_embedding = nn.Embedding(2 * num_relations, hidden_dim)
        
        # Composition layers
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(CompGCNLayer(hidden_dim, hidden_dim, comp_fn))
        
        # Output layer
        self.w_relation = nn.Parameter(torch.Tensor(num_relations, hidden_dim))
        
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        nn.init.xavier_uniform_(self.w_relation)
    
    def forward(self, entity_ids, edge_index, edge_type):
        """
        Forward pass
        Args:
            entity_ids: Entity IDs
            edge_index: Edge index [2, num_edges]
            edge_type: Edge types [num_edges]
        """
        x = self.entity_embedding(entity_ids)
        r = self.relation_embedding.weight
        
        for i, conv in enumerate(self.conv_layers):
            x, r = conv(x, edge_index, edge_type, r)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def predict_link(self, entity_ids, edge_index, edge_type, head_idx, tail_idx, rel_idx):
        """Predict link scores"""
        x = self.forward(entity_ids, edge_index, edge_type)
        
        head_emb = x[head_idx]
        tail_emb = x[tail_idx]
        rel_emb = self.w_relation[rel_idx]
        
        # DistMult scoring
        scores = torch.sum(head_emb * rel_emb * tail_emb, dim=1)
        return scores
    
    def get_embeddings(self, entity_ids, edge_index, edge_type):
        """Get entity embeddings"""
        return self.forward(entity_ids, edge_index, edge_type)


class CompGCNLayer(nn.Module):
    """Single CompGCN layer with composition function"""
    
    def __init__(self, in_dim, out_dim, comp_fn='mult'):
        super(CompGCNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.comp_fn = comp_fn
        
        # Separate weights for different edge directions
        self.w_loop = nn.Linear(in_dim, out_dim)
        self.w_in = nn.Linear(in_dim, out_dim)
        self.w_out = nn.Linear(in_dim, out_dim)
        self.w_rel = nn.Linear(in_dim, out_dim)
        
        self.bn = nn.BatchNorm1d(out_dim)
    
    def compose(self, h, r):
        """Composition function"""
        if self.comp_fn == 'mult':
            return h * r
        elif self.comp_fn == 'sub':
            return h - r
        elif self.comp_fn == 'corr':
            return torch.fft.ifft(torch.fft.fft(h) * torch.fft.fft(r)).real
        else:
            raise ValueError(f"Unknown composition function: {self.comp_fn}")
    
    def forward(self, x, edge_index, edge_type, rel_embed):
        """Forward pass"""
        num_nodes = x.size(0)
        
        # Self-loop
        out = self.w_loop(x)
        
        # Incoming edges
        row, col = edge_index
        for i in range(edge_type.max().item() + 1):
            mask = edge_type == i
            if mask.sum() == 0:
                continue
            
            masked_row = row[mask]
            masked_col = col[mask]
            
            # Compose entity and relation
            composed = self.compose(x[masked_col], rel_embed[i].unsqueeze(0))
            
            # Aggregate messages
            out.index_add_(0, masked_row, self.w_in(composed))
        
        # Update relation embeddings
        new_rel_embed = self.w_rel(rel_embed)
        
        out = self.bn(out)
        return out, new_rel_embed



