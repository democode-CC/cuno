"""
GNN Models for Homogeneous Graphs
Includes: GCN, GAT, GraphSAGE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


class GCN(nn.Module):
    """Graph Convolutional Network"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim))
        else:
            self.convs[0] = GCNConv(input_dim, output_dim)
    
    def forward(self, x, edge_index):
        """Forward pass"""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer (no activation, no dropout)
        x = self.convs[-1](x, edge_index)
        return x
    
    def get_embeddings(self, x, edge_index):
        """Get node embeddings before the final classification layer"""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
        return x


class GAT(nn.Module):
    """Graph Attention Network"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, 
                 dropout=0.5, heads=8, output_heads=1):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        # First layer
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, 
                                     heads=heads, dropout=dropout))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GATConv(hidden_dim * heads, output_dim, 
                                     heads=output_heads, concat=False, dropout=dropout))
        else:
            self.convs[0] = GATConv(input_dim, output_dim, 
                                   heads=output_heads, concat=False, dropout=dropout)
    
    def forward(self, x, edge_index):
        """Forward pass"""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.convs[-1](x, edge_index)
        return x
    
    def get_embeddings(self, x, edge_index):
        """Get node embeddings before the final classification layer"""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
        return x


class GraphSAGE(nn.Module):
    """GraphSAGE Model"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, output_dim))
        else:
            self.convs[0] = SAGEConv(input_dim, output_dim)
    
    def forward(self, x, edge_index):
        """Forward pass"""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.convs[-1](x, edge_index)
        return x
    
    def get_embeddings(self, x, edge_index):
        """Get node embeddings before the final classification layer"""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
        return x



