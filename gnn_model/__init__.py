"""
GNN Model Architectures
Includes models for both homogeneous graphs and knowledge graphs
"""

from .homogeneous_models import GCN, GAT, GraphSAGE
from .knowledge_graph_models import RGCN, CompGCN

__all__ = ['GCN', 'GAT', 'GraphSAGE', 'RGCN', 'CompGCN']



