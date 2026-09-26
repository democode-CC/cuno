"""
GNN model architectures for node classification on homogeneous graphs.
"""

from .homogeneous_models import GCN, GAT, GraphSAGE

__all__ = ['GCN', 'GAT', 'GraphSAGE']
