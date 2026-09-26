"""
Learning Phase: Initial training of GNN models
Supports both homogeneous graphs and knowledge graphs
"""

import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from gnn_model import GCN, GAT, GraphSAGE, RGCN, CompGCN
from data import load_dataset, split_data


def create_model(args, data, is_kg):
    """
    Create GNN model based on configuration
    
    Args:
        args: Arguments containing model configuration
        data: Dataset
        is_kg: Whether it's a knowledge graph
        
    Returns:
        model: Initialized GNN model
    """
    if is_kg:
        # Knowledge graph models
        num_entities = data['num_entities']
        num_relations = data['num_relations']
        
        if args.gnn_model == 'RGCN':
            model = RGCN(
                num_entities=num_entities,
                num_relations=num_relations,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout
            )
        elif args.gnn_model == 'CompGCN':
            model = CompGCN(
                num_entities=num_entities,
                num_relations=num_relations,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout
            )
        else:
            raise ValueError(f"Model {args.gnn_model} not supported for knowledge graphs. Use RGCN or CompGCN.")
    else:
        # Homogeneous graph models
        input_dim = data.num_features
        output_dim = data.y.max().item() + 1
        
        if args.gnn_model == 'GCN':
            model = GCN(
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                output_dim=output_dim,
                num_layers=args.num_layers,
                dropout=args.dropout
            )
        elif args.gnn_model == 'GAT':
            model = GAT(
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                output_dim=output_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                heads=8
            )
        elif args.gnn_model == 'GraphSAGE':
            model = GraphSAGE(
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                output_dim=output_dim,
                num_layers=args.num_layers,
                dropout=args.dropout
            )
        else:
            raise ValueError(f"Model {args.gnn_model} not supported for homogeneous graphs. Use GCN, GAT, or GraphSAGE.")
    
    return model


def train_homogeneous(model, data, optimizer, device):
    """Train model on homogeneous graph (node classification)"""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data.x.to(device), data.edge_index.to(device))
    
    # Calculate loss on training nodes
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask].to(device))
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()


def train_knowledge_graph(model, data, optimizer, device, batch_size=1024):
    """Train model on knowledge graph (link prediction)"""
    model.train()
    
    edge_index = data['train']['edge_index'].to(device)
    edge_type = data['train']['edge_type'].to(device)
    entity_ids = data['entity_ids'].to(device)
    num_edges = edge_index.size(1)
    
    # Sample a batch of edges
    indices = torch.randperm(num_edges)[:batch_size]
    
    head_idx = edge_index[0, indices]
    tail_idx = edge_index[1, indices]
    rel_idx = edge_type[indices]
    
    optimizer.zero_grad()
    
    # Positive scores
    pos_scores = model.predict_link(entity_ids, edge_index, edge_type, head_idx, tail_idx, rel_idx)
    
    # Negative sampling: corrupt tail entities
    neg_tail_idx = torch.randint(0, data['num_entities'], (batch_size,), device=device)
    neg_scores = model.predict_link(entity_ids, edge_index, edge_type, head_idx, neg_tail_idx, rel_idx)
    
    # Margin ranking loss
    loss = F.margin_ranking_loss(
        pos_scores, neg_scores,
        torch.ones(batch_size, device=device),
        margin=1.0
    )
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate_homogeneous(model, data, device, mask):
    """Evaluate model on homogeneous graph"""
    model.eval()
    
    out = model(data.x.to(device), data.edge_index.to(device))
    pred = out.argmax(dim=1)
    
    correct = (pred[mask] == data.y[mask].to(device)).sum()
    acc = int(correct) / int(mask.sum())
    
    return acc


@torch.no_grad()
def evaluate_knowledge_graph(model, data, device, num_samples=1000):
    """Evaluate model on knowledge graph using MRR"""
    model.eval()
    
    edge_index = data['train']['edge_index'].to(device)
    edge_type = data['train']['edge_type'].to(device)
    entity_ids = data['entity_ids'].to(device)
    num_edges = edge_index.size(1)
    
    # Sample edges for evaluation
    indices = torch.randperm(num_edges)[:num_samples]
    
    head_idx = edge_index[0, indices]
    tail_idx = edge_index[1, indices]
    rel_idx = edge_type[indices]
    
    ranks = []
    
    for i in range(num_samples):
        h, r, t = head_idx[i:i+1], rel_idx[i:i+1], tail_idx[i:i+1]
        
        # Score against all entities
        all_entities = torch.arange(data['num_entities'], device=device)
        scores = model.predict_link(entity_ids, edge_index, edge_type, 
                                   h.repeat(data['num_entities']), all_entities, r.repeat(data['num_entities']))
        
        # Get rank of true tail
        sorted_indices = torch.argsort(scores, descending=True)
        rank = (sorted_indices == t).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
    
    mrr = np.mean([1.0 / r for r in ranks])
    return mrr


def train_model(args, save_path=None):
    """
    Main training function
    
    Args:
        args: Training arguments
        save_path: Path to save the trained model
        
    Returns:
        model: Trained model
        data: Dataset
        is_kg: Whether it's a knowledge graph
    """
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    data, is_kg = load_dataset(args.dataset, args.data_dir)
    
    # Create model
    print(f"Creating model: {args.gnn_model}")
    model = create_model(args, data, is_kg).to(device)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_lr,
        weight_decay=args.weight_decay
    )
    
    # Move data to device
    if not is_kg:
        data = data.to(device)
    
    # Training loop
    print(f"\nTraining for {args.learning_epochs} epochs...")
    best_val_metric = 0
    best_model_state = None
    
    pbar = tqdm(range(args.learning_epochs), desc="Training")
    for epoch in pbar:
        # Train
        if is_kg:
            loss = train_knowledge_graph(model, data, optimizer, device)
        else:
            loss = train_homogeneous(model, data, optimizer, device)
        
        # Evaluate periodically
        if (epoch + 1) % 10 == 0 or epoch == 0:
            if is_kg:
                val_metric = evaluate_knowledge_graph(model, data, device)
                metric_name = "MRR"
            else:
                train_acc = evaluate_homogeneous(model, data, device, data.train_mask)
                val_metric = evaluate_homogeneous(model, data, device, data.val_mask)
                test_acc = evaluate_homogeneous(model, data, device, data.test_mask)
                metric_name = "Acc"
            
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_model_state = model.state_dict().copy()
            
            if is_kg:
                pbar.set_postfix({
                    'Loss': f'{loss:.4f}',
                    f'Val {metric_name}': f'{val_metric:.4f}',
                    f'Best {metric_name}': f'{best_val_metric:.4f}'
                })
            else:
                pbar.set_postfix({
                    'Loss': f'{loss:.4f}',
                    f'Train {metric_name}': f'{train_acc:.4f}',
                    f'Val {metric_name}': f'{val_metric:.4f}',
                    f'Test {metric_name}': f'{test_acc:.4f}'
                })
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation:")
    if is_kg:
        final_mrr = evaluate_knowledge_graph(model, data, device, num_samples=5000)
        print(f"MRR: {final_mrr:.4f}")
    else:
        train_acc = evaluate_homogeneous(model, data, device, data.train_mask)
        val_acc = evaluate_homogeneous(model, data, device, data.val_mask)
        test_acc = evaluate_homogeneous(model, data, device, data.test_mask)
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
    print("="*60)
    
    # Save model
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'args': args,
            'is_kg': is_kg
        }, save_path)
        print(f"\nModel saved to: {save_path}")
    
    return model, data, is_kg


if __name__ == '__main__':
    from my_parser import get_args
    
    args = get_args()
    
    # Define save path
    save_path = os.path.join(
        args.model_dir,
        f"{args.dataset}_{args.gnn_model}_trained.pt"
    )
    
    # Train model
    train_model(args, save_path)



