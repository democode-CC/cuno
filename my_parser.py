"""
Argument parser for Graph Unlearning experiments
Handles all hyperparameters for learning, unlearning, and evaluation
"""

import argparse


def create_parser():
    """Create argument parser with all hyperparameters for the experiments"""
    parser = argparse.ArgumentParser(
        description='Graph Unlearning with Curriculum Learning and NPO'
    )
    
    # ==================== Dataset & Model Configuration ====================
    parser.add_argument('--dataset', type=str, default='Cora',
                      choices=['Cora', 'CiteSeer', 'PubMed',
                               'AmazonComputers', 'CoauthorCS', 'Actor',
                               'RomanEmpire', 'AmazonRatings', 'AmazonPhoto',
                               'FB15k237', 'WN18RR'],
                      help='Dataset name (homogeneous / additional / knowledge graphs)')
    
    parser.add_argument('--gnn_model', type=str, default='GCN',
                      choices=['GCN', 'GAT', 'GraphSAGE', 'RGCN', 'CompGCN'],
                      help='GNN model architecture (GCN/GAT/GraphSAGE for homogeneous, RGCN/CompGCN for KG)')
    
    # ==================== Learning Phase ====================
    parser.add_argument('--learning_epochs', type=int, default=200,
                      help='Number of epochs for initial training')
    
    parser.add_argument('--learning_lr', type=float, default=0.01,
                      help='Learning rate for initial training')
    
    parser.add_argument('--hidden_dim', type=int, default=64,
                      help='Hidden dimension for GNN layers')
    
    parser.add_argument('--num_layers', type=int, default=2,
                      help='Number of GNN layers')
    
    parser.add_argument('--dropout', type=float, default=0.5,
                      help='Dropout rate')
    
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                      help='Weight decay (L2 regularization)')
    
    # ==================== Unlearning Configuration ====================
    parser.add_argument('--unlearn_method', type=str, default='gnndelete',
                      choices=['retrain', 'gradient_ascent', 'curriculum_ga', 'npo_ga', 'full_method', 'gif', 'gnndelete', 'grapheraser', 'inpo', 'etr'],
                      help='Unlearning method: retrain, gradient_ascent (GA), curriculum_ga (GA+Step1), npo_ga (GA+Step2), full_method (GA+Step1+Step2), gif (GIF), gnndelete (GNNDELETE), grapheraser (GraphEraser), inpo (INPO), etr (ETR)')
    
    parser.add_argument('--unlearn_rate', type=float, default=0.1,
                      choices=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
                      help='Percentage of training data to unlearn (1%, 2%, 5%, 10%, 20%, 50%)')
    
    parser.add_argument('--unlearn_epochs', type=int, default=50,
                      help='Number of epochs for unlearning')
    
    parser.add_argument('--unlearn_lr', type=float, default=0.1,
                      help='Learning rate for unlearning (gradient ascent)')
    parser.add_argument('--retain_weight', type=float, default=0.8,
                      help='Weight for retain CE regularization in gradient_ascent and curriculum_ga (default: 0.1)')

    # ==================== Curriculum Unlearning (Step 1) ====================
    parser.add_argument('--num_curricula', type=int, default=8,
                      choices=[1, 2, 4, 8],
                      help='Number of curriculum levels (C=1,2,4,8)')
    
    parser.add_argument('--complexity_metric', type=str, default='retain_coupling',
                      choices=['degree', 'betweenness', 'pagerank', 'clustering', 'eigenvector', 'label',
                               'prediction_confidence', 'retain_coupling', 'gradient_norm',
                               'multihop_retain_coverage', 'retain_betweenness', 'class_boundary'],
                      help='Graph complexity metric for curriculum design. '
                           'Structural: degree/betweenness/pagerank/clustering/eigenvector. '
                           'Model-aware: prediction_confidence / retain_coupling / gradient_norm. '
                           'Graph+Task-aware: multihop_retain_coverage (GNN L-hop propagation scope), '
                           'retain_betweenness (structural load-bearing for retain subgraph), '
                           'class_boundary (class-boundary heterophily for node classification).')

    parser.add_argument('--hop_decay', type=float, default=0.5,
                      help='Decay factor alpha for multihop_retain_coverage metric (default: 0.5). '
                           'Reflects that each additional hop has weaker message-passing influence.')
    
    parser.add_argument('--curriculum_mode', type=str, default='overlapping',
                      choices=['overlapping', 'non_overlapping'],
                      help='Whether curricula have overlapping nodes')

    parser.add_argument('--curriculum_order', type=str, default='hard_to_easy',
                      choices=['easy_to_hard', 'hard_to_easy'],
                      help='Order of curriculum stages: hard_to_easy (default) or easy_to_hard')
    
    parser.add_argument('--overlap_ratio', type=float, default=0.2,
                      help='Overlap ratio between consecutive curricula (only for overlapping mode)')
    
    # ==================== NPO Configuration (Step 2) ====================
    parser.add_argument('--npo_beta', type=float, default=0.01,
                      help='Beta parameter for NPO loss (controls preference strength)')
    
    parser.add_argument('--npo_temperature', type=float, default=1.0,
                      help='Temperature parameter for NPO')
    
    parser.add_argument('--npo_lambda', type=float, default=0.1,
                      help='Balance between unlearning and utility preservation')

    parser.add_argument('--npo_loss_mode', type=str, default='zero_sum',
                      choices=['zero_sum', 'decoupled'],
                      help='Loss formulation: zero_sum = lam*forget + (1-lam)*retain; '
                           'decoupled = forget + lam*retain (lam on retain only)')

    # ==================== INPO Configuration ====================
    parser.add_argument('--inpo_te_weight', type=float, default=0,
                      help='Weight for local structure alignment (TE) loss in INPO (default: 0.5)')

    # ==================== ETR Configuration ====================
    parser.add_argument('--etr_erase_ratio', type=float, default=1,
                      help='Fraction of parameters to erase in each FIM condition (default: 0.01 = 1%%; paper recommends 0.008–0.016)')
    parser.add_argument('--etr_erase_strength', type=float, default=0.9,
                      help='Erase strength: 0=soft scaling (p*=b/γ), 1=full zero, 0.5=milder shrink (default: 0)')
    parser.add_argument('--etr_lambda', type=float, default=0.01,
                      help='Step size for the Rectify gradient update (default: 0.3; paper recommends 0.3–0.5)')

    # ==================== GIF Configuration ====================
    parser.add_argument('--gif_iteration', type=int, default=100,
                      help='Number of HVP iterations for GIF approximation (default: 100)')
    
    parser.add_argument('--gif_scale', type=int, default=500,
                      help='Scaling factor for GIF approximation (default: 500)')
    
    parser.add_argument('--gif_damp', type=float, default=0.0,
                      help='Damping factor for GIF approximation (default: 0.0)')
    
    # ==================== GNNDelete Configuration ====================
    parser.add_argument('--gnndelete_lr', type=float, default=2e-2,
                      help='Learning rate for GNNDelete deletion operator (default: 2e-2, more aggressive)')
    parser.add_argument('--gnndelete_epochs', type=int, default=None,
                      help='Epochs for GNNDelete (default: max(100, unlearn_epochs))')
    parser.add_argument('--gnndelete_alpha', type=float, default=0.01,
                      help='Weight for retain loss (default: 0.8); lower = more aggressive forget, higher = preserve utility more')
    parser.add_argument('--gnndelete_consistency_weight', type=float, default=0.01,
                      help='Weight for consistency loss / forget strength (default: 3.0); increase to unlearn more aggressively')
    parser.add_argument('--gnndelete_id_weight', type=float, default=0.1,
                      help='Weight for identity loss (default: 0.1); lower = deletion can change more (more aggressive)')
    parser.add_argument('--gnndelete_forget_weight', type=float, default=5,
                      help='Extra weight on forget-set nodes in consistency loss (default: 2.0); >1 = push forget nodes harder')
    parser.add_argument('--gnndelete_ga_retain_weight', type=float, default=3,
                      help='Weight for gradient ascent on non-affected retain nodes (default: 1.0); >0 pushes non-affected retain nodes away from correct labels to reduce MU')
    parser.add_argument('--gnndelete_entropy_weight', type=float, default=1,
                      help='Weight for forget-node entropy loss (default: 0.5); >0 pushes deletion output on forget nodes toward uniform to improve FE')
    parser.add_argument('--gnndelete_entropy_on_original_weight', type=float, default=0.5,
                      help='Weight for entropy on deletion(base(original))[forget] (default: 0.3); extra push for FE')
    parser.add_argument('--gnndelete_id_all_affected', action='store_true',
                      help='Apply identity loss to ALL affected nodes (default: only retain-in-affected); set for less aggressive forget')
    
    # ==================== GraphEraser Configuration ====================
    parser.add_argument('--grapheraser_num_clusters', type=int, default=4,
                      help='Number of shards/clusters for GraphEraser (default: 4)')
    parser.add_argument('--grapheraser_kmeans_iters', type=int, default=20,
                      help='Max K-means iterations for graph partitioning (default: 20)')
    parser.add_argument('--grapheraser_terminate_delta', type=float, default=1e-4,
                      help='K-means early termination when centroid delta below this (default: 1e-4)')
    parser.add_argument('--grapheraser_shard_delta', type=float, default=0.1,
                      help='Shard size delta for constrained K-means (default: 0.1)')
    parser.add_argument('--grapheraser_epochs', type=int, default=None,
                      help='Epochs per shard (default: same as learning_epochs)')
    parser.add_argument('--grapheraser_entropy_weight', type=float, default=0.5,
                      help='Weight for forget-node entropy loss per shard (default: 0.5); >0 pushes shard output on forget nodes toward uniform to improve FE')
    
    # ==================== Evaluation Configuration ====================
    parser.add_argument('--eval_type', type=str, default='baseline_comparison',
                      choices=['baseline_comparison', 'hyperparameter_sensitivity'],
                      help='Type of evaluation to run')
    
    parser.add_argument('--methods', type=str, default=None,
                      help='Comma-separated unlearning methods to run (e.g. gnndelete,gif); default: all')
    parser.add_argument('--only_this_config', action='store_true',
                      help='Only run the single (dataset, gnn_model) given by --dataset and --gnn_model (for baseline/ablation)')
    parser.add_argument('--unlearn_rates', type=str, default=None,
                      help='Comma-separated unlearn rates for baseline (e.g. 0.1 or 0.1,0.2); default: full list')
    parser.add_argument('--skip_kg', dest='skip_kg', action='store_true', default=None,
                      help='Run only on homogeneous graphs (Cora, CiteSeer, PubMed); skip KG')
    parser.add_argument('--no_skip_kg', dest='skip_kg', action='store_false', default=None,
                      help='Run on all datasets including KG (FB15k237, WN18RR)')
    
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    parser.add_argument('--device', type=str, default='cuda',
                      choices=['cuda', 'cpu'],
                      help='Device to use for training')
    
    # ==================== File Paths ====================
    parser.add_argument('--data_dir', type=str, default='./data',
                      help='Directory for datasets')
    
    parser.add_argument('--model_dir', type=str, default='./stored_model',
                      help='Directory to save/load models')
    
    parser.add_argument('--result_dir', type=str, default='./results',
                      help='Directory to save results')
    
    # ==================== Experiment Control ====================
    parser.add_argument('--save_model', action='store_true', default=True,
                      help='Save trained models')
    
    parser.add_argument('--load_pretrained', action='store_true', default=True,
                      help='Load pretrained (trained) model if available; does not affect unlearned cache.')
    parser.add_argument('--load_cached_unlearned', dest='load_cached_unlearned', action='store_true', default=False,
                      help='Load cached unlearned models when available (default: False, always run unlearning).')
    parser.add_argument('--no_load_cached_unlearned', dest='load_cached_unlearned', action='store_false',
                      help='Do not load cached unlearned models (explicit default).')
    
    parser.add_argument('--verbose', action='store_true', default=True,
                      help='Print detailed training information')
    
    return parser


def get_args():
    """Parse and return arguments"""
    parser = create_parser()
    args = parser.parse_args()
    return args


def get_args_with_custom(custom_args=None):
    """Parse arguments with custom string (useful for programmatic calls)"""
    parser = create_parser()
    if custom_args:
        args = parser.parse_args(custom_args)
    else:
        args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Test parser
    args = get_args()
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")



