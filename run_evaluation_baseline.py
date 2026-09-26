"""
Evaluation: Baseline Methods Comparison
Compare baseline unlearning methods across all datasets and GNN models
Baseline methods: Retrain, Gradient Ascent, GIF, GNNDELETE, GraphEraser, Full Method (CUNO)
"""

import os
import glob
import hashlib
import torch
import numpy as np
import pandas as pd
import json
from copy import deepcopy
from tqdm import tqdm

from my_parser import get_args_with_custom
from learning import train_model, create_model, evaluate_homogeneous, evaluate_knowledge_graph
from unlearning import unlearn
from data import load_dataset, get_forget_retain_split
from evaluation import (
    calculate_forget_effect,
    calculate_model_utility,
    evaluate_knowledge_graph_subset,
    evaluate_homogeneous_filtered
)
from fe_evaluation_methods import MIAEvaluator


def _gnndelete_cache_suffix(args):
    """Return cache filename suffix for GNNDelete so any hyperparameter or implementation change triggers retrain."""
    from unlearning_methods.gnndelete import GNNDELETE_IMPL_VERSION
    _keys = (
        GNNDELETE_IMPL_VERSION,
        getattr(args, 'gnndelete_alpha', 0.5),
        getattr(args, 'gnndelete_consistency_weight', 1.0),
        getattr(args, 'gnndelete_epochs', None),
        getattr(args, 'gnndelete_lr', 2e-2),
    )
    return f"_v{GNNDELETE_IMPL_VERSION}_h{hashlib.md5(str(_keys).encode()).hexdigest()[:8]}"


def evaluate_method(method_name, args, original_model, data, is_kg, forget_mask, retain_mask, test_mask, device, dataset, model_name, unlearn_rate, mia_evaluator=None):
    """
    Evaluate a single unlearning method
    
    Args:
        method_name: Name of the unlearning method
        args: Arguments
        original_model: Original trained model
        data: Graph data
        is_kg: Whether it's a knowledge graph
        forget_mask: Mask for forget set
        retain_mask: Mask for retain set
        test_mask: Mask for test set
        device: Device to run on
        dataset: Dataset name
        model_name: Model name
        unlearn_rate: Unlearn rate
        
    Returns:
        result: Dictionary with evaluation results
        model: Unlearned model
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Method: {method_name}")
    print(f"{'='*60}")
    
    # Try to load cached unlearned model first
    unlearned_model_dir = os.path.join(args.model_dir, 'unlearned')
    rate_str = f"{unlearn_rate:.2f}".replace('.', 'p')
    # GNNDelete: cache key includes all training params (hash) so changing any triggers retrain
    if method_name == 'gnndelete':
        _gnn_suffix = _gnndelete_cache_suffix(args)
    elif method_name == 'grapheraser':
        _k = getattr(args, 'grapheraser_num_clusters', 4)
        _gnn_suffix = f"_k{_k}"
    else:
        _gnn_suffix = ""
    cached_model_path = os.path.join(
        unlearned_model_dir,
        f"{dataset}_{model_name}_{method_name}_rate{rate_str}{_gnn_suffix}_unlearned.pt"
    )
    
    load_cache = os.path.exists(cached_model_path) and args.load_cached_unlearned
    if load_cache:
        print(f"[CACHE HIT] Loading cached unlearned model from {cached_model_path}")
        try:
            checkpoint = torch.load(cached_model_path, map_location=device)
            # GNNDelete (homogeneous): reject old-format cache, then load wrapper
            if method_name == 'gnndelete' and not is_kg:
                state = checkpoint.get('model_state_dict') or {}
                if state and 'deletion.0.weight' in state and 'deletion.linear.weight' not in state:
                    raise ValueError("GNNDelete cache is old format; retraining and overwriting.")
                from unlearning_methods.gnndelete import load_gnndelete_wrapper
                model = load_gnndelete_wrapper(
                    checkpoint, args, data, is_kg,
                    forget_mask.to(device) if forget_mask is not None else None,
                    device,
                )
            elif method_name == 'grapheraser' and not is_kg and 'grapheraser_community_to_node' in checkpoint:
                from unlearning_methods.grapheraser import load_grapheraser_wrapper
                model = load_grapheraser_wrapper(checkpoint, args, data, device)
            else:
                model = create_model(args, data, is_kg).to(device)
                model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Model loaded successfully")
            
            # Load cached metrics if available
            fe_standard = checkpoint.get('forget_effect', None)
            mu = checkpoint.get('model_utility', None)
            mu_original = checkpoint.get('model_utility_original', None)
            
            # Initialize results with cached or computed metrics
            fe_results = {
                'fe_standard': fe_standard if fe_standard is not None else calculate_forget_effect(model, data, is_kg, forget_mask, retain_mask, device),
                'forget_effect': fe_standard if fe_standard is not None else calculate_forget_effect(model, data, is_kg, forget_mask, retain_mask, device)
            }
            
            if mu is None:
                mu = calculate_model_utility(model, data, is_kg, retain_mask, test_mask, forget_mask, device)
            if mu_original is None:
                mu_original = calculate_model_utility(model, data, is_kg, retain_mask, test_mask, forget_mask, device, use_original_graph=True)
            
            # Advanced FE evaluation (optional, controlled by args)
            if hasattr(args, 'use_advanced_fe') and args.use_advanced_fe:
                print(f"\n  [Advanced FE Evaluation]")
                try:
                    train_mask = data.train_mask if hasattr(data, 'train_mask') else None
                    
                    advanced_results, _ = calculate_forget_effect(
                        model, data, is_kg, forget_mask, retain_mask, device,
                        use_multi_method=True,
                        methods=['mia', 'embedding'],
                        train_mask=train_mask,
                        test_mask=test_mask,
                        mia_evaluator=mia_evaluator,
                        verbose=False
                    )
                    
                    # Add MIA results
                    if advanced_results.get('fe_mia') is not None:
                        fe_results['fe_mia'] = advanced_results['fe_mia']
                        print(f"    MIA-based FE: {advanced_results['fe_mia']:.4f}")
                    
                    # Add Embedding results
                    if advanced_results.get('fe_embedding') is not None:
                        fe_results['fe_embedding'] = advanced_results['fe_embedding']
                        print(f"    Embedding-based FE: {advanced_results['fe_embedding']:.4f}")
                        
                        # Add detailed embedding metrics
                        if 'embedding_details' in advanced_results:
                            details = advanced_results['embedding_details']
                            fe_results['forget_retain_distance'] = details.get('forget_retain_distance')
                            fe_results['forget_random_distance'] = details.get('forget_random_distance')
                            fe_results['forget_concentration'] = details.get('forget_concentration')
                    
                    # Add Backdoor results (if available)
                    if advanced_results.get('fe_backdoor') is not None:
                        fe_results['fe_backdoor'] = advanced_results['fe_backdoor']
                        print(f"    Backdoor-based FE: {advanced_results['fe_backdoor']:.4f}")
                        
                except Exception as e:
                    print(f"    ⚠️  Advanced FE evaluation failed: {e}")
            else:
                # Try to load cached advanced metrics
                if 'fe_mia' in checkpoint:
                    fe_results['fe_mia'] = checkpoint['fe_mia']
                if 'fe_embedding' in checkpoint:
                    fe_results['fe_embedding'] = checkpoint['fe_embedding']
                if 'forget_retain_distance' in checkpoint:
                    fe_results['forget_retain_distance'] = checkpoint['forget_retain_distance']
                if 'forget_random_distance' in checkpoint:
                    fe_results['forget_random_distance'] = checkpoint['forget_random_distance']
                if 'forget_concentration' in checkpoint:
                    fe_results['forget_concentration'] = checkpoint['forget_concentration']
                if 'fe_backdoor' in checkpoint:
                    fe_results['fe_backdoor'] = checkpoint['fe_backdoor']
            
        except Exception as e:
            print(f"⚠ Failed to load unlearned model: {e}")
            print(f"  Performing unlearning from scratch...")
            model = deepcopy(original_model)
            args.unlearn_method = method_name
            model = unlearn(args, model, data, is_kg, forget_mask, retain_mask)
            fe_standard = calculate_forget_effect(model, data, is_kg, forget_mask, retain_mask, device)
            mu = calculate_model_utility(model, data, is_kg, retain_mask, test_mask, forget_mask, device)
            # Also calculate MU on original graph for fair comparison across unlearn_rates
            mu_original = calculate_model_utility(model, data, is_kg, retain_mask, test_mask, forget_mask, device, use_original_graph=True)
            
            # Initialize results with standard FE
            fe_results = {
                'fe_standard': fe_standard,
                'forget_effect': fe_standard  # Backward compatibility
            }
    else:
        # Perform unlearning from scratch
        if method_name == 'gnndelete':
            _s = _gnndelete_cache_suffix(args)
            print(f"[CACHE MISS] GNNDelete will run from scratch; cache suffix: {_s} (save path: ...{_s}_unlearned.pt)")
        print(f"Performing unlearning...")
        model = deepcopy(original_model)
        args.unlearn_method = method_name
        model = unlearn(args, model, data, is_kg, forget_mask, retain_mask)
        
        # Calculate metrics (with corrected evaluation - no forget structure leakage)
        # Standard FE evaluation (always computed for backward compatibility)
        fe_standard = calculate_forget_effect(model, data, is_kg, forget_mask, retain_mask, device)
        
        # Initialize results with standard FE
        fe_results = {
            'fe_standard': fe_standard,
            'forget_effect': fe_standard  # Backward compatibility
        }
        
        # Advanced FE evaluation (optional, controlled by args)
        if hasattr(args, 'use_advanced_fe') and args.use_advanced_fe:
            print(f"\n  [Advanced FE Evaluation]")
            try:
                # Get train_mask for MIA
                train_mask = data.train_mask if hasattr(data, 'train_mask') else None
                
                # Use multi-method evaluation
                advanced_results, _ = calculate_forget_effect(
                    model, data, is_kg, forget_mask, retain_mask, device,
                    use_multi_method=True,
                    methods=['mia', 'embedding', 'backdoor'],
                    train_mask=train_mask,
                    test_mask=test_mask,
                    mia_evaluator=mia_evaluator,
                    verbose=False
                )
                
                # Add MIA results
                if advanced_results.get('fe_mia') is not None:
                    fe_results['fe_mia'] = advanced_results['fe_mia']
                    print(f"    MIA-based FE: {advanced_results['fe_mia']:.4f}")
                
                # Add Embedding results
                if advanced_results.get('fe_embedding') is not None:
                    fe_results['fe_embedding'] = advanced_results['fe_embedding']
                    print(f"    Embedding-based FE: {advanced_results['fe_embedding']:.4f}")
                    
                    # Add detailed embedding metrics
                    if 'embedding_details' in advanced_results:
                        details = advanced_results['embedding_details']
                        fe_results['forget_retain_distance'] = details.get('forget_retain_distance')
                        fe_results['forget_random_distance'] = details.get('forget_random_distance')
                        fe_results['forget_concentration'] = details.get('forget_concentration')
                
                # Add Backdoor results (if available)
                if advanced_results.get('fe_backdoor') is not None:
                    fe_results['fe_backdoor'] = advanced_results['fe_backdoor']
                    print(f"    Backdoor-based FE: {advanced_results['fe_backdoor']:.4f}")
                    
            except Exception as e:
                print(f"    ⚠️  Advanced FE evaluation failed: {e}")
        
        # Calculate Model Utility
        mu = calculate_model_utility(model, data, is_kg, retain_mask, test_mask, forget_mask, device)
        # Also calculate MU on original graph for fair comparison across unlearn_rates
        mu_original = calculate_model_utility(model, data, is_kg, retain_mask, test_mask, forget_mask, device, use_original_graph=True)
    
    # Print results
    print(f"\nResults for {method_name}:")
    print(f"  Forget Effect (FE): {fe_results['forget_effect']:.4f}")
    print(f"  Model Utility (MU - filtered): {mu:.4f}")
    print(f"  Model Utility (MU - original graph): {mu_original:.4f}")
    
    # Build result dictionary
    result = {
        'method': method_name,
        'forget_effect': fe_results['forget_effect'],
        'fe_standard': fe_results['fe_standard'],
        'model_utility': mu,
        'model_utility_original': mu_original
    }
    
    # Add advanced FE metrics if available
    if 'fe_mia' in fe_results:
        result['fe_mia'] = fe_results['fe_mia']
    if 'fe_embedding' in fe_results:
        result['fe_embedding'] = fe_results['fe_embedding']
    if 'forget_retain_distance' in fe_results:
        result['forget_retain_distance'] = fe_results['forget_retain_distance']
    if 'forget_random_distance' in fe_results:
        result['forget_random_distance'] = fe_results['forget_random_distance']
    if 'forget_concentration' in fe_results:
        result['forget_concentration'] = fe_results['forget_concentration']
    if 'fe_backdoor' in fe_results:
        result['fe_backdoor'] = fe_results['fe_backdoor']
    
    return result, model


def set_seed(seed):
    """Set random seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        # Additional settings for deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_baseline_comparison(args, methods_override=None, skip_kg=False,
                            datasets_override=None, result_prefix=None):
    """Run baseline methods comparison across all datasets and models.
    methods_override:  if provided (e.g. ['gif']), only run these methods; otherwise run all.
    skip_kg:           if True, skip KG datasets (FB15k237, WN18RR).
    datasets_override: if provided, replace the default dataset list entirely.
    result_prefix:     prefix for output CSV/JSON filenames (default: 'baseline_comparison').
    """

    # Set global random seed for reproducibility
    set_seed(args.seed)
    print(f"✓ Random seed set to: {args.seed}")
    
    # Enable advanced FE evaluation methods
    args.use_advanced_fe = True
    print("✓ Advanced FE evaluation enabled (MIA + Embedding)")
    print("  Results will include: fe_standard, fe_mia, fe_embedding")
    print("  Plus detailed metrics: forget_retain_distance, forget_random_distance, forget_concentration")
    print()
    
    # Define experiment configurations
    KG_DATASETS = ['FB15k237', 'WN18RR']
    if datasets_override is not None:
        datasets = datasets_override
        print(f"✓ Using custom dataset list: {datasets}")
    else:
        datasets = ['Cora', 'CiteSeer', 'PubMed', 'FB15k237', 'WN18RR']
    if skip_kg:
        datasets = [d for d in datasets if d not in KG_DATASETS]
        print("✓ Skipping KG: only homogeneous datasets")
        print()
    homogeneous_models = ['GCN', 'GAT', 'GraphSAGE']
    kg_models = ['RGCN', 'CompGCN']
    models_filter = None
    if getattr(args, 'only_this_config', False):
        datasets = [args.dataset]
        models_filter = [args.gnn_model]
        print(f"✓ Only this config: dataset={args.dataset}, model={args.gnn_model}")
    
    # Baseline methods: Retrain, Gradient Ascent, GIF, GNNDELETE, GraphEraser, Full Method (CUNO), INPO
    all_methods = ['retrain', 'gradient_ascent', 'gif', 'gnndelete', 'grapheraser', 'full_method', 'inpo', 'etr']
    methods = methods_override if methods_override else all_methods
    if methods_override:
        invalid = [m for m in methods_override if m not in all_methods]
        if invalid:
            raise ValueError(f"Unknown method(s): {invalid}. Valid: {all_methods}")
        print(f"✓ Running only specified methods: {methods}")
    method_names = {
        'baseline_trained': 'Baseline (Trained Model)',
        'retrain': 'Retrain',
        'gradient_ascent': 'Gradient Ascent',
        'gif': 'GIF (Graph Influence Function)',
        'gnndelete': 'GNNDELETE',
        'grapheraser': 'GraphEraser',
        'full_method': 'CUNO',
        'inpo': 'INPO (Influence-aware NPO)',
        'etr': 'ETR (Erase then Rectify)',
    }
    
    # Test multiple unlearn rates (override with --unlearn_rates 0.1 for quick debug)
    if getattr(args, 'unlearn_rates', None):
        unlearn_rates = [float(x.strip()) for x in args.unlearn_rates.split(',')]
        print(f"✓ Unlearn rates (from args): {[f'{r*100:.0f}%' for r in unlearn_rates]}")
    else:
        unlearn_rates = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.5]
        print(f"✓ Testing multiple unlearn rates: {[f'{r*100:.0f}%' for r in unlearn_rates]}")
    print()
    
    all_results = []
    
    for dataset in datasets:
        # Determine if KG and select appropriate models
        is_kg = dataset in KG_DATASETS
        models = kg_models if is_kg else homogeneous_models
        if models_filter is not None:
            models = [m for m in models if m in models_filter]
        if not models:
            continue
        for model_name in models:
            print(f"\n{'#'*80}")
            print(f"# Dataset: {dataset}, Model: {model_name}")
            print(f"{'#'*80}")
            
            # Update args
            args.dataset = dataset
            args.gnn_model = model_name
            
            # Set device
            device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
            
            # Load data
            data, is_kg = load_dataset(dataset, args.data_dir)
            
            # Check if trained model exists
            model_path = os.path.join(args.model_dir, f"{dataset}_{model_name}_trained.pt")
            
            if os.path.exists(model_path) and args.load_pretrained:
                print(f"Loading pretrained model from {model_path}")
                try:
                    checkpoint = torch.load(model_path, map_location=device)
                    original_model = create_model(args, data, is_kg).to(device)
                    original_model.load_state_dict(checkpoint['model_state_dict'])
                    print("✓ Model loaded successfully")
                except Exception as e:
                    print(f"⚠ Failed to load model: {e}")
                    print(f"  Reason: Model was likely trained with incompatible PyG version")
                    print(f"  Retraining model from scratch...")
                    original_model, data, is_kg = train_model(args, save_path=model_path)
            else:
                # Train model
                print("Training model from scratch...")
                original_model, data, is_kg = train_model(args, save_path=model_path)
            
            # Store results for this dataset+model combination
            current_results = []
            
            # Split into forget and retain sets
            for unlearn_rate in unlearn_rates:
                print(f"\nUnlearn Rate: {unlearn_rate*100:.0f}%")
                
                forget_mask, retain_mask = get_forget_retain_split(data, unlearn_rate, is_kg, seed=args.seed)

                # Get test mask
                if is_kg:
                    test_mask = retain_mask  # For KG, we use retain set for evaluation
                else:
                    test_mask = data.test_mask

                # Train MIA attack once on original model — shared across all methods
                # so that fe_mia is comparable across methods.
                mia_evaluator = None
                if hasattr(args, 'use_advanced_fe') and args.use_advanced_fe and not is_kg:
                    train_mask = data.train_mask if hasattr(data, 'train_mask') else None
                    if train_mask is not None:
                        print(f"\n  [MIA] Training attack model on original model (shared across methods)...")
                        mia_evaluator = MIAEvaluator(device=device)
                        mia_evaluator.train_attack(original_model, data, is_kg, train_mask, test_mask)
                        print(f"  [MIA] Attack model ready.")
                
                # ✅ Evaluate original trained model (baseline)
                print(f"\n{'='*60}")
                print(f"Model: {model_name}")
                print(f"Evaluating Original Trained Model (Baseline)")
                print(f"{'='*60}")
                
                # Calculate baseline metrics on the trained model
                baseline_fe = calculate_forget_effect(original_model, data, is_kg, forget_mask, retain_mask, device)
                baseline_mu = calculate_model_utility(original_model, data, is_kg, retain_mask, test_mask, forget_mask, device)
                # Also calculate MU on original graph for fair comparison
                baseline_mu_original = calculate_model_utility(original_model, data, is_kg, retain_mask, test_mask, forget_mask, device, use_original_graph=True)
                
                print(f"\nBaseline Results:")
                print(f"  Forget Effect (FE): {baseline_fe:.4f}")
                print(f"  Model Utility (MU - filtered): {baseline_mu:.4f}")
                print(f"  Model Utility (MU - original graph): {baseline_mu_original:.4f}")
                
                # Save baseline results
                baseline_result = {
                    'method': 'baseline_trained',
                    'forget_effect': baseline_fe,
                    'fe_standard': baseline_fe,  # Consistency with other results
                    'model_utility': baseline_mu,  # MU on filtered graph
                    'model_utility_original': baseline_mu_original,  # MU on original graph (fair comparison)
                    'dataset': dataset,
                    'model': model_name,
                    'unlearn_rate': unlearn_rate,
                    'is_kg': is_kg
                }
                
                # Add advanced FE evaluation for baseline
                if hasattr(args, 'use_advanced_fe') and args.use_advanced_fe:
                    print(f"\n  [Advanced FE Evaluation for Baseline]")
                    try:
                        train_mask = data.train_mask if hasattr(data, 'train_mask') else None
                        
                        advanced_results, _ = calculate_forget_effect(
                            original_model, data, is_kg, forget_mask, retain_mask, device,
                            use_multi_method=True,
                            methods=['mia', 'embedding'],
                            train_mask=train_mask,
                            test_mask=test_mask,
                            mia_evaluator=mia_evaluator,
                            verbose=False
                        )
                        
                        # Add MIA results
                        if advanced_results.get('fe_mia') is not None:
                            baseline_result['fe_mia'] = advanced_results['fe_mia']
                            print(f"    MIA-based FE: {advanced_results['fe_mia']:.4f}")
                        
                        # Add Embedding results
                        if advanced_results.get('fe_embedding') is not None:
                            baseline_result['fe_embedding'] = advanced_results['fe_embedding']
                            print(f"    Embedding-based FE: {advanced_results['fe_embedding']:.4f}")
                            
                            # Add detailed embedding metrics
                            if 'embedding_details' in advanced_results:
                                details = advanced_results['embedding_details']
                                baseline_result['forget_retain_distance'] = details.get('forget_retain_distance')
                                baseline_result['forget_random_distance'] = details.get('forget_random_distance')
                                baseline_result['forget_concentration'] = details.get('forget_concentration')
                        
                        # Add Backdoor results (if available)
                        if advanced_results.get('fe_backdoor') is not None:
                            baseline_result['fe_backdoor'] = advanced_results['fe_backdoor']
                            print(f"    Backdoor-based FE: {advanced_results['fe_backdoor']:.4f}")
                            
                    except Exception as e:
                        print(f"    ⚠️  Baseline advanced FE evaluation failed: {e}")
                
                all_results.append(baseline_result)
                current_results.append(baseline_result)
                
                # Evaluate each baseline method
                for method in methods:
                    try:
                        result, unlearned_model = evaluate_method(
                            method, args, original_model, data, is_kg,
                            forget_mask, retain_mask, test_mask, device,
                            dataset, model_name, unlearn_rate,
                            mia_evaluator=mia_evaluator
                        )
                        
                        result.update({
                            'dataset': dataset,
                            'model': model_name,
                            'unlearn_rate': unlearn_rate,
                            'is_kg': is_kg
                        })
                        
                        all_results.append(result)
                        current_results.append(result)
                        
                        # ✅ Save unlearned model
                        if args.save_model:
                            unlearned_model_dir = os.path.join(args.model_dir, 'unlearned')
                            os.makedirs(unlearned_model_dir, exist_ok=True)
                            
                            # Format: {dataset}_{model}_{method}_rate{unlearn_rate}[_suffix]_unlearned.pt
                            rate_str = f"{unlearn_rate:.2f}".replace('.', 'p')
                            if method == 'gnndelete':
                                _gnn_suffix = _gnndelete_cache_suffix(args)
                            elif method == 'grapheraser':
                                _k = getattr(args, 'grapheraser_num_clusters', 4)
                                _gnn_suffix = f"_k{_k}"
                            else:
                                _gnn_suffix = ""
                            unlearned_model_path = os.path.join(
                                unlearned_model_dir,
                                f"{dataset}_{model_name}_{method}_rate{rate_str}{_gnn_suffix}_unlearned.pt"
                            )
                            # GNNDelete: remove old-format caches (e.g. _a0.8_g4.0) so only new hash-format remains
                            if method == 'gnndelete':
                                prefix = os.path.join(unlearned_model_dir, f"{dataset}_{model_name}_gnndelete_rate{rate_str}")
                                for old_path in glob.glob(prefix + "_a*_g*_unlearned.pt"):
                                    try:
                                        os.remove(old_path)
                                        print(f"  Removed old-format cache: {old_path}")
                                    except OSError:
                                        pass
                            # Build checkpoint with all available metrics
                            checkpoint_data = {
                                'model_state_dict': unlearned_model.state_dict(),
                                'method': method,
                                'dataset': dataset,
                                'model': model_name,
                                'unlearn_rate': unlearn_rate,
                                'forget_effect': result['forget_effect'],
                                'model_utility': result['model_utility'],
                                'model_utility_original': result.get('model_utility_original'),  # Fair comparison metric
                                'args': vars(args)
                            }
                            if method == 'gnndelete':
                                checkpoint_data['gnndelete_version'] = 2  # new simplified (DeletionOperator)
                            if method == 'grapheraser' and hasattr(unlearned_model, 'community_to_node'):
                                checkpoint_data['grapheraser_community_to_node'] = unlearned_model.community_to_node
                                checkpoint_data['grapheraser_num_classes'] = unlearned_model.num_classes
                            
                            # Add all FE metrics if available
                            fe_metric_keys = ['fe_standard', 'fe_mia', 'fe_embedding', 'fe_backdoor',
                                            'forget_retain_distance', 'forget_random_distance', 
                                            'forget_concentration']
                            for key in fe_metric_keys:
                                if key in result:
                                    checkpoint_data[key] = result[key]
                            
                            torch.save(checkpoint_data, unlearned_model_path)
                            
                            print(f"  ✓ Unlearned model saved to: {unlearned_model_path}")
                        
                    except Exception as e:
                        print(f"Error evaluating {method}: {e}")
                        import traceback
                        traceback.print_exc()
            
            # Print and save results for this dataset+model combination
            if current_results:
                print(f"\n{'='*80}")
                print(f"RESULTS SUMMARY: {dataset} - {model_name}")
                print(f"{'='*80}")
                
                # Create DataFrame for current results
                current_df = pd.DataFrame(current_results)
                
                # Print results table
                print(f"\n{'Method':<30} {'Forget Effect (FE)':<20} {'Model Utility (MU)':<20}")
                print("-" * 70)
                for _, row in current_df.iterrows():
                    method_display = method_names.get(row['method'], row['method'])
                    print(f"{method_display:<30} {row['forget_effect']:<20.4f} {row['model_utility']:<20.4f}")
                
                # Find best method for this configuration
                best_idx = current_df.assign(
                    combined=lambda x: x['forget_effect'] + x['model_utility']
                )['combined'].idxmax()
                best = current_df.iloc[best_idx]
                print(f"\nBest Method: {method_names.get(best['method'], best['method'])} "
                      f"(FE={best['forget_effect']:.4f}, MU={best['model_utility']:.4f}, "
                      f"Combined={best['forget_effect'] + best['model_utility']:.4f})")
                
                # Save individual results file
                os.makedirs(args.result_dir, exist_ok=True)
                individual_csv = os.path.join(args.result_dir, f'{dataset}_{model_name}_baseline_results.csv')
                current_df.to_csv(individual_csv, index=False)
                print(f"\n✓ Results saved to: {individual_csv}")
                
                individual_json = os.path.join(args.result_dir, f'{dataset}_{model_name}_baseline_results.json')
                with open(individual_json, 'w') as f:
                    json.dump(current_results, f, indent=2)
                print(f"✓ Results saved to: {individual_json}")
                print(f"{'='*80}\n")
    
    # Save results (merge with existing when running partial methods to avoid overwriting)
    results_df = pd.DataFrame(all_results)
    
    # Create results directory
    os.makedirs(args.result_dir, exist_ok=True)
    
    seed_suffix = f"_seed{args.seed}" if hasattr(args, 'seed') else ""
    _prefix = result_prefix if result_prefix else 'baseline_comparison'
    csv_path  = os.path.join(args.result_dir, f'{_prefix}{seed_suffix}.csv')
    json_path = os.path.join(args.result_dir, f'{_prefix}{seed_suffix}.json')
    
    def result_key(r):
        return (r.get('dataset'), r.get('model'), r.get('unlearn_rate'), r.get('method'))
    
    ran_all_methods = (methods_override is None) or (set(methods) == set(all_methods))
    
    # Seed-specific files: full run => overwrite; partial run => merge so we don't wipe other methods
    if seed_suffix and not ran_all_methods and os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                existing_list = json.load(f)
            new_keys = {result_key(r) for r in all_results}
            existing_kept = [r for r in existing_list if result_key(r) not in new_keys]
            merged_list = existing_kept + all_results
            all_results = merged_list
            results_df = pd.DataFrame(merged_list)
            print(f"\nMerged with existing {len(existing_list)} rows; kept {len(existing_kept)} from file, added/updated {len(all_results)} from this run.")
        except Exception as e:
            print(f"\nCould not merge with existing {json_path}: {e}. Writing current results only.")
    elif seed_suffix and not ran_all_methods and os.path.exists(csv_path) and not os.path.exists(json_path):
        try:
            prev_df = pd.read_csv(csv_path)
            new_keys = {result_key(r) for r in all_results}
            prev_df['_key'] = list(zip(
                prev_df['dataset'].astype(str),
                prev_df['model'].astype(str),
                prev_df['unlearn_rate'].astype(float),
                prev_df['method'].astype(str),
            ))
            prev_kept = prev_df[~prev_df['_key'].isin(new_keys)].drop(columns=['_key'])
            results_df = pd.concat([prev_kept, results_df], ignore_index=True)
            all_results = results_df.to_dict('records')
            print(f"\nMerged with existing CSV: kept {len(prev_kept)} rows, added/updated {len(all_results) - len(prev_kept)} from this run.")
        except Exception as e:
            print(f"\nCould not merge with existing CSV: {e}. Writing current results only.")
    elif seed_suffix and ran_all_methods:
        print(f"\nFull run: overwriting seed-specific results (no merge).")
    
    # Save to CSV (with seed in filename for multi-seed runs)
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Save to JSON
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {json_path}")
    
    # Only overwrite "latest" when we ran ALL methods; partial runs must not overwrite
    csv_latest  = os.path.join(args.result_dir, f'{_prefix}.csv')
    json_latest = os.path.join(args.result_dir, f'{_prefix}.json')
    if ran_all_methods:
        results_df.to_csv(csv_latest, index=False)
        with open(json_latest, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Also saved to: {csv_latest} (latest, full run)")
    else:
        print(f"Skipped overwriting {csv_latest} / {json_latest} (partial methods run; use seed-specific files)")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: Baseline Methods Comparison")
    print("="*80)
    
    # Group by method and compute averages
    summary = results_df.groupby('method').agg({
        'forget_effect': ['mean', 'std'],
        'model_utility': ['mean', 'std']
    }).round(4)
    
    print("\nAverage Performance Across All Configurations:")
    print(summary)
    
    # Also show summary by unlearn_rate
    print("\n" + "="*80)
    print("Performance by Unlearn Rate:")
    print("="*80)
    rate_summary = results_df.groupby(['unlearn_rate', 'method']).agg({
        'forget_effect': 'mean',
        'model_utility': 'mean'
    }).round(4)
    print(rate_summary)
    
    # Find best method
    best_fe = results_df.loc[results_df['forget_effect'].idxmax()]
    best_mu = results_df.loc[results_df['model_utility'].idxmax()]
    best_combined = results_df.assign(
        combined=lambda x: x['forget_effect'] + x['model_utility']
    ).loc[lambda x: x['combined'].idxmax()]
    
    print(f"\nBest Forget Effect: {method_names.get(best_fe['method'], best_fe['method'])} "
          f"(FE={best_fe['forget_effect']:.4f}, {best_fe['dataset']}, {best_fe['model']})")
    print(f"Best Model Utility: {method_names.get(best_mu['method'], best_mu['method'])} "
          f"(MU={best_mu['model_utility']:.4f}, {best_mu['dataset']}, {best_mu['model']})")
    print(f"Best Combined: {method_names.get(best_combined['method'], best_combined['method'])} "
          f"(FE+MU={best_combined['combined']:.4f}, {best_combined['dataset']}, {best_combined['model']})")
    
    return results_df


if __name__ == '__main__':

    import sys
    cmdline_args = sys.argv[1:]

    # CUNO's default hyperparameters (K, rho, complexity metric, ordering, mode,
    # beta, lambda, epochs, lr). Kept identical across all datasets and GNN
    # architectures reported in the paper. Command-line flags override these.
    custom_args = [
        '--eval_type', 'baseline_comparison',
        '--device', 'cuda',
        '--num_curricula', '4',
        '--complexity_metric', 'gradient_norm',
        '--curriculum_mode', 'overlapping',
        '--curriculum_order', 'hard_to_easy',
        '--overlap_ratio', '0.5',
        '--unlearn_rate', '0.1',
        '--learning_epochs', '200',
        '--unlearn_epochs', '50',
        '--npo_beta', '0.1',
        '--npo_lambda', '0.5',
        '--gif_iteration', '100',
        '--gif_scale', '1000',
        '--gif_damp', '0.01',
        '--save_model',
        '--load_pretrained'
    ]
    args = get_args_with_custom(custom_args + cmdline_args)

    methods_override = None
    if getattr(args, 'methods', None) is not None:
        methods_override = [m.strip() for m in args.methods.split(',')]

    # Homogeneous datasets are the ones reported in the paper; knowledge-graph
    # experiments are provided for completeness but not part of the main results.
    # Use --no_skip_kg to include FB15k237 and WN18RR.
    skip_kg = True if getattr(args, 'skip_kg', None) is None else args.skip_kg

    results_df = run_baseline_comparison(
        args,
        methods_override=methods_override,
        skip_kg=skip_kg,
    )
    
    print("\n" + "="*80)
    print("Baseline comparison completed!")
    print("="*80)
