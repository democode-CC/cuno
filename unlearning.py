"""
Unlearning Methods for Graph Neural Networks
Includes: Retrain, Gradient Ascent, Curriculum Unlearning, NPO, and Full Method
"""

import os
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import networkx as nx
from copy import deepcopy

from learning import create_model, train_homogeneous, train_knowledge_graph
from data import get_forget_retain_split

from curriculum import CurriculumDesigner, ComplexityCalculator

from utils import set_unlearn_seed
from unlearning_methods.retrain import retrain_baseline
from unlearning_methods.gradient_ascent import gradient_ascent_baseline
from unlearning_methods.cuno_ours import curriculum_gradient_ascent
from unlearning_methods.cuno_ours import npo_gradient_ascent
from unlearning_methods.cuno_ours import full_method
from unlearning_methods.cuno_ours import full_method_kl
from unlearning_methods.cuno_ours import npo_ga as npo_ga_fn
from unlearning_methods.gif import gif_unlearn
from unlearning_methods.gnndelete import gnndelete_unlearn
from unlearning_methods.grapheraser import grapheraser_unlearn
from unlearning_methods.inpo import inpo
from unlearning_methods.etr import etr






def unlearn(args, model, data, is_kg, forget_mask, retain_mask):
    """
    Main unlearning function that dispatches to appropriate method
    
    Args:
        args: Arguments
        model: Trained model
        data: Graph data
        is_kg: Whether it's a knowledge graph
        forget_mask: Mask for forget set
        retain_mask: Mask for retain set
        
    Returns:
        model: Unlearned model
    """
    if args.unlearn_method == 'retrain':
        return retrain_baseline(args, data, is_kg, retain_mask)
    elif args.unlearn_method == 'gradient_ascent':
        return gradient_ascent_baseline(args, model, data, is_kg, forget_mask, retain_mask)
    elif args.unlearn_method == 'curriculum_ga':
        return curriculum_gradient_ascent(args, model, data, is_kg, forget_mask, retain_mask)
    elif args.unlearn_method == 'npo_ga':
        return npo_ga_fn(args, model, data, is_kg, forget_mask, retain_mask)
    elif args.unlearn_method == 'npo_ga_kl':
        return npo_gradient_ascent(args, model, data, is_kg, forget_mask, retain_mask)
    elif args.unlearn_method == 'full_method':
        return full_method(args, model, data, is_kg, forget_mask, retain_mask)
    elif args.unlearn_method == 'full_method_kl':
        return full_method_kl(args, model, data, is_kg, forget_mask, retain_mask)
    elif args.unlearn_method == 'gif':
        return gif_unlearn(args, model, data, is_kg, forget_mask, retain_mask)
    elif args.unlearn_method == 'gnndelete':
        return gnndelete_unlearn(args, model, data, is_kg, forget_mask, retain_mask)
    elif args.unlearn_method == 'grapheraser':
        return grapheraser_unlearn(args, model, data, is_kg, forget_mask, retain_mask)
    elif args.unlearn_method == 'inpo':
        return inpo(args, model, data, is_kg, forget_mask, retain_mask)
    elif args.unlearn_method == 'etr':
        return etr(args, model, data, is_kg, forget_mask, retain_mask)
    else:
        raise ValueError(f"Unknown unlearning method: {args.unlearn_method}")



