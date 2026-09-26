# CUNO: Curriculum-Based Graph Unlearning to Mitigate Utility Collapse at High Deletion Ratios

This repository contains the reference implementation of **CUNO** and every baseline used in our submission. CUNO combines a curriculum that orders the forget set by unlearning difficulty with a distribution-level negative preference optimization (NPO) objective, and is designed to preserve model utility under mass deletion (up to 50% of training labels).

The code reproduces the main tables and figures reported in the paper: model utility across deletion rates, the joint forget-effect vs. utility scatter, ablation studies, and hyperparameter sensitivity.

---

## What is in this release

```
cuno-code/
├── learning.py                   # Pre-training of GNNs
├── unlearning.py                 # Dispatcher that calls the requested unlearning method
├── curriculum.py                 # Complexity metrics and curriculum construction
├── evaluation.py                 # Model-utility metrics
├── fe_evaluation_methods.py      # Forget-effect metrics: FE-Standard, FE-MIA, FE-Embedding, FE-Backdoor
├── my_parser.py                  # Command-line argument parser
├── utils.py                      # Seed and helper utilities
├── data/
│   ├── data_loader.py            # Dataset loading (Cora, CiteSeer, PubMed, and additional homophilous / heterophilous datasets)
│   └── __init__.py
├── gnn_model/
│   ├── homogeneous_models.py     # GCN, GAT, GraphSAGE
│   └── knowledge_graph_models.py # RGCN, CompGCN (kept for completeness; the paper reports homogeneous graphs only)
├── unlearning_methods/
│   ├── cuno_ours.py              # CUNO (curriculum + distribution-level NPO); also `curriculum_ga`, `npo_ga`
│   ├── retrain.py                # Retrain-from-scratch reference
│   ├── gradient_ascent.py        # Gradient ascent baseline
│   ├── gif.py                    # GIF (Wu et al., 2023)
│   ├── gnndelete.py              # GNNDelete (Cheng et al., 2023)
│   ├── grapheraser.py            # GraphEraser (Chen et al., 2022)
│   ├── inpo.py                   # INPO
│   └── etr.py                    # ETR
├── run_evaluation_baseline.py    # End-to-end evaluation entry point used to produce the main results
├── scripts/
│   └── run_evaluation_baseline.sh
├── requirements.txt
├── LICENSE
└── README.md                     # this file
```

---

## Requirements

- **Python** 3.9 (tested)
- **CUDA** 11.8 (any recent GPU with 6+ GB memory is enough for the datasets studied)
- **PyTorch** 1.12 or newer
- **PyTorch Geometric** 2.0 or newer, with `torch-scatter` and `torch-sparse`
- See `requirements.txt` for the full list

CPU-only execution works but is slow; the CUDA path is exercised by every experiment we report.

---

## Setup

### Option A: Conda (recommended)

```bash
conda create -n cuno python=3.9 -y
conda activate cuno

# Install PyTorch matching your CUDA version (11.8 shown here)
pip install torch==2.0.0 torchvision --index-url https://download.pytorch.org/whl/cu118

# Install PyG and its C++ extensions matching torch/CUDA
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# The rest
pip install -r requirements.txt
```

### Option B: Plain pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Then install matching torch, torch-geometric, torch-scatter and torch-sparse
# following https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
```

If you see a `GLIBCXX_3.4.29` runtime error, add the conda env's `lib` directory to `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

---

## Datasets

Datasets download automatically the first time they are used. All datasets are cached under `data/<name>/`.

| Dataset | Task | Notes |
|---|---|---|
| `Cora`, `CiteSeer`, `PubMed` | Node classification | Standard Planetoid splits (used in the paper) |
| `AmazonComputers`, `CoauthorCS`, `Actor` | Node classification | Deterministic 60/20/20 splits |
| `RomanEmpire`, `AmazonRatings` | Node classification | Official 10 fixed 50/25/25 splits from Platonov et al. (ICLR 2023); split 0 is used |
| `AmazonPhoto` | Node classification | Shchur et al. (2018) protocol: 20 labelled nodes per class for training, 30 per class for validation |
| `FB15k237`, `WN18RR` | Knowledge-graph completion | Loaders provided; the paper does not report these experiments |

---

## Quick start (single configuration, one seed, one dataset)

Train a base model and run one round of every unlearning method on it:

```bash
# 1. Train the base GNN once (reused by every subsequent unlearning method)
python learning.py --dataset Cora --gnn_model GCN --learning_epochs 200

# 2. Run every unlearning method on that trained model, at a single deletion rate
python run_evaluation_baseline.py \
  --only_this_config --dataset Cora --gnn_model GCN --seed 42 \
  --unlearn_rates "0.2" \
  --load_pretrained --skip_kg
```

Results land in `results/` as one CSV per (dataset, model) plus a merged `baseline_comparison_seed42.csv`. Typical wall-clock for the command above is under two minutes on a single L4-class GPU.

---

## Reproduce the main table (Section 4.2 in the paper)

The main experiment is a grid over three datasets, three GNN architectures, seven deletion rates and five random seeds. All hyperparameters are the same across datasets and architectures (no per-dataset tuning); the exact values used are the ones baked into `run_evaluation_baseline.py`.

```bash
for seed in 42 123 456 789 2024; do
  python run_evaluation_baseline.py --seed $seed --skip_kg --load_pretrained
done
```

Each seed produces `results/baseline_comparison_seed${seed}.csv`. Averaging over the five seeds reproduces the numbers reported in the paper.

The shell wrapper `scripts/run_evaluation_baseline.sh` runs the same sweep with optional `--parallel` mode and log files under `logs/`.

---

## Ablation variants shipped with CUNO

The paper's ablation study in Section 4.4 uses three variants that share the same file as `full_method`:

| Paper name (`--unlearn_method`) | Function in `unlearning_methods/cuno_ours.py` | What it removes |
|---|---|---|
| `full_method` (CUNO) | `full_method` | Nothing; this is the full method |
| `curriculum_ga` (CUNO w/o NPO) | `curriculum_gradient_ascent` | Removes the NPO objective, keeps curriculum |
| `npo_ga` (CUNO w/o Curriculum) | `npo_ga` | Removes curriculum staging, keeps distribution-level NPO |
| `full_method_kl` (CUNO-KL) | `full_method_kl` | Replaces the log-ratio NPO forget loss with a KL variant |

These variants are wired through `unlearning.py` and can be selected via the standard `--unlearn_method` flag. They are provided so that the ablation numbers in the paper are directly reproducible.

## Method configuration reference

CUNO's key hyperparameters (default values reported in the paper):

| Argument | Default | Meaning |
|---|---|---|
| `--num_curricula` | `4` | Number of curriculum stages $K$ |
| `--curriculum_mode` | `overlapping` | Adjacent stages share a fraction of samples |
| `--overlap_ratio` | `0.5` | Overlap fraction $\rho$ between adjacent stages |
| `--curriculum_order` | `hard_to_easy` | Ordering direction |
| `--complexity_metric` | `gradient_norm` | Per-sample forget-gradient norm (M-GN); other options in `curriculum.py` |
| `--npo_beta` | `0.1` | NPO preference strength $\beta$ |
| `--npo_lambda` | `0.5` | Forget vs. retain weight $\lambda$ |
| `--npo_temperature` | `1.0` | Softmax temperature $\tau$ |
| `--unlearn_epochs` | `50` | Total unlearning budget $T$ (split across stages) |
| `--unlearn_lr` | `0.1` | Unlearning learning rate |

Available methods for `--unlearn_method` (used by `unlearning.py`):

`retrain`, `gradient_ascent`, `curriculum_ga`, `npo_ga`, `full_method` (CUNO), `gif`, `gnndelete`, `grapheraser`, `inpo`, `etr`.

`run_evaluation_baseline.py` calls all of them by default. Use `--methods "m1,m2,..."` to restrict the set.

---

## Evaluation metrics

Two families of metrics are reported by every experiment:

- **Model Utility (MU).** Test-set classification accuracy after unlearning, plus utility drop $\Delta U(\gamma)$ against the original model.
- **Forget Effect (FE).** Two complementary measurements with different status:
  - `fe_mia` is the calibrated forget-effect metric of record. It is computed by `MIAEvaluator` in `fe_evaluation_methods.py`: a shadow membership-inference classifier is trained once on the original model to distinguish members from non-members, then applied without modification to every unlearned model. Closeness to the retrain reference is the signal we compare.
  - `fe_standard` = $1 - \text{Acc}(\mathcal{S}_d)$ is reported as a descriptive diagnostic only. It is confounded on graphs by neighbourhood-driven label inference (the Retrain reference itself attains only 0.11–0.47 across our datasets).

Auxiliary FE measurements (`fe_embedding`, `fe_backdoor`, `forget_retain_distance`, `forget_random_distance`, `forget_concentration`) are also written to the output CSVs and are described in the source of `fe_evaluation_methods.py`.

---

## Output layout

Everything a run produces is under `results/` (created if missing):

```
results/
├── <Dataset>_<Model>_baseline_results.csv    # per (dataset, model) rollup
├── baseline_comparison_seed<seed>.csv        # merged over datasets and models for one seed
└── baseline_comparison_seed<seed>.json       # same content in JSON
```

Model checkpoints are cached under `stored_model/<Dataset>_<Model>_trained.pt`. Passing `--load_pretrained` reuses them across seeds and methods; passing `--load_cached_unlearned` reuses previously computed unlearned models where available.

---

## Notes on scope and reproducibility

- Every configuration uses identical hyperparameters across datasets and architectures. There is no per-dataset tuning.
- Base models are trained once per (dataset, architecture) and shared across the five random seeds; only the forget-set sampling and unlearning optimisation are seed-dependent.
- The paper's main results are node-classification-only on `Cora`, `CiteSeer` and `PubMed`. The additional dataset loaders shipped here (`AmazonComputers`, `CoauthorCS`, `Actor`, `RomanEmpire`, `AmazonRatings`, `AmazonPhoto`) are provided for extended experimentation; they are not part of the reported main table.
- Certified graph unlearning methods (Chien et al. 2022; Wu et al. 2023; Yi and Wei 2025; Dong et al. 2024) are discussed in the paper but not included as baselines here, because their certificates require convexity assumptions that do not hold for the non-convex GCN/GAT/GraphSAGE objectives studied.

---

## License

Released under the MIT License (see `LICENSE`).

---

## Contact

Authors and affiliations are omitted for double-blind review. Please contact us through the paper's OpenReview thread.
