"""SNA Group 09 - Topic C.10: Making Graph Machine Learning Models Explainable.
Modular Monolith Architecture for Scalable Research Experiments.
"""

from .core import PROJECT_ROOT, DATA_DIR, NOTEBOOKS_DIR, CONFIGS_DIR
from .data import load_dataset, extract_candidate_edges
from .models import GCN, GAT, load_gcn_checkpoint
from .explainers import explain_edges, get_explainer
from .metrics import compute_fidelity, compute_jaccard_at_k, compute_kendall_tau, compute_ground_truth_auc
from .pipelines import run_batch_explaining

__version__ = "0.1.0"

__all__ = [
    'PROJECT_ROOT',
    'DATA_DIR',
    'NOTEBOOKS_DIR',
    'CONFIGS_DIR',
    'load_dataset',
    'extract_candidate_edges',
    'GCN',
    'GAT',
    'load_gcn_checkpoint',
    'explain_edges',
    'get_explainer',
    'compute_fidelity',
    'compute_jaccard_at_k',
    'compute_kendall_tau',
    'compute_ground_truth_auc',
    'run_batch_explaining',
]
