"""Data module for loading graphs, candidate edges, and preprocessing."""

from .loader import load_dataset
from .candidate_edges import extract_candidate_edges

__all__ = [
    'load_dataset',
    'extract_candidate_edges',
]
