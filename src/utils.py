"""Utility helpers for the Shadow-Structured Cosmology repo."""
import numpy as np
import random

def set_global_seed(seed: int = 42):
    """Set NumPy and Python RNG seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
