"""Information Lagrangian and simple entropy-based diagnostics."""
import numpy as np

def entropy_gaussian(cov: np.ndarray) -> float:
    """Differential entropy of a zero-mean Gaussian with covariance cov (up to constant)."""
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("Covariance must be positive definite.")
    return 0.5 * logdet

def information_lagrangian(S_universe: float, S_observer: float,
                           alpha: float = 1.0, beta: float = 1.0) -> float:
    """Simple scalar L_info = alpha S_universe + beta S_observer."""
    return alpha * S_universe + beta * S_observer
