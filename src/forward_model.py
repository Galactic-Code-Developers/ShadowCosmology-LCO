"""Forward model A u + n used in the paper (linear, simplified)."""
import numpy as np

def apply_forward_operator(A: np.ndarray, u: np.ndarray, noise: np.ndarray | None = None) -> np.ndarray:
    """Compute d = A u + n.

    Parameters
    ----------
    A : ndarray
        Forward operator matrix.
    u : ndarray
        True sky / field (vectorized).
    noise : ndarray or None
        Optional noise vector.

    Returns
    -------
    ndarray
        Simulated data vector d.
    """
    d = A @ u
    if noise is not None:
        d = d + noise
    return d
