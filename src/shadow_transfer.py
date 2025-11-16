"""Shadow Transfer Function T_s(k) implementation."""
import numpy as np

def shadow_transfer(P_obs: np.ndarray, P_model: np.ndarray) -> np.ndarray:
    """Compute T_s(k) = (P_obs - P_model) / P_model with safe handling of zeros."""
    eps = 1e-12
    denom = np.where(np.abs(P_model) < eps, eps, P_model)
    return (P_obs - P_model) / denom
