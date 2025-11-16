"""Dual geometry constructs: physical + cognitive metric (toy implementation)."""
import numpy as np

def shadow_optical_metric(g: np.ndarray, n_g: float) -> np.ndarray:
    """Return \tilde{g}_{mu nu} = n_g^2 g_{mu nu}."""
    return (n_g**2) * g

def void_refractivity_tensor(phi_hessian: np.ndarray, c: float = 3e5) -> np.ndarray:
    """Return N_ij = delta_ij + 2/c^2 * Phi_{,ij}."""
    dim = phi_hessian.shape[0]
    eye = np.eye(dim)
    return eye + 2.0 / (c**2) * phi_hessian

def dual_metric(g_spacetime: np.ndarray, g_cognition: np.ndarray) -> np.ndarray:
    """Block-diagonal combination of physical and cognitive metrics."""
    return np.block([[g_spacetime, np.zeros_like(g_spacetime)],
                     [np.zeros_like(g_cognition), g_cognition]])
