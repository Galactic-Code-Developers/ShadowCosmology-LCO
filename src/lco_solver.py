"""Lexicographic Coherence Operator (LCO) solver (proximal, simplified)."""
import numpy as np
from dataclasses import dataclass
from .utils import set_global_seed

@dataclass
class LCOConfig:
    lambdas: np.ndarray  # shape (4,)
    epsilons: np.ndarray  # shape (4,)
    step_size: float = 1e-3
    max_iters: int = 500
    tol: float = 1e-6
    seed: int = 42

def L1(u: np.ndarray, D: np.ndarray | None = None) -> float:
    """Physical invariants (here: simple quadratic form)."""
    if D is None:
        return float(np.dot(u, u))
    v = D @ u
    return float(np.dot(v, v))

def L2(u: np.ndarray, A: np.ndarray, d: np.ndarray, Ninv: np.ndarray | None = None) -> float:
    """Data fidelity term."""
    r = d - A @ u
    if Ninv is None:
        return float(r @ r)
    return float(r.T @ Ninv @ r)

def L3(u: np.ndarray, Sinv: np.ndarray | None = None) -> float:
    """Statistical coherence term."""
    if Sinv is None:
        return float(u @ u)
    return float(u.T @ Sinv @ u)

def L4(u: np.ndarray) -> float:
    """Interpretive / visualization proxy (simple L2)."""
    return float(u @ u)

def grad_L2(u: np.ndarray, A: np.ndarray, d: np.ndarray, Ninv: np.ndarray | None = None) -> np.ndarray:
    r = d - A @ u
    if Ninv is None:
        return -2.0 * (A.T @ r)
    return -2.0 * (A.T @ (Ninv @ r))

def grad_L1(u: np.ndarray, D: np.ndarray | None = None) -> np.ndarray:
    if D is None:
        return 2.0 * u
    return 2.0 * (D.T @ (D @ u))

def grad_L3(u: np.ndarray, Sinv: np.ndarray | None = None) -> np.ndarray:
    if Sinv is None:
        return 2.0 * u
    return 2.0 * (Sinv @ u)

def grad_L4(u: np.ndarray) -> np.ndarray:
    return 2.0 * u

def prox_Sinv(u: np.ndarray, Sinv: np.ndarray | None, tau_lambda3: float) -> np.ndarray:
    """Simple proximal operator for quadratic L3 term: (I + 2 tau lambda3 S^-1)^{-1} u."""
    if Sinv is None or tau_lambda3 <= 0:
        return u
    n = u.shape[0]
    M = np.eye(n) + 2.0 * tau_lambda3 * Sinv
    return np.linalg.solve(M, u)

def project_tier(u: np.ndarray, value: float, eps: float) -> np.ndarray:
    """Project onto L(u) <= eps by simple rescaling when violated."""
    if value <= eps or value <= 0:
        return u
    scale = np.sqrt(eps / value)
    return u * scale

def lco_solve(A: np.ndarray,
              d: np.ndarray,
              config: LCOConfig,
              Ninv: np.ndarray | None = None,
              Sinv: np.ndarray | None = None,
              D: np.ndarray | None = None,
              u0: np.ndarray | None = None) -> np.ndarray:
    """Lexicographic proximal solver (4 tiers: L1..L4).

    This is a compact, inspectable implementation matching the paper's structure,
    not an optimized production solver.
    """
    set_global_seed(config.seed)
    n = A.shape[1]
    u = np.zeros(n) if u0 is None else u0.copy()

    lam = config.lambdas
    eps = config.epsilons
    tau = config.step_size

    for it in range(config.max_iters):
        # Compute gradients (L1, L2, L4; L3 handled via proximal)
        g1 = grad_L1(u, D)
        g2 = grad_L2(u, A, d, Ninv)
        g4 = grad_L4(u)

        grad = lam[0]*g1 + lam[1]*g2 + lam[3]*g4

        u_new = u - tau * grad
        u_new = prox_Sinv(u_new, Sinv, tau * lam[2])

        # Tiered projection
        v1 = L1(u_new, D)
        u_new = project_tier(u_new, v1, eps[0])
        v2 = L2(u_new, A, d, Ninv)
        u_new = project_tier(u_new, v2, eps[1])
        v3 = L3(u_new, Sinv)
        u_new = project_tier(u_new, v3, eps[2])

        if np.linalg.norm(u_new - u) < config.tol:
            break
        u = u_new

    return u
