import numpy as np
from src.lco_solver import LCOConfig, lco_solve, L1, L2, L3

def test_lco_basic_convergence():
    n = 10
    A = np.eye(n)
    d = np.ones(n)
    config = LCOConfig(
        lambdas=np.array([1e4, 1e2, 10.0, 1.0]),
        epsilons=np.array([1e2, 1e2, 1e2, 1e6]),
        step_size=1e-3,
        max_iters=200,
        tol=1e-6,
        seed=42,
    )
    u_hat = lco_solve(A, d, config)
    assert np.all(np.isfinite(u_hat))
    # Physical invariant (here simple L2) should not blow up
    assert L1(u_hat, None) < 1e4
