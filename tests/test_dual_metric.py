import numpy as np
from src.dual_metric import shadow_optical_metric, void_refractivity_tensor, dual_metric

def test_shadow_optical_metric_scaling():
    g = np.eye(4)
    n_g = 1.1
    gtilde = shadow_optical_metric(g, n_g)
    assert np.allclose(gtilde, (n_g**2) * g)

def test_void_refractivity_tensor_shape():
    H = np.eye(3)
    N = void_refractivity_tensor(H)
    assert N.shape == (3, 3)

def test_dual_metric_block_diag():
    g1 = np.eye(2)
    g2 = 2 * np.eye(2)
    G = dual_metric(g1, g2)
    assert G.shape == (4, 4)
