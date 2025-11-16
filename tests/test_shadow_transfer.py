import numpy as np
from src.shadow_transfer import shadow_transfer

def test_shadow_zero_when_equal():
    k = np.linspace(0.01, 0.1, 5)
    P = k**-3
    Ts = shadow_transfer(P, P)
    assert np.allclose(Ts, 0.0)
