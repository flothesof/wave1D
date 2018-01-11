import numpy as np
from wave1D import functional


def test_ricker():
    """
    Tests Ricker wavelet.
    """
    vals = functional.ricker(x=np.array([1.]), f=1.)

    assert vals.size == 1


def test_gated_cosine():
    """
    Tests gated cosine functional.
    """
    vals = functional.gated_cosine(x=np.array([1.]), f=1., sigma=0.1)

    assert vals.size == 1
