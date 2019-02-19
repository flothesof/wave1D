import numpy as np


def ricker(x, f):
    """
    Definition of a Ricker wavelet.
    :param x: input coordinates.
    :param f: principal frequency.
    :return: Ricker wavelet evaluated in input coordinates.
    """
    return (1.0 - 2.0 * np.pi**2 * f**2 * x**2) * np.exp(-np.pi**2 * f**2 * x**2)


def gated_cosine(x, f, sigma):
    """
    Definition of a gated cosine.
    :param x: input coordinates.
    :param f: principal frequency.
    :param sigma: gaussian gate standard variation.
    :return: gated cosine function evaluated in input coordinates.
    """
    return np.exp(-(x / sigma)**2) * np.cos(2.0 * np.pi * f *x)
