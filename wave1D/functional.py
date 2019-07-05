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


def hanning(x, f, n):
    """
    Definition of a hanning signal.
    :param x: input coordinates.
    :param f: principal frequency.
    :param n: number of cycles in the signal.
    :return: values of the hanning function evaluated in input coordinates.
    """
    xmax = n / f
    val = np.zeros_like(x)
    for i, v in enumerate(x):
        if v >= 0.0 and v <= xmax:
            val[i] = 0.5 * np.sin(2.0 * np.pi * f * v) * (1.0 - np.cos(2.0 * np.pi * f * v / n))
    return val


def heaviside(x, eps):
    """
    Definition of the regularized heavised function.
    :param x: function input argument.
    :param eps: regularization coefficient.
    :return: the evaluation of the heaviside function
    """
    return 1.0 / (1.0 + np.exp(-x / eps))
