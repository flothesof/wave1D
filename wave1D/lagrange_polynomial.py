import numpy as np
from enum import Enum
from scipy import integrate as sp_int
from sympy.integrals import quadrature as syp_int_quad


class PointDistributionType(Enum):
    """
    Definitions of Lagrange polynomials' points distributions.
    """
    EQUALLY_DISTRIBUTED = 0
    GAUSS_LOBATTO = 1


def make_quadrature_formula(order=1, distribution_type=PointDistributionType.GAUSS_LOBATTO):
    """
    Creating points and weight associated to a quadrature formula on the reference segment [0; 1]
    :param order: order of the quadrature formula.
    :param distribution_type: type of points distribution.
    :return: points and weights in an 1D array.
    """
    if distribution_type is PointDistributionType.EQUALLY_DISTRIBUTED:
        w, _ = sp_int.newton_cotes(order)
        return np.linspace(0, 1, order+1), (1.0 / order) * np.ones_like(w) * w
    elif distribution_type is PointDistributionType.GAUSS_LOBATTO:
        p, w = syp_int_quad.gauss_lobatto(order+1, 9)
        return 0.5 * (p + np.ones_like(p)), 0.5 * np.ones_like(w) * w


def eval_lagrange_polynomial(pnts, idx, coord):
    """
    Evaluating a specific Lagrange polynomial at input coordinate.
    :param pnts: points defining the set of Lagrange polynomials.
    :param idx: index of the Lagrange polynomial to evaluate.
    :param coord: input coordinate.
    :return: the value of the Lagrange polynomial at input coordinate.
    """
    return np.prod([(coord - pnts[m]) / (pnts[idx] - pnts[m]) for m in range(len(pnts)) if m != idx])


def eval_lagrange_polynomial_derivative(pnts, idx, coord):
    """
    Evaluating the first derivative of a specific Lagrange polynomial at input coordinate.
    :param pnts: points defining the set of Lagrange polynomials.
    :param idx: index of the Lagrange polynomial to evaluate.
    :param coord: input coordinate.
    :return: the value of the first derivative of a Lagrange polynomial at input coordinate.
    """
    v = 0.0
    for i in range(len(pnts)):
        if i != idx:
            v += np.prod([(coord - pnts[m]) / (pnts[idx] - pnts[m]) for m in range(len(pnts)) if m != idx and m != i])\
                 / (pnts[idx] - pnts[i])
    return v


def eval_lagrange_polynomials(pnts, coords):
    """
    Evaluating every Lagrange polynomials defined from their corresponding points at given coordinates
    :param pnts: points defining the Lagrange polynomuals
    :param coords: set of coordinates to evalute the Lagrange polynomials.
    :return: a matrix such that M_ij is the evaluation if the i-th Lagrange polynomial at the j-th input point.
    """
    result = np.zeros((len(pnts), len(coords)))
    for il in range(len(pnts)):
        for ic in range(len(coords)):
            result[il, ic] = eval_lagrange_polynomial(pnts, il, coords[ic])
    return result


def eval_lagrange_polynomials_derivatives(pnts, coords):
    """
    Evaluating the first derivative of every Lagrange polynomials defined from their corresponding points
     at given coordinates
    :param pnts: points defining the Lagrange polynomuals
    :param coords: set of coordinates to evalute the Lagrange polynomials.
    :return: a matrix such that M_ij is the evaluation if the first derivative of the i-th Lagrange polynomial
        at the j-th input point.
    """
    result = np.zeros((len(pnts), len(coords)))
    for il in range(len(pnts)):
        for ic in range(len(coords)):
            result[il, ic] = eval_lagrange_polynomial_derivative(pnts, il, coords[ic])
    return result
