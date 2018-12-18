from wave1D import lagrange_polynomial
import numpy as np
from numpy import testing as np_test
import random


def test_make_trapezoid_rule():
    """
    Testing creation of quadrature formula on equally distributed points of order 1.
    """
    p, w = lagrange_polynomial.make_quadrature_formula(1, lagrange_polynomial.PointDistributionType.EQUALLY_DISTRIBUTED)
    np_test.assert_array_almost_equal(p, [0.0, 1.0])
    np_test.assert_array_almost_equal(w, [0.5, 0.5])


def test_make_simpson_1_3_rule():
    """
    Testing creation of quadrature formula on equally distributed points of order 2.
    """
    p, w = lagrange_polynomial.make_quadrature_formula(2, lagrange_polynomial.PointDistributionType.EQUALLY_DISTRIBUTED)
    np_test.assert_array_almost_equal(p, [0.0, 0.5, 1.0])
    np_test.assert_array_almost_equal(w, [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])


def test_make_simpson_3_8_rule():
    """
    Testing creation of quadrature formula on equally distributed points of order 3.
    """
    p, w = lagrange_polynomial.make_quadrature_formula(3, lagrange_polynomial.PointDistributionType.EQUALLY_DISTRIBUTED)
    np_test.assert_array_almost_equal(p, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    np_test.assert_array_almost_equal(w, [1.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0])


def test_make_boole_villarceau():
    """
    Testing creation of quadrature formula on equally distributed points of order 4.
    """
    p, w = lagrange_polynomial.make_quadrature_formula(4, lagrange_polynomial.PointDistributionType.EQUALLY_DISTRIBUTED)
    np_test.assert_array_almost_equal(p, [0.0, 0.25, 0.50, 0.75, 1.0])
    np_test.assert_array_almost_equal(w, [7.0 / 90.0, 32.0 / 90.0, 12.0 / 90.0, 32.0 / 90.0, 7.0 / 90.0])


def test_make_gauss_lobatto_order1():
    """
    Testing creation of gauss-lobatto quadrature formula of order 1.
    """
    p, w = lagrange_polynomial.make_quadrature_formula(1, lagrange_polynomial.PointDistributionType.GAUSS_LOBATTO)
    np_test.assert_array_almost_equal(p, [0.0, 1.0])
    np_test.assert_array_almost_equal(w, [0.5, 0.5])


def test_make_gauss_lobatto_order2():
    """
    Testing creation of gauss-lobatto quadrature formula of order 2.
    """
    p, w = lagrange_polynomial.make_quadrature_formula(2, lagrange_polynomial.PointDistributionType.GAUSS_LOBATTO)
    np_test.assert_array_almost_equal(p, [0.0, 0.5, 1.0])
    np_test.assert_array_almost_equal(w, [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])


def test_make_gauss_lobatto_order3():
    """
    Testing creation of gauss-lobatto quadrature formula of order 3.
    """
    p, w = lagrange_polynomial.make_quadrature_formula(3, lagrange_polynomial.PointDistributionType.GAUSS_LOBATTO)
    np_test.assert_array_almost_equal(p, [0.0, 0.276393202, 0.723606798, 1.0])
    np_test.assert_array_almost_equal(w, [0.0833333333, 0.4166666667, 0.4166666667, 0.0833333333])


def test_eval_lagrange_polynomial():
    """
    Testing evaluation of lagrange polynomials at specific DoFs.
    """
    pnts = [0.0, 0.5, 1.0]

    def l0(s):
        return 2.0 * (s - 0.5) * (s - 1.0)

    def l1(s):
        return -4.0 * s * (s - 1.0)

    def l2(s):
        return 2.0 * s * (s - 0.5)

    random.seed()
    for _ in range(50):
        v = random.random()
        np_test.assert_almost_equal(l0(v), lagrange_polynomial.eval_lagrange_polynomial(pnts, 0, v))
        np_test.assert_almost_equal(l1(v), lagrange_polynomial.eval_lagrange_polynomial(pnts, 1, v))
        np_test.assert_almost_equal(l2(v), lagrange_polynomial.eval_lagrange_polynomial(pnts, 2, v))


def test_eval_lagrange_polynomial_derivative():
    """
    Testing evaluation of derivative of lagrange polynomials at specific DoFs.
    """
    pnts = [0.0, 0.5, 1.0]

    def dl0(s):
        return 4.0 * s - 3.0

    def dl1(s):
        return -8.0 * s + 4.0

    def dl2(s):
        return 4.0 * s - 1.0

    random.seed()
    for _ in range(50):
        v = random.random()
        np_test.assert_almost_equal(dl0(v), lagrange_polynomial.eval_lagrange_polynomial_derivative(pnts, 0, v))
        np_test.assert_almost_equal(dl1(v), lagrange_polynomial.eval_lagrange_polynomial_derivative(pnts, 1, v))
        np_test.assert_almost_equal(dl2(v), lagrange_polynomial.eval_lagrange_polynomial_derivative(pnts, 2, v))


def test_eval_lagrange_polynomials():
    """
    Testing evaluation of every lagrange polynomials at every input DoFs.
    """
    pnts = [0.0, 0.5, 1.0]

    def l0(s):
        return 2.0 * (s - 0.5) * (s - 1.0)

    def l1(s):
        return -4.0 * s * (s - 1.0)

    def l2(s):
        return 2.0 * s * (s - 0.5)

    test_coords = np.random.rand(5, 1)
    result = lagrange_polynomial.eval_lagrange_polynomials(pnts, test_coords)

    np_test.assert_array_equal(result.shape, [len(pnts), len(test_coords)])
    for iv in range(len(test_coords)):
        np_test.assert_almost_equal(l0(test_coords[iv]), result[0, iv])
        np_test.assert_almost_equal(l1(test_coords[iv]), result[1, iv])
        np_test.assert_almost_equal(l2(test_coords[iv]), result[2, iv])


def test_eval_lagrange_polynomials_derivatives():
    """
    Testing evaluation of the first derivative of every lagrange polynomials at every input DoFs.
    """
    pnts = [0.0, 0.5, 1.0]

    def dl0(s):
        return 4.0 * s - 3.0

    def dl1(s):
        return -8.0 * s + 4.0

    def dl2(s):
        return 4.0 * s - 1.0

    test_coords = np.random.rand(5, 1)
    result = lagrange_polynomial.eval_lagrange_polynomials_derivatives(pnts, test_coords)

    np_test.assert_array_equal(result.shape, [len(pnts), len(test_coords)])
    for iv in range(len(test_coords)):
        np_test.assert_almost_equal(dl0(test_coords[iv]), result[0, iv])
        np_test.assert_almost_equal(dl1(test_coords[iv]), result[1, iv])
        np_test.assert_almost_equal(dl2(test_coords[iv]), result[2, iv])







