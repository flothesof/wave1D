from wave1D import finite_element_space as fe_sp
from wave1D import lagrange_polynomial as lag_poly
from wave1D import mesh
from numpy import testing as np_test
import numpy as np
import random


def test_make_finite_element_space():
    """
    Simple construction of a finite element space.
    """
    fe_space = fe_sp.FiniteElementSpace(mesh.Mesh([0.0, 1.0, 4.0]), fe_order=3)
    np_test.assert_equal(fe_space.get_ndof(), 7)
    np_test.assert_equal(fe_space.get_nelem(), 2)
    np_test.assert_almost_equal(fe_space.get_elem_length(0), 1.0)
    np_test.assert_almost_equal(fe_space.get_elem_length(1), 3)
    np_test.assert_equal(fe_space.get_nlocaldof(), 4)
    np_test.assert_equal(fe_space.get_left_idx(), 0)
    np_test.assert_equal(fe_space.get_right_idx(), 6)


def test_get_coord():
    """
    Testing coordinate computation from parametric coordinate of an element.
    """
    random.seed()

    # Mesh with positive coordinates.
    fe_space = fe_sp.FiniteElementSpace(mesh.Mesh([0.0, 1.0, 4.0]), fe_order=3)
    for _ in range(50):
        v = random.random()
        np_test.assert_almost_equal(fe_space.get_coord(1, v), 1.0 + 3.0 * v)

    # Mesh with negative coordinates.
    fe_space = fe_sp.FiniteElementSpace(mesh.Mesh([0.0, -1.0, -4.0]), fe_order=3)
    for _ in range(50):
        v = random.random()
        np_test.assert_almost_equal(fe_space.get_coord(0, v), -4.0 + 3.0 * v)


def test_locals_to_globals():
    """
    Testing extracting of global indexes from an element index.
    """
    fe_space = fe_sp.FiniteElementSpace(mesh.Mesh([0.0, -1.0, -4.0]), fe_order=3)
    np_test.assert_array_equal(fe_space.locals_to_globals(1), [3, 4, 5, 6])


def test_eval_at_quadrature_pnts():
    """
    Testing evaluation of callable at quadrature points.
    """
    fe_space = fe_sp.FiniteElementSpace(mesh.Mesh([0.0, -1.0, -4.0]), quad_order=3,
                                        quad_type=lag_poly.PointDistributionType.EQUALLY_DISTRIBUTED)

    # Testing quadrature point indexes.
    np_test.assert_array_equal(fe_space.eval_at_quadrature_pnts(lambda k, s: k), [0, 1, 2, 3])

    # Testing quadrature points coordinates.
    p, _ = lag_poly.make_quadrature_formula(3, lag_poly.PointDistributionType.EQUALLY_DISTRIBUTED)
    np_test.assert_array_equal(fe_space.eval_at_quadrature_pnts(lambda k, s: s), p)


def test_apply_basis_diag_basis():
    """
    Testing basis^T * diag * basis operation.
    """
    distrib_type = lag_poly.PointDistributionType.EQUALLY_DISTRIBUTED
    fe_space = fe_sp.FiniteElementSpace(mesh.Mesh([0.0, 1.0]), fe_order=2, basis_type=distrib_type,
                                        quad_order=2, quad_type=distrib_type)

    test_diag = [666.0, 666.0, 666.0]
    np_test.assert_array_almost_equal(fe_space.apply_basis_diag_basis(test_diag), np.diag(test_diag))






