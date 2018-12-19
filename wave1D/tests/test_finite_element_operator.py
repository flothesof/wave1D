import numpy as np
from numpy import testing as np_test
from scipy.sparse import SparseEfficiencyWarning
import warnings
from wave1D import finite_element_operator as fe_op


def test_mlt():
    """
    Testing multiplication operation on finite element operators.
    """
    # Testing assembled case.
    op = fe_op.make_from_data(np.diag(np.array([1.0, 2.0, 3.0])), fe_op.AssemblyType.ASSEMBLED)

    u = np.array([1.0, 1.0, 1.0])
    v = np.array([666.0, 666.0, 666.0])
    fe_op.mlt(op, u, v)

    np_test.assert_array_almost_equal(v, [1.0, 2.0, 3.0])

    # Testing lumped case.
    op = fe_op.make_from_data(np.array([1.0, 2.0, 3.0]), fe_op.AssemblyType.LUMPED)

    u = np.array([1.0, 1.0, 1.0])
    v = np.array([666.0, 666.0, 666.0])
    fe_op.mlt(op, u, v)

    np_test.assert_array_almost_equal(v, [1.0, 2.0, 3.0])


def test_mlt_add():
    """
    Testing addition & multiplication operation on finite element operators.
    """
    # Testing assembled case.
    op = fe_op.make_from_data(np.diag(np.array([1.0, 2.0, 3.0])), fe_op.AssemblyType.ASSEMBLED)

    u = np.array([1.0, 1.0, 1.0])
    v = np.array([2.0, 3.0, 4.0])
    fe_op.mlt_add(op, u, v)

    np_test.assert_array_almost_equal(v, [3.0, 5.0, 7.0])

    # Testing lumped case.
    op = fe_op.make_from_data(np.array([1.0, 2.0, 3.0]), fe_op.AssemblyType.LUMPED)

    u = np.array([1.0, 1.0, 1.0])
    v = np.array([2.0, 3.0, 4.0])
    fe_op.mlt_add(op, u, v)

    np_test.assert_array_almost_equal(v, [3.0, 5.0, 7.0])


def test_inv():
    """
    Testing inplace inversion of finite element operators.
    """
    dense_mat = np.array([[4., 2., 1.], [2., 4., 1.], [1., 1., 3.]])

    # Testing assembled case.
    op = fe_op.make_from_data(dense_mat, fe_op.AssemblyType.ASSEMBLED)
    fe_op.inv(op)
    np_test.assert_array_almost_equal(op.data.todense(), np.linalg.inv(dense_mat))

    # Testing lumped case.
    op = fe_op.make_from_data(np.array([1.0, 2.0, 3.0]), fe_op.AssemblyType.LUMPED)
    fe_op.inv(op)
    np_test.assert_array_almost_equal(op.data, [1.0, 0.5, 1.0 / 3.0])


def test_linear_combination():
    """
    Testing combination of finite element operators.
    """
    # Testing assembled case.
    op0 = fe_op.make_from_data(np.diag(np.array([1.0, 1.0, 1.0])), fe_op.AssemblyType.ASSEMBLED)
    op1 = fe_op.make_from_data(np.diag(np.array([2.0, 2.0, 2.0])), fe_op.AssemblyType.ASSEMBLED)
    op2 = fe_op.linear_combination(2.0, op0, 0.5, op1)
    np_test.assert_array_almost_equal(op2.data.todense(), np.diag(np.array([3.0, 3.0, 3.0])))

    # Testing lumped case.
    op0 = fe_op.make_from_data(np.array([1.0, 1.0, 1.0]), fe_op.AssemblyType.LUMPED)
    op1 = fe_op.make_from_data(np.array([2.0, 2.0, 2.0]), fe_op.AssemblyType.LUMPED)
    op2 = fe_op.linear_combination(2.0, op0, 0.5, op1)
    np_test.assert_array_almost_equal(op2.data, [3.0, 3.0, 3.0])

    # Testing lumped and assembled case.
    op0 = fe_op.make_from_data(np.array([1.0, 1.0, 1.0]), fe_op.AssemblyType.LUMPED)
    op1 = fe_op.make_from_data(np.diag(np.array([2.0, 2.0, 2.0])), fe_op.AssemblyType.ASSEMBLED)
    op2 = fe_op.linear_combination(2.0, op0, 0.5, op1)
    np_test.assert_array_almost_equal(op2.data.todense(), np.diag(np.array([3.0, 3.0, 3.0])))


def test_clone():
    """
    Testing cloning of finite element operators.
    """
    # Testing assembled case.
    op0 = fe_op.make_from_data(np.diag(np.array([1.0, 1.0, 1.0])), fe_op.AssemblyType.ASSEMBLED)
    op1 = fe_op.clone(666.0, op0)
    np_test.assert_array_almost_equal(666.0 * op0.data.todense(), op1.data.todense())

    # Testing lumped case.
    op0 = fe_op.make_from_data(np.array([1.0, 1.0, 1.0]), fe_op.AssemblyType.LUMPED)
    op1 = fe_op.clone(666.0, op0)
    np_test.assert_array_almost_equal(666.0 * op0.data, op1.data)


def test_add_value():
    """
    Testing adding value at specific DoF index in finite element operator.
    """
    # Testing assembled case.
    warnings.simplefilter('ignore', SparseEfficiencyWarning)
    op0 = fe_op.make_from_data(np.diag(np.array([1.0, 1.0])), fe_op.AssemblyType.ASSEMBLED)
    fe_op.add_value(op0, 666.0, 0, 1)
    np_test.assert_array_almost_equal(op0.data.todense(), np.array([[1.0, 666.0], [0.0, 1.0]]))
    warnings.resetwarnings()

    # Testing lumped case.
    op0 = fe_op.make_from_data(np.array([1.0, 1.0]), fe_op.AssemblyType.LUMPED)
    fe_op.add_value(op0, 665.0, 1, 1)
    np_test.assert_array_almost_equal(op0.data, np.array([1.0, 666.0]))


def test_spectral_radius():
    """
    Testing computation of spectral radius.
    """
    # Testing on diagonal matrices.
    m = 2.0
    k = 4.0
    Mop = fe_op.make_from_data(np.diag(np.array([m, m, m])), fe_op.AssemblyType.ASSEMBLED)
    Kop = fe_op.make_from_data(np.diag(np.array([k, k, k])), fe_op.AssemblyType.ASSEMBLED)
    radius = fe_op.spectral_radius(Mop, Kop)
    np_test.assert_almost_equal(radius, 2.0)

    # testing with varying mass values
    m = 2.0
    k = 4.0
    Mop = fe_op.make_from_data(np.diag(np.array([3*m, 2*m, m])), fe_op.AssemblyType.ASSEMBLED)
    Kop = fe_op.make_from_data(np.diag(np.array([k, k, k])), fe_op.AssemblyType.ASSEMBLED)
    radius = fe_op.spectral_radius(Mop, Kop)
    np_test.assert_almost_equal(radius, 2.0)


def test_apply_pseudo_elimination():
    """
    Tests pseudo elimination.
    """
    # Testing assembled case.
    op0 = fe_op.make_from_data(np.array([[1.0, 5.0], [2.0, 3.0]]), fe_op.AssemblyType.ASSEMBLED)
    fe_op.apply_pseudo_elimination(op0, 1)
    np_test.assert_array_almost_equal(op0.data.todense(), np.array([[1.0, 0.0], [0.0, 1.0]]))

    # Testing lumped case.
    op0 = fe_op.make_from_data(np.array([1.0, 5.0]), fe_op.AssemblyType.LUMPED)
    fe_op.apply_pseudo_elimination(op0, 1)
    np_test.assert_array_almost_equal(op0.data, np.array([1.0, 1.0]))



