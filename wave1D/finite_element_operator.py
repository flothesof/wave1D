from enum import Enum
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import finite_element_space as fe_sp


class AssemblyType(Enum):
    """
    Definitions of mass types
    """
    ASSEMBLED = 0
    LOCALLY_ASSEMBLED = 1
    LUMPED = 2


class FiniteElementOperator:
    """
    Definition of finite element operators.
    """
    def __init__(self, fe_space=fe_sp.FiniteElementSpace(), assembly_type=AssemblyType.ASSEMBLED):
        """
        Constructor of operator.
        :param fe_space: input finite element space.
        :param assembly_type: type of assembling procedure.
        """
        self.assembly_type = assembly_type
        if self.assembly_type is AssemblyType.ASSEMBLED:
            self.data = scipy.sparse.csc_matrix((fe_space.get_ndof(), fe_space.get_ndof()))
        elif self.assembly_type is AssemblyType.LOCALLY_ASSEMBLED:
            self.data = np.zeros(fe_space.get_nelem() * fe_space.get_nlocaldof())
        elif self.assembly_type is AssemblyType.LUMPED:
            self.data = np.zeros(fe_space.get_ndof())


def make_from_data(data, assembly_type):
    """
    Creating a finite element operator from internal data and an assembly type.
    :param: data: data defining the operator.
    :param: assembly_type: type of assembling procedure.
    """
    operator = FiniteElementOperator()
    operator.assembly_type = assembly_type

    if assembly_type is AssemblyType.ASSEMBLED:
        operator.data = scipy.sparse.csc_matrix(data)
    elif assembly_type is AssemblyType.LOCALLY_ASSEMBLED:
        operator.data = data
    elif assembly_type is AssemblyType.LUMPED:
        operator.data = data

    return operator


def mlt(A, u, v):
    """
    Performing operation : v <- A * u
    :param A: input finite element operator.
    :param u: input vector.
    :param v: output vector.
    """
    if A.assembly_type is AssemblyType.ASSEMBLED:
        v[:] = A.data.dot(u)
    elif A.assembly_type is AssemblyType.LOCALLY_ASSEMBLED:
        raise NotImplementedError()
    elif A.assembly_type is AssemblyType.LUMPED:
        v[:] = A.data * u


def mlt_add(A, u, v):
    """
    Performing operation v <- v + A * u
    :param A: input finite element operator.
    :param u: input vector.
    :param v: output vector.
    """
    if A.assembly_type is AssemblyType.ASSEMBLED:
        v[:] += A.data.dot(u)
    elif A.assembly_type is AssemblyType.LOCALLY_ASSEMBLED:
        raise NotImplementedError()
    elif A.assembly_type is AssemblyType.LUMPED:
        v[:] += A.data * u


def inv(A):
    """
    Inverting in-place an operator: A <- A^{-1}
    :param A: input finite element operator.
    """
    if A.assembly_type is AssemblyType.ASSEMBLED:
        A.data = scipy.sparse.linalg.inv(A.data)
    elif A.assembly_type is AssemblyType.LOCALLY_ASSEMBLED:
        raise ValueError("Cannot invert inplace locally assembled operator.")
    elif A.assembly_type is AssemblyType.LUMPED:
        A.data[:] = 1.0 / A.data[:]


def linear_combination(a, A, b, B):
    """
    Building an operator from combining two different operators, i.e. C <- a * A + b * B
    :param a: input coefficient for A operator.
    :param A: input finite element operator.
    :param b: input coefficient for B operator.
    :param B: input finite element operator.
    :return: C s.t. C <- a * A + b * B
    """
    if A.assembly_type is AssemblyType.ASSEMBLED and B.assembly_type is AssemblyType.ASSEMBLED:
        return make_from_data(a * A.data + b * B.data, AssemblyType.ASSEMBLED)
    elif A.assembly_type is AssemblyType.LOCALLY_ASSEMBLED and B.assembly_type is AssemblyType.LOCALLY_ASSEMBLED:
        raise NotImplementedError()
    elif A.assembly_type is AssemblyType.LUMPED and B.assembly_type is AssemblyType.LUMPED:
        return make_from_data(a * A.data + b * B.data, AssemblyType.LUMPED)
    else:
        raise NotImplementedError()


def clone(a, A):
    """
    Cloning input operator (deep copy).
    :param a: input coefficient for A operator.
    :param A: input operator to be cloned.
    :return: Deep copy of a * A.
    """
    return make_from_data(a * A.data, A.assembly_type)


def spectral_radius(M, K):
    """
    Extract absolute value of largest eigen value of generalized eigen problem: K * u = l * M * u.
    :param M: invertible finite element operator.
    :param K: input finite element operator.
    """
    if K.assembly_type is AssemblyType.ASSEMBLED:
        Minv = clone(1.0, M)
        inv(Minv)
        if Minv.assembly_type is AssemblyType.LUMPED:
            Minv = scipy.sparse.dia_matrix((Minv.data, [0]), shape=K.data.shape)
        return scipy.sparse.linalg.eigs(Minv.data * K.data, k=1, which='LM', return_eigenvectors=False)
    else:
        raise NotImplementedError()


def add_value(A, a, i, j):
    """
    Adding value to an element of an operator A_ij <- A_ij + a
    :param A: input finite element operator to be modified.
    :param a: input value to add.
    :param i: row index.
    :param j: column index.
    """
    if A.assembly_type is AssemblyType.ASSEMBLED:
        A.data[i, j] += a
    elif A.assembly_type is AssemblyType.LOCALLY_ASSEMBLED:
        raise NotImplementedError()
    elif A.assembly_type is AssemblyType.LUMPED:
        if i != j:
            raise ValueError("Cannot access extra-diagonal element in LUMPED operator.")
        else:
            A.data[i] += a


def apply_pseudo_elimination(A, i):
    """
    Applying pseudo elimination of an operator on a specific row.
    :param A: input finite element operator to be modified.
    :param i: row (equivalently column) index.
    """
    if A.assembly_type is AssemblyType.ASSEMBLED:
        I, J, _ = scipy.sparse.find(A.data)
        slice = (I == i) | (J == i)
        A.data[I[slice], J[slice]] = 0
        A.data[i, i] = 1
    elif A.assembly_type is AssemblyType.LOCALLY_ASSEMBLED:
        raise NotImplementedError()
    elif A.assembly_type is AssemblyType.LUMPED:
        A.data[i] = 1.0




