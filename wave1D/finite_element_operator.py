from enum import Enum
import numpy as np
import scipy.sparse
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
            self.data = scipy.sparse.csr_matrix((fe_space.get_ndof(), fe_space.get_ndof()))
        elif self.assembly_type is AssemblyType.LOCALLY_ASSEMBLED:
            self.data = np.zeros(fe_space.get_nelem() * fe_space.get_nlocaldof())
        elif self.assembly_type is AssemblyType.LUMPED:
            self.data = np.zeros(fe_space.get_ndof())


def mlt(A, u, v):
    """
    Performing operation : v <- A * u
    :param A: input finite element operator.
    :param u: input vector.
    :param v: output vector.
    """
    raise NotImplementedError()


def mlt_add(A, u, v):
    """
    Performing operation v <- v + A * u
    :param A: input finite element operator.
    :param u: input vector.
    :param v: output vector.
    """
    raise NotImplementedError()


def inv(A):
    """
    Inverting in-place an operator: A <- A^{-1}
    :param A: input finite element operator.
    """
    raise NotImplementedError()


def linear_combination(a, A, b, B):
    """
    Building an operator from combining two different operators, i.e. C <- a * A + b * B
    :param a: input coefficient for A operator.
    :param A: input finite element operator.
    :param b: input coefficient for B operator.
    :param B: input finite element operator.
    :return: C s.t. C <- a * A + b * B
    """
    raise NotImplementedError()


def clone(a, A):
    """
    Cloning input operator (deep copy).
    :param a: input coefficient for A operator.
    :param A: input operator to be cloned.
    :return: Deep copy of a * A.
    """
    raise NotImplementedError()


def spectral_radius(M, K):
    """
    Extract absolute value of largest eigen value of generalized eigen problem: K * u = l * M * u.
    :param M: invertible finite element operator.
    :param K: input finite element operator.
    """
    raise NotImplementedError()


def add_value(A, a, i, j):
    """
    Adding value to an element of an operator A_ij <- A_ij + a
    :param A: input finite element operator to be modified.
    :param a: input value to add.
    :param i: row index.
    :param j: column index.
    """
    raise NotImplementedError()


def apply_pseudo_elimination(A, i):
    """
    Applying pseudo elimination of an operator on a specific row.
    :param A: input finite element operator to be modified.
    :param i: row index.
    """
    raise NotImplementedError()




