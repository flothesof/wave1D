from enum import Enum
import numpy as np
import scipy.sparse
import finite_element_space


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
    def __init__(self, fe_space=finite_element_space.FiniteElementSpace(), assembly_type=AssemblyType.ASSEMBLED):
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


def assemble_mass(density=lambda x: 1.0, fe_space=finite_element_space.FiniteElementSpace(),
                  assembly_type=AssemblyType.ASSEMBLED):
    """
    Assembling mass matrix.
    :param density: function of space variable.
    :param fe_space: input finite element space.
    :param assembly_type: type of assembling procedure.
    :return: instance of FiniteElementOperator class representing the mass operator.
    """
    mass = FiniteElementOperator(fe_space, assembly_type)

    if mass.assembly_type is AssemblyType.ASSEMBLED:
        __apply_mass_assembling(density, fe_space, mass)
    elif mass.assembly_type is AssemblyType.LOCALLY_ASSEMBLED:
        __apply_mass_local_assembling(density, fe_space, mass)
    elif mass.assembly_type is AssemblyType.LUMPED:
        __apply_mass_lumping(density, fe_space, mass)

    return mass


def __apply_mass_assembling(density, fe_space, mass):
    """
    Applying mass assembling procedure on a finite element operator.
    :param density: mass density, function of space variable.
    :param fe_space: finite element space.
    :param mass: finite element operator bearing the assembling procedure.
    """
    raise NotImplementedError()


def __apply_mass_local_assembling(density, fe_space, mass):
    """
    Applying mass local assembling procedure on a finite element operator.
    :param density: mass density, function of space variable.
    :param fe_space: finite element space.
    :param mass: finite element operator bearing the local assembling procedure.
    """
    raise NotImplementedError()


def __apply_mass_lumping(density, fe_space, mass):
    """
    Applying mass lumping procedure on a finite element operator.
    :param density: mass density, function of space variable.
    :param fe_space: finite element space.
    :param mass: finite element operator bearing the lumping procedure.
    """
    raise NotImplementedError()


def assemble_stiffness(param=lambda x: 1.0, fe_space=finite_element_space.FiniteElementSpace(),
                  assembly_type=AssemblyType.ASSEMBLED):
    """
    Assembling stiffness matrix.
    :param param: function of space variable.
    :param fe_space: input finite element space.
    :param assembly_type: type of assembling procedure.
    :return: instance of FiniteElementOperator class representing the stiffness operator.
    """
    stiffness = FiniteElementOperator(fe_space, assembly_type)
    return stiffness




