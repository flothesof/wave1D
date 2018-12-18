import finite_element_operator as fe_op
import finite_element_space as fe_sp
import numpy as np
import scipy.sparse


def assemble_mass(density=lambda x: 1.0, fe_space=fe_sp.FiniteElementSpace(),
                  assembly_type=fe_op.AssemblyType.ASSEMBLED):
    """
    Assembling mass matrix.
    :param density: function of space variable.
    :param fe_space: input finite element space.
    :param assembly_type: type of assembling procedure.
    :return: instance of FiniteElementOperator class representing the mass operator.
    """
    mass = fe_op.FiniteElementOperator(fe_space, assembly_type)

    if mass.assembly_type is fe_op.AssemblyType.ASSEMBLED:
        lil_mass = scipy.sparse.lil_matrix((fe_space.get_ndof(), fe_space.get_ndof()))
        apply_mass_assembling(density, fe_space, lil_mass)
        mass.data = lil_mass.tocsc()
    elif mass.assembly_type is fe_op.AssemblyType.LOCALLY_ASSEMBLED:
        raise NotImplementedError()
    elif mass.assembly_type is fe_op.AssemblyType.LUMPED:
        apply_mass_lumping(density, fe_space, mass.data)

    return mass


def apply_mass_assembling(density, fe_space, mass):
    """
    Applying mass assembling procedure on a finite element operator.
    :param density: mass density, function of space variable.
    :param fe_space: finite element space.
    :param mass: sparse matrix in lil format bearing the assembling procedure.
    """
    if mass.format is 'lil':
        for ie in range(fe_space.get_nelem()):
            local_mass = fe_space.apply_basis_diag_basis(fe_space.eval_at_quadrature_pnts(
                lambda k, s:
                fe_space.get_quadrature_weight(k) * fe_space.get_elem_length(ie) * density(fe_space.get_coord(ie, s))))
            locals_to_globals = fe_space.locals_to_globals(ie)
            mass[np.array(locals_to_globals)[:, None], locals_to_globals] += local_mass
    else:
        raise ValueError("Assembling procedure are expecting lil format sparse matrices.")


def apply_mass_lumping(density, fe_space, mass):
    """
    Applying mass lumping procedure on a finite element operator.
    :param density: mass density, function of space variable.
    :param fe_space: finite element space.
    :param mass: array bearing the lumping procedure.
    """
    for ie in range(fe_space.get_nelem()):
        mass[fe_space.locals_to_globals(ie)] += fe_space.eval_at_quadrature_pnts(
            lambda k, s:
            fe_space.get_quadrature_weight(k) * fe_space.get_elem_length(ie) * density(fe_space.get_coord(ie, s)))