import finite_element_operator as fe_op
import finite_element_space as fe_sp


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
        apply_mass_assembling(density, fe_space, mass)
    elif mass.assembly_type is fe_op.AssemblyType.LOCALLY_ASSEMBLED:
        raise NotImplementedError()
    elif mass.assembly_type is fe_op.AssemblyType.LUMPED:
        apply_mass_lumping(density, fe_space, mass)

    return mass


def apply_mass_assembling(density, fe_space, mass):
    """
    Applying mass assembling procedure on a finite element operator.
    :param density: mass density, function of space variable.
    :param fe_space: finite element space.
    :param mass: finite element operator bearing the assembling procedure.
    """
    for ie in range(fe_space.get_nelem()):
        locals_to_globals = fe_space.locals_to_globals(ie)
        mass.data[locals_to_globals, locals_to_globals] += fe_space.apply_basis_diag_basis(fe_space.eval_at_quadrature_pnts(
            lambda k, s:
            fe_space.get_quadrature_weight(k) * fe_space.get_elem_length(ie) * density(fe_space.get_coord(ie, s))))


def apply_mass_lumping(density, fe_space, mass):
    """
    Applying mass lumping procedure on a finite element operator.
    :param density: mass density, function of space variable.
    :param fe_space: finite element space.
    :param mass: finite element operator bearing the lumping procedure.
    """
    raise NotImplementedError()
