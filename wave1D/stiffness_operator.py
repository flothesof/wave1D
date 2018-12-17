import finite_element_operator as fe_op
import finite_element_space as fe_sp


def assemble_stiffness(param=lambda x: 1.0, fe_space=fe_sp.FiniteElementSpace(),
                       assembly_type=fe_op.AssemblyType.ASSEMBLED):
    """
    Assembling stiffness matrix.
    :param param: function of space variable.
    :param fe_space: input finite element space.
    :param assembly_type: type of assembling procedure.
    :return: instance of FiniteElementOperator class representing the stiffness operator.
    """
    stiffness = fe_op.FiniteElementOperator(fe_space, assembly_type)

    if stiffness.assembly_type is fe_op.AssemblyType.ASSEMBLED:
        __apply_stiffness_assembling(param, fe_space, stiffness)
    elif stiffness.assembly_type is fe_op.AssemblyType.LOCALLY_ASSEMBLED:
        raise NotImplementedError()
    elif stiffness.assembly_type is fe_op.AssemblyType.LUMPED:
        raise NotImplementedError()

    return stiffness


def __apply_stiffness_assembling(param, fe_space, stiffness):
    """
    Applying assembling procedure on stiffness operator.
    :param param: input parameter.
    :param fe_space: finite element space.
    :param stiffness: finite element operator bearing the assembling procedure.
    """
    raise NotImplementedError()