import wave1D.finite_element_operator as fe_op
import wave1D.finite_element_space as fe_sp
import numpy as np
import scipy.sparse


def assemble_stiffness(fe_space, param=lambda x: 1.0, assembly_type=fe_op.AssemblyType.ASSEMBLED):
    """
    Assembling stiffness matrix.
    :param fe_space: input finite element space.
    :param param: function of space variable.
    :param assembly_type: type of assembling procedure.
    :return: instance of FiniteElementOperator class representing the stiffness operator.
    """
    stiffness = fe_op.FiniteElementOperator(fe_space, assembly_type)

    if stiffness.assembly_type is fe_op.AssemblyType.ASSEMBLED:
        lil_stiffness = scipy.sparse.lil_matrix((fe_space.get_ndof(), fe_space.get_ndof()))
        apply_stiffness_assembling(fe_space, param, lil_stiffness)
        stiffness.data = lil_stiffness.tocsc()
    else:
        raise NotImplementedError()

    return stiffness


def apply_stiffness_assembling(fe_space, param, stiffness):
    """
    Applying assembling procedure on stiffness operator.
    :param fe_space: finite element space.
    :param param: input parameter.
    :param stiffness: sparse matrix in lil formatbearing the assembling procedure.
    """
    if stiffness.format is 'lil':
        for ie in range(fe_space.get_nelem()):
            local_stiffness = fe_space.apply_dbasis_diag_dbasis(fe_space.eval_at_quadrature_pnts(
                lambda k, s:
                fe_space.get_quadrature_weight(k) * param(fe_space.get_coord(ie, s)) / fe_space.get_elem_length(ie)))
            locals_to_globals = fe_space.locals_to_globals(ie)
            stiffness[np.array(locals_to_globals)[:, None], locals_to_globals] += local_stiffness
    else:
        raise ValueError("Assembling procedure are expecting lil format sparse matrices.")