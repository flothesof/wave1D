import finite_element_operator as fe_op
import finite_element_space as fe_sp
import numpy as np
import scipy.sparse


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
        lil_stiffness = scipy.sparse.lil_matrix((fe_space.get_ndof(), fe_space.get_ndof()))
        apply_stiffness_assembling(param, fe_space, lil_stiffness)
        stiffness.data = lil_stiffness.tocsr()
    else:
        raise NotImplementedError()

    return stiffness


def apply_stiffness_assembling(param, fe_space, stiffness):
    """
    Applying assembling procedure on stiffness operator.
    :param param: input parameter.
    :param fe_space: finite element space.
    :param stiffness: sparse matrix in lil formatbearing the assembling procedure.
    """
    if stiffness.format is 'lil':
        for ie in range(fe_space.get_nelem()):
            local_stiffness = fe_space.apply_dbasis_diag_dbasis(fe_space.eval_at_quadrature_pnts(
                lambda k, s:
                fe_space.get_quadrature_weight(k) * fe_space.get_elem_length(ie) * param(fe_space.get_coord(ie, s))))
            locals_to_globals = fe_space.locals_to_globals(ie)
            stiffness[np.array(locals_to_globals)[:, None], locals_to_globals] += local_stiffness
    else:
        raise ValueError("Assembling procedure are expecting lil format sparse matrices.")