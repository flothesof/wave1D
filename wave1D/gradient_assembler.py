import wave1D.finite_element_operator as fe_op
import wave1D.finite_element_space as fe_sp
import numpy as np
import scipy.sparse


def assemble_gradient(fe_space, param=lambda x: 1.0, assembly_type=fe_op.AssemblyType.ASSEMBLED):
    """
    Assembling gradient operator applied on a unknown in H^1-conform discrete space and return a result in a L^2-conform
    discrete space.
    :param fe_space: input finite element space.
    :param param: function of space variable.
    :param assembly_type: type of assembling procedure.
    :return: instance of FiniteElementOperator class representing the gradient operator.
    """
    if assembly_type is fe_op.AssemblyType.ASSEMBLED:

        # Computing operator data.
        lil_gradient = scipy.sparse.lil_matrix((fe_space.get_nelem() * fe_space.get_nlocaldof(), fe_space.get_ndof()))
        apply_gradient_assembling(fe_space, param, lil_gradient)

        # Setting operator data.
        gradient = fe_op.FiniteElementOperator(fe_space=fe_sp.FiniteElementSpace(), assembly_type=assembly_type)
        gradient.data = lil_gradient.tocsc()

        return gradient
    else:
        raise NotImplementedError()


def assemble_transposed_gradient(fe_space, param=lambda x: 1.0, assembly_type=fe_op.AssemblyType.ASSEMBLED):
    """
    Assembling transposed gradient operator applied on a unknown in L^2-conform discrete space and return a result in a
    H^1-conform discrete space.
    :param fe_space: input finite element space.
    :param param: function of space variable.
    :param assembly_type: type of assembling procedure.
    :return: instance of FiniteElementOperator class representing the transposed gradient operator.
    """
    if assembly_type is fe_op.AssemblyType.ASSEMBLED:
        gradient = assemble_gradient(fe_space, param, assembly_type)
        gradient.data = gradient.data.transpose()
        return gradient
    else:
        raise NotImplementedError()


def apply_gradient_assembling(fe_space, param, gradient):
    """
    Applying assembling procedure on gradient operator.
    :param fe_space: finite element space.
    :param param: input parameter.
    :param gradient: sparse matrix in lil format bearing the assembling procedure.
    """
    if gradient.format is 'lil':
        for ie in range(fe_space.get_nelem()):

            # Computing local operator.
            local_gradient = fe_space.apply_basis_diag_dbasis(fe_space.eval_at_quadrature_pnts(
                lambda k, s: fe_space.get_quadrature_weight(k) * param(fe_space.get_coord(ie, s))))

            # Extracting local to global index mapping
            locals_to_globals_h1 = fe_space.locals_to_globals(ie)
            locals_to_globals_l2 = fe_space.locals_to_globals_discontinuous(ie)

            # Adding local operator into global operator.
            gradient[np.array(locals_to_globals_l2)[:, None], locals_to_globals_h1] += local_gradient
    else:
        raise ValueError("Assembling procedure are expecting lil format sparse matrices.")


