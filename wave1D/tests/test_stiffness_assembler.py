from numpy import testing as np_test
from wave1D import stiffness_assembler
from wave1D import finite_element_space as fe_sp
from wave1D import mesh


def test_assembled_stiffness_p2():
    """
    Testing assembling routine in the case of p2 finite element space on one element.
    """
    p = 2.3
    fe_space = fe_sp.FiniteElementSpace(mesh.Mesh([0.0, 1.0]), fe_order=2, quad_order=2)
    stiffness = stiffness_assembler.assemble_stiffness(fe_space, lambda s: p)
    np_test.assert_array_almost_equal(stiffness.data.data, [7.0 * p / 3.0, -8.0 * p / 3.0, p / 3.0, -8.0 * p / 3.0,
                                                            16.0 * p / 3.0, -8.0 * p / 3.0, p / 3.0, -8.0 * p / 3.0,
                                                            7.0 * p / 3.0])
