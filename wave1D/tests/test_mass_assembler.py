from numpy import testing as np_test
import numpy as np
from wave1D import mass_assembler
from wave1D import finite_element_operator as fe_op
from wave1D import finite_element_space as fe_sp
from wave1D import mesh


def test_assembled_mass_p1():
    """
    Testing assembling routine in the case of p1 finite element space on one element.
    """
    cte_density = 2.3
    fe_space = fe_sp.FiniteElementSpace(mesh.Mesh([0.0, 1.0]), fe_order=1, quad_order=2)
    mass = mass_assembler.assemble_mass(fe_space, lambda s: cte_density)
    np_test.assert_array_almost_equal(mass.data.data,
                                      [cte_density / 3.0, cte_density / 6.0, cte_density / 6.0, cte_density / 3.0])


def test_lumped_mass_p2():
    """
    Testing mass lumping routine in the case of p2 finite element spaces.
    """
    def density(x):
        return 2.3 * x + 1.3

    fe_space = fe_sp.FiniteElementSpace(mesh.Mesh([0.0, 1.0, 4.0]), fe_order=2, quad_order=2)
    assemb_mass = mass_assembler.assemble_mass(fe_space, density, fe_op.AssemblyType.ASSEMBLED).data.todense()
    lumped_mass = np.diag(mass_assembler.assemble_mass(fe_space, density, fe_op.AssemblyType.LUMPED).data)
    np_test.assert_array_almost_equal(assemb_mass, lumped_mass)

