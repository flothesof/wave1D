from wave1D import mesh, operator
import numpy as np


def test_assemble_p1_mass():
    """
    Tests for assemble_p1_mass() routine.
    """
    mass = operator.assemble_p1_mass(mesh.make_mesh_from_npt(0.0, 1.3, 10))

    assert mass.size == 10
    assert np.abs(mass.sum() - 1.3) <= 1e-16

    mass = operator.assemble_p1_mass(mesh.make_mesh_from_npt(0.0, 1.0, 3))

    assert np.array_equal(mass, np.array([0.25, 0.5, 0.25]))


def test_assemble_p1_stiffness():
    """
    Tests for assemble_p1_stiffness() routine.
    """
    # Not defined data.
    stiffness = operator.assemble_p1_stiffness(mesh.make_mesh_from_npt(0.0, 1.0, 3)).todense()
    expected_stiffness = [[2.0, -2.0, 0.0], [-2.0, 4.0, -2.0], [0.0, -2.0, 2.0]]

    assert np.array_equal(stiffness, expected_stiffness)

    # Comparing constante data with not defined velocity case.
    msh = mesh.make_mesh_from_npt(0.0, 1.3, 15)
    stiffness0 = operator.assemble_p1_stiffness(msh).todense()
    stiffness1 = operator.assemble_p1_stiffness(msh, data=np.ones_like(msh)).todense()

    assert np.abs(stiffness0.sum()) <= 1e-12
    assert np.abs(stiffness1.sum()) <= 1e-12
    assert np.array_equal(stiffness0, stiffness1)

    # Defined data.
    stiffness = operator.assemble_p1_stiffness(mesh.make_mesh_from_npt(0.0, 1.0, 3), data=[1.0, 2.0, 3.0]).todense()
    expected_stiffness = [[3.0, -3.0, 0.0], [-3.0, 8.0, -5.0], [0.0, -5.0, 5.0]]

    assert np.array_equal(stiffness, expected_stiffness)



