import numpy as np
from wave1D import mesh


def test_make_mesh_from_npt():
    """
    Tests for make_mesh_from_npt() routine.
    """
    msh = mesh.make_mesh_from_npt(1., 1.1, 10)

    # Testing size.
    assert msh.size == 10

    # Testing step.
    h = np.abs(msh[1] - msh[0])
    expected_h = 0.1 / 9

    assert np.abs(h - expected_h) <= 1e-16


def test_get_mesh_nelem():
    """
    Tests for get_mesh_nelem() routine.
    """
    assert mesh.get_mesh_nelem(mesh.make_mesh_from_npt(1., 1.1, 10)) == 9


def test_empty_mesh():
    """
    Testing creation and manipulation of empty mesh.
    """
    msh = mesh.make_mesh_from_npt(1., 1.1, 0)

    assert msh.size == 0
    assert mesh.get_mesh_nelem(msh) == 0


def test_fuse_mesh():
    """
    Tests for fuse_mesh() routine.
    """
    msh0 = mesh.make_mesh_from_npt(1., 1.1, 2)
    msh1 = mesh.make_mesh_from_npt(1.1, 1.2, 2)
    msh = mesh.fuse_meshes(msh0, msh1)

    assert np.array_equal(msh, [1.0, 1.1, 1.2])



