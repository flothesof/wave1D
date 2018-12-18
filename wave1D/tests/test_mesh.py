import numpy as np
from numpy import testing as np_test
from wave1D import mesh


def test_make_mesh_from_npt():
    """
    Tests for make_mesh_from_npt() routine.
    """
    msh = mesh.make_mesh_from_npt(1., 1.1, 10)

    # Testing size.
    assert msh.pnts.size == 10

    # Testing step.
    h = np.abs(msh.pnts[1] - msh.pnts[0])
    expected_h = 0.1 / 9

    assert np.abs(h - expected_h) <= 1e-16


def test_fuse_mesh():
    """
    Tests for fuse_mesh() routine.
    """
    msh0 = mesh.make_mesh_from_npt(1., 1.1, 2)
    msh1 = mesh.make_mesh_from_npt(1.1, 1.2, 2)
    msh = mesh.fuse_meshes(msh0, msh1)

    assert np.array_equal(msh.pnts, [1.0, 1.1, 1.2])



