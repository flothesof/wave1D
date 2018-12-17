import numpy as np


def make_mesh_from_npt(start=0, stop=1, n=2):
    """
    Create a regular 1D mesh from start, ending coordinates and number of points in mesh.
    """
    return np.linspace(start, stop, n)


def fuse_meshes(mesh0, mesh1):
    """
    Fusing two meshes. Points coordinates are ordered and common points w.r.t to tolerance are deleted.
    :param mesh0: first input mesh.
    :param mesh1: second input mesh.
    :return: fused mesh with size <= mesh0.size + mesh1.size
    """
    return np.unique(np.concatenate([mesh0, mesh1], axis=0))
