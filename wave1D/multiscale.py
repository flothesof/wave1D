import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from wave1D import mesh


def get_multiscale_p1_basis(stiffness):
    """
    Computing multiscale basis functions by solving local elliptic problems with p1 Lagrangian basis function.
    :param stiffness: local discretization of elliptic operator.
    """
    # Inverting stiffness matrix
    reduced_stiff_inv = scipy.sparse.linalg.inv(stiffness[1:-1, 1:-1])

    # Solving problems at interior points.
    reduced_ms_basis0 = (-reduced_stiff_inv * stiffness[1:-1, 0]).toarray().squeeze()
    reduced_ms_basis1 = (-reduced_stiff_inv * stiffness[1:-1, -1]).toarray().squeeze()

    # Concanating extremity values.
    return (np.concatenate(([1.0], reduced_ms_basis0, [0.0]), axis=0),
            np.concatenate(([0.0], reduced_ms_basis1, [1.0]), axis=0))


def assemble_multiscale_p1_local_mass(mass, ms_basis0, ms_basis1):
    """
    Compute local multiscale mass matrix.
    :param mass: local discretization of L2 product.
    :param ms_basis0: first multiscale basis function.
    :param ms_basis1: second multiscale basis function.
    :return a 2 x 2 matrix corresponding to the local multiscale mass matrix
    """
    val0 = np.dot(ms_basis0, mass * ms_basis0)
    val1 = np.dot(ms_basis0, mass * ms_basis1)
    val2 = np.dot(ms_basis1, mass * ms_basis1)

    return np.matrix([[val0, val1], [val1, val2]])


def assemble_multiscale_p1_local_stiffness(stiffness, ms_basis0, ms_basis1):
    """
    Compute local multiscale stiffness matrix.
    :param stiffness: local discretization of elliptic operator.
    :param ms_basis0: first multiscale basis function.
    :param ms_basis1: second multiscale basis function.
    :return a 2 x 2 matrix corresponding to the local multiscale stiffness matrix
    """
    val0 = np.dot(ms_basis0, stiffness * ms_basis0)
    val1 = np.dot(ms_basis0, stiffness * ms_basis1)
    val2 = np.dot(ms_basis1, stiffness * ms_basis1)

    return np.matrix([[val0, val1], [val1, val2]])


def assemble_multiscale_p1_mass(msh, local_ms_masses):
    """
    Assembling multiscale first order mass matrix from a mesh and a dict. with local multiscale mass matrices.
    :param msh: input 1D mesh.
    :param local_ms_masses: local multiscale mass matrix as a dict. with key being the index of ms elements.
    :return: assembled multiscale mass matrix.
    """
    mass = scipy.sparse.lil_matrix((msh.size, msh.size), dtype=np.float64)

    if msh.size > 1:
        for ie in range(0, mesh.get_mesh_nelem(msh)):
            if ie not in local_ms_masses:
                val = 0.5 * np.abs(msh[ie + 1] - msh[ie])
                mass[ie, ie] += val
                mass[ie+1, ie+1] += val
            else:
                mass[ie:ie + 2, ie:ie + 2] += local_ms_masses[ie]

    return scipy.sparse.csc_matrix(mass)


def assemble_multiscale_p1_stiffness(msh, local_ms_stiffnesses, data=None):
    """
    Assembling multiscale first order stiffness matrix from a mesh and a dict. with local multiscale stiffness matrices.
    :param msh: input 1D mesh.
    :param local_ms_stiffnesses: local multiscale stiffness matrix as a dict. with key being the index of ms elements.
    :param data: optional input coefficients, should respect data.size == msh.size
    :return: assembled multiscale stiffness matrix.
    """
    stiffness = scipy.sparse.lil_matrix((msh.size, msh.size), dtype=np.float64)

    if msh.size > 1:
        if data is None:
            for ie in range(0, mesh.get_mesh_nelem(msh)):
                if ie not in local_ms_stiffnesses:
                    val = 1.0 / np.abs(msh[ie + 1] - msh[ie])
                    stiffness[ie, ie] += val
                    stiffness[ie + 1, ie] -= val
                    stiffness[ie, ie + 1] -= val
                    stiffness[ie + 1, ie + 1] += val
                else:
                    stiffness[ie:ie + 2, ie:ie + 2] += local_ms_stiffnesses[ie]
        else:
            assert len(data) == msh.size

            for ie in range(0, mesh.get_mesh_nelem(msh)):
                if ie not in local_ms_stiffnesses:
                    val = 0.5 * (data[ie] + data[ie + 1]) / np.abs(msh[ie + 1] - msh[ie])
                    stiffness[ie, ie] += val
                    stiffness[ie + 1, ie] -= val
                    stiffness[ie, ie + 1] -= val
                    stiffness[ie + 1, ie + 1] += val
                else:
                    stiffness[ie:ie + 2, ie:ie + 2] += local_ms_stiffnesses[ie]

    return stiffness
