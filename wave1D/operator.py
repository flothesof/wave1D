import numpy as np
import scipy.sparse
from wave1D import mesh


def assemble_p1_mass(msh):
    """
    Assembling mass associated to a mesh and using first order Lagrange polynomials.
    :param msh: input 1D mesh.
    :return: mass as np.array.
    """
    mass = np.zeros_like(msh)

    if msh.size > 1:
        for ie in range(0, mesh.get_mesh_nelem(msh)):
            val = 0.5 * np.abs(msh[ie + 1] - msh[ie])
            mass[ie] += val
            mass[ie+1] += val

    return mass


def assemble_p1_stiffness(msh, data=None):
    """
    Assembling stiffness matrix associated to a mesh and with data as coefs in integrand, using first order Lagrange
    polynomials.
    :param msh: input 1D mesh.
    :param data: optional input coefficients, should respect data.size == msh.size
    :return: stiffness matrix as a scipy.sparse.csc_matrix
    """
    stiffness = scipy.sparse.lil_matrix((msh.size, msh.size), dtype=np.float64)

    if msh.size > 1:
        if data is None:
            for ie in range(0, mesh.get_mesh_nelem(msh)):
                val = 1.0 / np.abs(msh[ie + 1] - msh[ie])
                stiffness[ie, ie] += val
                stiffness[ie + 1, ie] -= val
                stiffness[ie, ie + 1] -= val
                stiffness[ie + 1, ie + 1] += val

        else:
            assert len(data) == msh.size

            for ie in range(0, mesh.get_mesh_nelem(msh)):
                val = 0.5 * (data[ie] + data[ie + 1]) / np.abs(msh[ie + 1] - msh[ie])
                stiffness[ie, ie] += val
                stiffness[ie + 1, ie] -= val
                stiffness[ie, ie + 1] -= val
                stiffness[ie + 1, ie + 1] += val

    return scipy.sparse.csc_matrix(stiffness)
