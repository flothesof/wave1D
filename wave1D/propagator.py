import numpy as np
import scipy.sparse.linalg


class Propagator:
    """
    Base class for proagators.
    """
    def __init__(self, mass, mass_inv, stiffness):
        """
        Construction from mass, inverse of mass matrix, and stiffness matrix.
        """
        self.mass = mass
        self.mass_inv = mass_inv
        self.stiffness = stiffness

        ndof = self.mass.shape[0]
        self.u0 = np.zeros(ndof)
        self.u1 = np.zeros(ndof)
        self.u2 = np.zeros(ndof)

        self.time = 0.
        self.ts = 0.

    def set_init_cond(self, u1=None, u2=None):
        """
        Initilazing propagator.
        :param u1: optional input initial conditions of setp n
        :param u2: optional input initial conditions of setp n-1
        """
        if u1 is not None:
            self.u1 = u1
        if u2 is not None:
            self.u2 = u2

    def set_timestep(self, ts):
        """
        Setting propagator time step.
        """
        self.ts = ts

    def swap(self):
        """
        Swapping DoF vectors.
        """
        u2_tmp = self.u2
        self.u2 = self.u1
        self.u1 = self.u0
        self.u0 = u2_tmp
        self.time += self.ts

    def reset(self):
        """
        Reseting propagator by setting zeros to every solution vectors and time value to 0.
        """
        self.u0.fill(0.)
        self.u1.fill(0.)
        self.u2.fill(0.)
        self.time = 0.


class LeapFrog(Propagator):
    """
    Implementation of a leap frog propagation model.
    """
    def get_cfl(self):
        """
        Computing CFL.
        :return: value of stability time step.
        """
        radius = scipy.sparse.linalg.eigs(self.mass_inv * self.stiffness, k=1, which='LM', return_eigenvectors=False)
        return 2.0 / np.sqrt(np.real(radius))

    def forward(self):
        """
        Moving model forward.
        """
        self.u0 = self.stiffness * self.u1
        self.u0 *= -self.ts * self.ts
        self.u0 = self.mass_inv * self.u0
        self.u0 += 2.0 * self.u1 - self.u2


class ModifiedEquation(Propagator):
    """
    Implementation of a modified equation propagation model.
    """
    def get_cfl(self):
        """
        Computing CFL.
        :return: value of stability time step.
        """
        radius = scipy.sparse.linalg.eigs(self.mass_inv * self.stiffness, k=1, which='LM', return_eigenvectors=False)
        return np.sqrt(12.0 / np.real(radius))

    def forward(self):
        """
        Moving model forward.
        """
        self.u0 = self.stiffness * self.u1
        self.u0 = self.mass_inv * self.u0
        self.u0 *= - self.ts * self.ts / 12.0
        self.u0 += self.u1
        self.u0 = self.stiffness * self.u0
        self.u0 = self.mass_inv * self.u0
        self.u0 *= -self.ts * self.ts
        self.u0 += 2.0 * self.u1 - self.u2
