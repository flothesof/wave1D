"""
This script is a 1D simulation of a Resonant Ultrasound Spectroscopy (RUS) configuration.

A displacement function is imposed on the left side while the right side is left free.
The source is chosen so as to excite a mode, with its frequency being a resonance frequency.
"""

import matplotlib.pyplot as plt
import numpy as np
import wave1D.configuration as configuration
import wave1D.elastic_propagator as elastic_propagator
import wave1D.functional as functional
import wave1D.finite_element_space as fe_sp
import wave1D.finite_element_operator as fe_op
import wave1D.mesh as mesh
import wave1D.mass_assembler as mass_assembler


# Material properties.
def celerity(x):
    return 1.


# Warning: celerity is equal to celerity(x) iff beta(x) = 1 for all x
def alpha(x):
    c = celerity(x)
    c2 = c * c
    return 1.0 / c2


def beta(x):
    return 1.0


def source(t, freq=10 * 0.333, offset=0.5, n_periods=1000):
    eps = 0.01
    return np.cos(2 * np.pi * freq * (t - offset)) * functional.heaviside(t - offset, eps) * functional.heaviside(
        -(t - (offset + n_periods * 1 / freq)), eps)


left_bc = configuration.BoundaryCondition(boundary_condition_type=configuration.BoundaryConditionType.ROBIN, param=0.0,
                                          value=source)

# Creating configuration.
config = configuration.Elastic(alpha=alpha, beta=beta, left_bc=left_bc)

# Creating finite element space.
fe_space = fe_sp.FiniteElementSpace(mesh=mesh.make_mesh_from_npt(0.0, 1.5, 140), fe_order=5, quad_order=5)

# Creating propagator.
propag = elastic_propagator.Elastic(config=config, fe_space=fe_space,
                                    mass_assembly_type=fe_op.AssemblyType.LUMPED,
                                    stiffness_assembly_type=fe_op.AssemblyType.ASSEMBLED)

# Computing mass operator.
mass = mass_assembler.assemble_mass(fe_space, assembly_type=fe_op.AssemblyType.LUMPED)

# Initializing.
propag.initialize()

# Running and plot.
fig, ax = plt.subplots()
l, = ax.plot(propag.u0)
ax.set_ylim(-0.2, 0.2)
for i in range(10000):
    propag.forward()
    if i % 15 == 0:
        l.set_ydata(propag.u1)
        plt.pause(0.01)
    propag.swap()
