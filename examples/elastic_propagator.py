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
    return 2.0 + 1.0 * np.exp(-1000 * (x - 0.3) ** 2)


def alpha(x):
    c = celerity(x)
    c2 = c * c
    return 1.0 / c2


def beta(x):
    return 2.0


# Creating left robin boundary condition.
left_bc = configuration.BoundaryCondition(boundary_condition_type=configuration.BoundaryConditionType.DIRICHLET,
                                          value=lambda t: functional.ricker(t - 0.4, 25.0))

absorbing_param = np.sqrt(beta(1.5) * alpha(1.5))
right_bc = configuration.BoundaryCondition(boundary_condition_type=configuration.BoundaryConditionType.ABSORBING,
                                           value=lambda t: 0, param=absorbing_param)

# Creating configuration.
config = configuration.Elastic(alpha=alpha, beta=beta, left_bc=left_bc, right_bc=right_bc)

# Creating finite element space.
fe_space = fe_sp.FiniteElementSpace(mesh=mesh.make_mesh_from_npt(0.0, 1.5, 140), fe_order=5, quad_order=5)

# Creating propagator.
propag = elastic_propagator.ElasticExplicitOrderTwo(config=config, fe_space=fe_space,
                                                    mass_assembly_type=fe_op.AssemblyType.LUMPED,
                                                    stiffness_assembly_type=fe_op.AssemblyType.ASSEMBLED)

# Computing mass operator.
mass = mass_assembler.assemble_mass(fe_space, assembly_type=fe_op.AssemblyType.LUMPED)

# Initializing.
propag.initialize()

# Runing.
fig, ax = plt.subplots()
lines = ax.plot(propag.u0)
ax.set_ylim((-1, 1))
for i in range(10000):
    propag.forward()
    if i % 15 == 0:
        lines[0].set_ydata(propag.u0)
        ax.set_title('Absorbing param: {} Energy: {}'.format(absorbing_param, fe_op.apply_as_linear_form(mass,
               (propag.u0 - propag.u2) / (2.0 * propag.timestep), (propag.u0 - propag.u2) / (2.0 * propag.timestep))))
        plt.pause(0.00001)
    propag.swap()
plt.show()

