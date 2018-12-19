import matplotlib.pyplot as plt
import numpy as np
import wave1D.configuration as configuration
import wave1D.elastic_propagator as elastic_propagator
import wave1D.functional as functional
import wave1D.finite_element_space as fe_sp
import wave1D.finite_element_operator as fe_op
import wave1D.mesh as mesh


# Material properties.
def alpha(x):
    return 1.0 + 10.0 * ((x > 0.3) & (x < 0.32))


def beta(x):
    return 1.0 # + 10.0 * ((x > 0.3) & (x < 0.32))


# Creating left robin boundary condition.
left_bc = configuration.BoundaryCondition(boundary_condition_type=configuration.BoundaryConditionType.DIRICHLET,
                                          value=lambda t: functional.ricker(t - 10.0, 0.1))

# Creating configuration.
config = configuration.Elastic(alpha=alpha, beta=beta, left_bc=left_bc)

# Creating finite element space.
fe_space = fe_sp.FiniteElementSpace(mesh=mesh.make_mesh_from_npt(0.0, 1.0, 200), fe_order=5, quad_order=5)

# Creating propagator.
propag = elastic_propagator.ElasticExplicitOrderTwo(config=config, fe_space=fe_space,
                                                    mass_assembly_type=fe_op.AssemblyType.LUMPED,
                                                    stiffness_assembly_type=fe_op.AssemblyType.ASSEMBLED)

# Initializing.
propag.initialize()

# Runing.

fig, ax = plt.subplots(nrows=2)
lines = ax[0].plot(propag.u0)
ax[0].set_ylim((-1, 1))
ax[1].plot([alpha(x) for x in np.linspace(0, 1, num=propag.u0.size)])
ax[1].plot([beta(x) for x in np.linspace(0, 1, num=propag.u0.size)])
# plt.show()
for i in range(5000):
    propag.forward()
    if i % 8 == 0:
        lines[0].set_ydata(propag.u0)
        plt.pause(0.00001)
    propag.swap()
plt.show()

