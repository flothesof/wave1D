import matplotlib.pyplot as plt
import numpy as np
import wave1D.configuration as configuration
import wave1D.visco_elastic_maxwell_propagator as visco_elastic_maxwell_propagator
import wave1D.functional as functional
import wave1D.finite_element_space as fe_sp
import wave1D.mesh as mesh

target_law = 5.0e-2


# Material properties.
def rho(x):
    return 8.0


def modulus(x):
    return 288.0


def eta(x):
    vp = np.sqrt(modulus(x) / rho(x))
    return rho(x) / (2.0 * target_law * vp)


# Creating left dirichlet boundary condition.
left_bc = configuration.BoundaryCondition(boundary_condition_type=configuration.BoundaryConditionType.DIRICHLET,
                                          value=lambda t: functional.ricker(t - 0.4, 2.0 * 2.0 * np.pi))

absorbing_param = np.sqrt(rho(0.) * modulus(0.))
right_bc = configuration.BoundaryCondition(boundary_condition_type=configuration.BoundaryConditionType.ABSORBING,
                                           value=lambda t: 0, param=absorbing_param)

# Creating configuration.
config = configuration.ViscoElasticMaxwell(density=rho, modulus=modulus, eta=eta, left_bc=left_bc, right_bc=right_bc)

# Creating finite element space.
fe_space = fe_sp.FiniteElementSpace(mesh=mesh.make_mesh_from_npt(0.0, 10.0, 300), fe_order=5, quad_order=5)

# Creating propagator.
propag = visco_elastic_maxwell_propagator.ViscoElasticMaxwell(config=config, fe_space=fe_space)

# Initializing.
propag.initialize()

# Runing.
fig, ax = plt.subplots()
lines = ax.plot(propag.u0)
ax.set_ylim((-1, 1))
plt.ylabel("u (mm)")
plt.xlabel("x (mm)")
lines[0].set_xdata(fe_space.get_dof_coords())
plt.xlim([np.min(fe_space.mesh.pnts), np.max(fe_space.mesh.pnts)])
for i in range(10000):
    propag.forward()
    if i % 10 == 0:
        lines[0].set_ydata(propag.u0)
        plt.pause(0.01)
    propag.swap()
plt.show()

