import matplotlib.pyplot as plt
import numpy as np
import wave1D.configuration as configuration
import wave1D.visco_elastic_zener_propagator as visco_elastic_zener_propagator
import wave1D.functional as functional
import wave1D.finite_element_space as fe_sp
import wave1D.mesh as mesh


# definition of target velocity.
target_vp = 6.0

# definition of target angular frequency.
target_f = 2.0  # (in MHz)
target_w = 2.0 * np.pi * target_f

# definition of attenuation law at target angular frequancy.
law_at_target_w = 1.0e-1
law_slope = law_at_target_w / target_w

# definition of mass density.
density_value = 8.0


# definition of reoligical parameters of Zener model.
def density(x):
    return density_value


def modulus1(x):
    return density(x) * target_vp ** 2


def modulus2(x):
    return (np.sqrt(1.0 / (4.0 * law_slope ** 2 * target_vp ** 2) + 1.0) - 1.0) * modulus1(x)


def eta(x):
    return modulus1(x) / (4.0 * target_w * law_slope * target_vp)


# Creating left dirichlet boundary condition.
left_bc = configuration.BoundaryCondition(boundary_condition_type=configuration.BoundaryConditionType.DIRICHLET,
                                          value=lambda t: functional.ricker(t - 0.4, 2.0 * 2.0 * np.pi))

absorbing_param = np.sqrt(density(0.) * modulus1(0.))
right_bc = configuration.BoundaryCondition(boundary_condition_type=configuration.BoundaryConditionType.ABSORBING,
                                           value=lambda t: 0, param=absorbing_param)

# Creating configuration.
config = configuration.ViscoElasticZener(density=density, modulus1=modulus1, modulus2=modulus2, eta=eta,
                                         left_bc=left_bc, right_bc=right_bc)

# Creating finite element space.
fe_space = fe_sp.FiniteElementSpace(mesh=mesh.make_mesh_from_npt(0.0, 10.0, 300), fe_order=5, quad_order=5)

# Creating propagator.
propag = visco_elastic_zener_propagator.ViscoElasticZener(config=config, fe_space=fe_space)

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