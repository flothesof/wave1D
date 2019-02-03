import matplotlib.pyplot as plt
import numpy as np
import wave1D.configuration as configuration
import wave1D.visco_elastic_kelvin_voigt_propagator as visco_elastic_kelvin_voigt_propagator
import wave1D.functional as functional
import wave1D.finite_element_space as fe_sp
import wave1D.finite_element_operator as fe_op
import wave1D.mesh as mesh
import wave1D.mass_assembler as mass_assembler


# Material properties.
def rho(x):
    return 1.0 / 2.0


def modulus(x):
    return 2.0


def eta(x):
    return 0.001


# Creating left dirichlet boundary condition.
left_bc = configuration.BoundaryCondition(boundary_condition_type=configuration.BoundaryConditionType.DIRICHLET,
                                          value=lambda t: functional.ricker(t - 0.4, 25.0))

absorbing_param = np.sqrt(rho(0.) * modulus(0.))
right_bc = configuration.BoundaryCondition(boundary_condition_type=configuration.BoundaryConditionType.ABSORBING,
                                           value=lambda t: 0, param=absorbing_param)

# Creating configuration.
config = configuration.ViscoElasticKelvinVoigt(density=rho, modulus=modulus, eta=eta, left_bc=left_bc,
                                               right_bc=right_bc)

# Creating finite element space.
fe_space = fe_sp.FiniteElementSpace(mesh=mesh.make_mesh_from_npt(0.0, 1.5, 140), fe_order=5, quad_order=5)

# Creating propagator.
propag = visco_elastic_kelvin_voigt_propagator.ViscoElasticKelvinVoigt(config=config, fe_space=fe_space,
                                                                       scheme_type=visco_elastic_kelvin_voigt_propagator
                                                                       .SchemeType.EXPLICIT_ORDERONE)

# Computing mass operator.
mass = mass_assembler.assemble_mass(fe_space, assembly_type=fe_op.AssemblyType.LUMPED)

# Initializing.
propag.initialize(cfl_factor=0.99999999)

# Runing.
fig, ax = plt.subplots()
lines = ax.plot(propag.u0)
ax.set_ylim((-1, 1))
for i in range(50000):
    propag.forward()
    if i % 15 == 0:
        lines[0].set_ydata(propag.u0)
        ax.set_title('Absorbing param: {} Energy: {}'.format(absorbing_param, fe_op.apply_as_linear_form(mass,
               (propag.u0 - propag.u2) / (2.0 * propag.timestep), (propag.u0 - propag.u2) / (2.0 * propag.timestep))))
        plt.pause(0.01)
    propag.swap()
plt.show()

