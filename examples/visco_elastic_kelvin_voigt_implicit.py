import matplotlib.pyplot as plt
import numpy as np
import wave1D.configuration as configuration
import wave1D.visco_elastic_kelvin_voigt_propagator as visco_elastic_kelvin_voigt_propagator
import wave1D.functional as functional
import wave1D.finite_element_space as fe_sp
import wave1D.mesh as mesh
import wave1D.signal_processing as signal_processing


# Simulation parameters.
show_snapshot = False
make_output = False
n_step = 5000
central_frequency = 4.0
src_offset = 1.0

# definition of target velocity.
target_vp = 6.0

# Material properties.
target_law_coef = 1.0e-5


def rho(x):
    return 8.0


def modulus(x):
    return rho(x) * (target_vp ** 2)


def eta(x):
    return 2.0 * modulus(x) * target_vp * target_law_coef


# Creating left dirichlet boundary condition.
left_bc = configuration.BoundaryCondition(boundary_condition_type=configuration.BoundaryConditionType.DIRICHLET,
                                          value=lambda t: functional.ricker(t - src_offset, central_frequency))

absorbing_param = np.sqrt(rho(0.) * modulus(0.))
right_bc = configuration.BoundaryCondition(boundary_condition_type=configuration.BoundaryConditionType.ABSORBING,
                                           value=lambda t: 0, param=absorbing_param)

# Creating configuration.
config = configuration.ViscoElasticKelvinVoigt(density=rho, modulus=modulus, eta=eta, left_bc=left_bc,
                                               right_bc=right_bc)

# Creating finite element space.
fe_space = fe_sp.FiniteElementSpace(mesh=mesh.make_mesh_from_npt(0.0, 10.0, 300), fe_order=5, quad_order=5)

# Creating propagator.
propag = visco_elastic_kelvin_voigt_propagator.ViscoElasticKelvinVoigt(config=config, fe_space=fe_space)

# Initializing.
propag.initialize()

# Runing.
if show_snapshot:

    fig, ax = plt.subplots()
    lines = ax.plot(propag.u0)
    ax.set_ylim((-1, 1))
    plt.ylabel("u (mm)")
    plt.xlabel("x (mm)")
    lines[0].set_xdata(fe_space.get_dof_coords())
    plt.xlim([np.min(fe_space.mesh.pnts), np.max(fe_space.mesh.pnts)])

    for i in range(n_step):
        propag.forward()
        if i % 10 == 0:
            lines[0].set_ydata(propag.u0)
            plt.pause(0.01)
        propag.swap()
    plt.show()

else:

    # Observation point index.
    obs_idx = int(fe_space.get_ndof() / 2)
    obs_coord = fe_space.get_dof_coords()[obs_idx]
    obs_sol = []

    # Extracting solution at observation point.
    for i in range(n_step):
        propag.forward()
        obs_sol.append(propag.u0[obs_idx])
        propag.swap()

    # Computing exact solution without attenuation
    T = n_step * propag.timestep
    times = np.linspace(0., T, n_step)
    exact_solution_no_att = functional.ricker(times - (obs_coord / target_vp) - src_offset, central_frequency)

    # Computing frequency components.
    obs_sol_f, freqs = signal_processing.frequency_synthesis(obs_sol, propag.timestep)
    exact_solution_no_att_f, freqs = signal_processing.frequency_synthesis(exact_solution_no_att, propag.timestep)

    # Compouting attenuation law.
    numerical_law = (np.log(np.abs(exact_solution_no_att_f)) - np.log(np.abs(obs_sol_f))) / obs_coord
    target_law = target_law_coef * (2.0 * np.pi * freqs) ** 2

    # Plotting frequency analysis and attenuation law.
    plt.subplot(311)
    plt.plot(freqs, obs_sol_f)
    plt.plot(freqs, exact_solution_no_att_f)
    plt.xlim([0., 10.0])

    plt.subplot(312)
    plt.plot(freqs, numerical_law)
    plt.plot(freqs, target_law)
    plt.xlim([0., 10.0])
    plt.ylim([0., 0.05])

    plt.subplot(313)
    plt.plot(times, obs_sol)
    plt.plot(times, exact_solution_no_att)
    plt.xlim([0., np.max(times)])
    plt.ylim([-1.0, 1.0])
    plt.show()

    if make_output is True:
        np.savetxt('kelving_voigt_implicit_numerical_solution_attenuation.txt', obs_sol_f)
        np.savetxt('kelving_voigt_implicit_exact_solution_no_attenuation.txt', exact_solution_no_att_f)
        np.savetxt('kelving_voigt_implicit_frequencies.txt', freqs)

