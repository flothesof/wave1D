import matplotlib.pyplot as plt
import numpy as np
import wave1D.configuration as configuration
import wave1D.visco_elastic_zener_propagator as visco_elastic_zener_propagator
import wave1D.functional as functional
import wave1D.finite_element_space as fe_sp
import wave1D.mesh as mesh
import wave1D.signal_processing as signal_processing


# Simulation parameters.
show_snapshot = False
make_output = False
n_step = 5000
central_frequency = 2.0
src_offset = 1.0

# definition of target velocity.
target_vp = 6.0

# definition of target angular frequency.
target_f = central_frequency  # (in MHz)
target_w = 2.0 * np.pi * target_f

# definition of attenuation law at target angular frequency.
law_at_target_w = 2.5e-02
law_slope = law_at_target_w / target_w

# definition of mass density.
density_value = 8.0

# Computing zener calibration parameters.
Q = 1.0 / (2.0 * law_slope * target_vp)
k1 = density_value * target_vp ** 2
k2 = 0.5 * (np.sqrt(Q ** 2 + 1.0) - 1.0) * k1
eta_zener = Q * k1 / (2.0 * target_w)


# definition of reoligical parameters of Zener model.
def density(x):
    return density_value


def modulus1(x):
    return k1


def modulus2(x):
    return k2


def eta(x):
    return eta_zener


# Creating left dirichlet boundary condition.
left_bc = configuration.BoundaryCondition(boundary_condition_type=configuration.BoundaryConditionType.DIRICHLET,
                                          value=lambda t: functional.ricker(t - src_offset, central_frequency))

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
    exact_solution_att = signal_processing.apply_attenuation_filter(exact_solution_no_att, T, propag.timestep,
                                                                    path_length=obs_coord,
                                                                    attenuation_filter=lambda f: law_slope * 2.0 * np.pi * f)

    # Computing frequency components.
    obs_sol_f, freqs = signal_processing.frequency_synthesis(obs_sol, T, propag.timestep)
    exact_solution_att_f, freqs = signal_processing.frequency_synthesis(exact_solution_att, T, propag.timestep)
    exact_solution_no_att_f, freqs = signal_processing.frequency_synthesis(exact_solution_no_att, T, propag.timestep)

    # Compouting attenuation law.
    numerical_law = (np.log(np.abs(exact_solution_no_att_f)) - np.log(np.abs(obs_sol_f))) / obs_coord
    target_law = law_slope * 2.0 * np.pi * freqs

    # Plotting frequency analysis and attenuation law.
    plt.subplot(311)
    plt.plot(freqs, np.abs(obs_sol_f))
    plt.plot(freqs, np.abs(exact_solution_att_f))
    plt.plot(freqs, np.abs(exact_solution_no_att_f))
    plt.xlim([0., 10.0])

    plt.subplot(312)
    plt.plot(freqs, numerical_law)
    plt.plot(freqs, target_law)
    plt.xlim([0., 5.0])
    plt.ylim([0, 0.06])

    plt.subplot(313)
    plt.plot(times, obs_sol)
    plt.plot(times, exact_solution_att)
    plt.plot(times, exact_solution_no_att)
    plt.xlim([0., np.max(times)])
    plt.ylim([-1.0, 1.0])
    plt.show()

    if make_output is True:
        np.savetxt('zener_times.txt', times)

        np.savetxt('zener_numerical_solution_f.txt', np.abs(obs_sol_f))
        np.savetxt('zener_exact_solution_att_f.txt', np.abs(exact_solution_att_f))
        np.savetxt('zener_exact_solution_no_att_f.txt', np.abs(exact_solution_no_att_f))

        np.savetxt('zener_numerical_law.txt', numerical_law)
        np.savetxt('zener_target_law.txt', target_law)

        np.savetxt('zener_numerical_solution.txt', obs_sol)
        np.savetxt('zener_exact_solution_att.txt', exact_solution_att)
        np.savetxt('zener_exact_solution_no_att.txt', exact_solution_no_att)

        np.savetxt('zener_frequencies.txt', freqs)
        np.savetxt('zener_observation_point.txt', np.array([obs_coord]))