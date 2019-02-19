import matplotlib.pyplot as plt
import numpy as np
import wave1D.configuration as configuration
import wave1D.visco_elastic_zener_propagator as visco_elastic_zener_propagator
import wave1D.visco_elastic_kelvin_voigt_propagator as visco_elastic_kelvin_voigt_propagator
import wave1D.functional as functional
import wave1D.finite_element_space as fe_sp
import wave1D.mesh as mesh
import wave1D.signal_processing as signal_processing


def run_propagator(nstep, space, propag):
    fig, ax = plt.subplots()
    lines = ax.plot(propag.u0)
    ax.set_ylim((-1, 1))
    plt.ylabel("u (mm)")
    plt.xlabel("x (mm)")
    lines[0].set_xdata(space.get_dof_coords())
    plt.xlim([np.min(space.mesh.pnts), np.max(space.mesh.pnts)])

    for i in range(nstep):
        propag.forward()
        if i % 10 == 0:
            lines[0].set_ydata(propag.u0)
            plt.pause(0.01)
        propag.swap()
    plt.show()


def MRtau_to_k1k2eta(MR, tsig, teps):
    k1 = MR * teps / tsig
    k2 = (k1 * MR) / (k1 - MR)
    eta = k2 * teps
    return k1, k2, eta


def kelvin_voigt_to_zener(C, D, wstar):
    Qstar = C / (wstar * D)
    Q2star = Qstar ** 2
    teps = (D / C) * (np.sqrt(1.0 + Q2star) + 1.0)
    tsig = (D / C) * (np.sqrt(1.0 + Q2star) - 1.0)
    MR = 0.5 * C * (1.0 + wstar ** 2 * tsig ** 2)
    return MRtau_to_k1k2eta(MR, tsig, teps)


# Simulation parameters.
show_snapshot = False
run_kv = False
run_z = True
make_output = True
wide_band_src = False
n_step_kv = 30000
n_step_zener = 10000
central_frequency = 4.0
narrow_band_std_dev = 0.5
narrow_band_offset = 1.5
wide_band_offset = 1.0

# Definition of target frequency.
target_f = central_frequency  # (in MHz)
target_w = 2.0 * np.pi * target_f

# definition of material parameters for KV model.
rho = 8.0
C = 288.0
D = 0.1
VP = np.sqrt(C / rho)

# definition of material parameters for Zener model.
k1, k2, eta = kelvin_voigt_to_zener(C, D, target_w)

# Creating finite element space.
fe_space = fe_sp.FiniteElementSpace(mesh=mesh.make_mesh_from_npt(0.0, 30.0, 900), fe_order=5, quad_order=5)

# Creating boundary conditions.
if wide_band_src is False:
    left_bc = configuration.BoundaryCondition(boundary_condition_type=configuration.BoundaryConditionType.DIRICHLET,
              value=lambda t: functional.gated_cosine(t - narrow_band_offset, central_frequency, narrow_band_std_dev))
elif wide_band_src is True:
    left_bc = configuration.BoundaryCondition(boundary_condition_type=configuration.BoundaryConditionType.DIRICHLET,
              value=lambda t: functional.ricker(t - wide_band_offset, central_frequency))

absorbing_param = np.sqrt(rho * C)
right_bc = configuration.BoundaryCondition(boundary_condition_type=configuration.BoundaryConditionType.ABSORBING,
                                           value=lambda t: 0, param=absorbing_param)

# Creating configurations.
config_KV = configuration.ViscoElasticKelvinVoigt(density=lambda x: rho, modulus=lambda x: C, eta=lambda x: D,
                                                  left_bc=left_bc, right_bc=right_bc)

config_Z = configuration.ViscoElasticZener(density=lambda x: rho, modulus1=lambda x: k1, modulus2=lambda x: k2,
                                           eta=lambda x: eta, left_bc=left_bc, right_bc=right_bc)

# Creating propagators.
propag_KV = visco_elastic_kelvin_voigt_propagator.ViscoElasticKelvinVoigt(config=config_KV, fe_space=fe_space,
                                       scheme_type=visco_elastic_kelvin_voigt_propagator.SchemeType.EXPLICIT_ORDERONE)

propag_Z = visco_elastic_zener_propagator.ViscoElasticZener(config=config_Z, fe_space=fe_space)

# Initializing.
propag_KV.initialize()
propag_Z.initialize()

# Runing.
if show_snapshot:

    if run_kv:
        run_propagator(n_step_kv, fe_space, propag_KV)

    if run_z:
        run_propagator(n_step_zener, fe_space, propag_Z)

else:

    # Observation point index.
    obs_idx = int(fe_space.get_ndof() / 2)
    obs_coord = fe_space.get_dof_coords()[obs_idx]
    obs_sol_KV = []
    obs_sol_Z = []

    # Extracting solution at observation point.
    for i in range(n_step_kv):
        propag_KV.forward()
        obs_sol_KV.append(propag_KV.u0[obs_idx])
        propag_KV.swap()

    for i in range(n_step_zener):
        propag_Z.forward()
        obs_sol_Z.append(propag_Z.u0[obs_idx])
        propag_Z.swap()

    # Computing frequency components.
    T_KV = n_step_kv * propag_KV.timestep
    T_Z = n_step_zener * propag_Z.timestep

    times_KV = np.linspace(0., T_KV, n_step_kv)
    times_Z = np.linspace(0., T_Z, n_step_zener)

    obs_sol_KV_f, freqs_KV = signal_processing.frequency_synthesis(obs_sol_KV, propag_KV.timestep, n_fft=64000)
    obs_sol_Z_f, freqs_Z = signal_processing.frequency_synthesis(obs_sol_Z, propag_Z.timestep, n_fft=64000)

    # No attenuated exact solution.
    if wide_band_src is False:
        exact_solution_no_att = functional.gated_cosine(times_Z - (obs_coord / VP) - narrow_band_offset,
                                                        central_frequency, narrow_band_std_dev)
    else:
        exact_solution_no_att = functional.ricker(times_Z - (obs_coord / VP) - wide_band_offset, central_frequency)
    exact_solution_no_att_f, freqs_exact = signal_processing.frequency_synthesis(exact_solution_no_att,
                                                                                 propag_Z.timestep, n_fft=64000)

    # Plotting frequency analysis and attenuation law.
    plt.subplot(211)
    plt.plot(freqs_exact, np.abs(exact_solution_no_att_f))
    plt.plot(freqs_KV, np.abs(obs_sol_KV_f))
    plt.plot(freqs_Z, np.abs(obs_sol_Z_f))
    plt.xlim([0., 10.0])

    plt.subplot(212)
    plt.plot(times_Z, exact_solution_no_att)
    plt.plot(times_KV, obs_sol_KV)
    plt.plot(times_Z, obs_sol_Z)
    plt.xlim([0., np.max([np.max(times_Z), np.max(times_KV)])])
    plt.ylim([-1.0, 1.0])
    plt.show()

    if make_output is True:
        np.savetxt('kv_times.txt', times_KV)
        np.savetxt('kv_freqs.txt', freqs_KV)
        np.savetxt('kv_sol.txt', obs_sol_KV)
        np.savetxt('kv_sol_hat.txt', np.abs(obs_sol_KV_f))

        np.savetxt('z_times.txt', times_Z)
        np.savetxt('z_freqs.txt', freqs_Z)
        np.savetxt('z_sol.txt', obs_sol_Z)
        np.savetxt('z_sol_hat.txt', np.abs(obs_sol_Z_f))

        np.savetxt('exact_no_att_times.txt', times_Z)
        np.savetxt('exact_no_att_freqs.txt', freqs_Z)
        np.savetxt('exact_no_att_sol.txt', exact_solution_no_att)
        np.savetxt('exact_no_att_sol_hat.txt', np.abs(exact_solution_no_att_f))


