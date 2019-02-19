import matplotlib.pyplot as plt
import numpy as np
import wave1D.configuration as configuration
import wave1D.visco_elastic_zener_propagator as visco_elastic_zener_propagator
import wave1D.visco_elastic_kelvin_voigt_propagator as visco_elastic_kelvin_voigt_propagator
import wave1D.functional as functional
import wave1D.finite_element_space as fe_sp
import wave1D.mesh as mesh


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
central_frequency = 4.0

# Definition of target frequency.
target_f = central_frequency  # (in MHz)
target_w = 2.0 * np.pi * target_f

# definition of material parameters for KV model.
rho = 8.0
C = 288.0
D = 1.0 # 0.1
VP = np.sqrt(C / rho)

# definition of material parameters for Zener model.
k1, k2, eta = kelvin_voigt_to_zener(C, D, target_w)


# Creating configurations.
config_KV = configuration.ViscoElasticKelvinVoigt(density=lambda x: rho, modulus=lambda x: C, eta=lambda x: D)
config_Z = configuration.ViscoElasticZener(density=lambda x: rho, modulus1=lambda x: k1, modulus2=lambda x: k2)

cfl_z = []
cfl_kv = []
npts = np.linspace(30, 400, num=10)

for npt in npts:

    # Creating finite element space.
    fe_space = fe_sp.FiniteElementSpace(mesh=mesh.make_mesh_from_npt(0.0, 30.0, npt), fe_order=5, quad_order=5)

    # Creating propagators.
    propag_KV = visco_elastic_kelvin_voigt_propagator.ViscoElasticKelvinVoigt(config=config_KV, fe_space=fe_space,
                                           scheme_type=visco_elastic_kelvin_voigt_propagator.SchemeType.EXPLICIT_ORDERONE)

    propag_Z = visco_elastic_zener_propagator.ViscoElasticZener(config=config_Z, fe_space=fe_space)

    # Initializing.
    propag_KV.initialize()
    propag_Z.initialize()

    # Extracting CFL.
    cfl_kv.append(propag_KV.timestep)
    cfl_z.append(propag_Z.timestep)


plt.plot(npts, cfl_kv)
plt.plot(npts, cfl_z)
plt.show()

np.savetxt('npts.txt', npts)
np.savetxt('cfl_kv.txt', cfl_kv)
np.savetxt('cfl_z.txt', cfl_z)

