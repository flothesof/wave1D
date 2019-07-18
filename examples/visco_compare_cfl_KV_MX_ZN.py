from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import wave1D.configuration as configuration
import wave1D.elastic_propagator as elastic_propagator
import wave1D.visco_elastic_kelvin_voigt_propagator as visco_elastic_kelvin_voigt_propagator
import wave1D.visco_elastic_maxwell_propagator as visco_elastic_maxwell_propagator
import wave1D.visco_elastic_zener_propagator as visco_elastic_zener_propagator
import wave1D.finite_element_space as fe_sp
import wave1D.mesh as mesh


# Definition of calibration procedures.
def calibration_kv(wstar, Cstar, Dstar):
    return Cstar, Dstar / wstar


def calibration_mx(wstar, Cstar, Dstar):
    M = (Cstar ** 2 + Dstar ** 2) / Cstar
    eta = (Cstar ** 2 + Dstar ** 2) / (wstar * Dstar)
    return M, eta


def calibration_zn(wstar, Cstar, Dstar):
    tau0 = 1.0 / wstar
    C = Cstar - Dstar
    D = Dstar + Cstar
    return tau0, C, D


# Computing cfls of each visco elastic models.
def get_model_cfls(rho, Cstar, Dstar, wstar, fe_space):

    M_kv, eta_kv = calibration_kv(wstar, Cstar, Dstar)
    M_mx, eta_mx = calibration_mx(wstar, Cstar, Dstar)
    tau_zn, M_zn, eta_zn = calibration_zn(wstar, Cstar, Dstar)

    config_kv = configuration.ViscoElasticKelvinVoigt(density=lambda x: rho, modulus=lambda x: M_kv, eta=lambda x: eta_kv)
    config_mx = configuration.ViscoElasticMaxwell(density=lambda x: rho, modulus=lambda x: M_mx, eta=lambda x: eta_mx)
    config_zn = configuration.ViscoElasticZener(density=lambda x: rho, modulus=lambda x: M_zn, eta=lambda x: eta_zn,
                                                tau=lambda x: tau_zn)
    config_inviscid = configuration.Elastic(alpha=lambda x: rho, beta=lambda x: Cstar)

    propag_kv = visco_elastic_kelvin_voigt_propagator.ViscoElasticKelvinVoigt(config=config_kv, fe_space=fe_space)
    propag_mx = visco_elastic_maxwell_propagator.ViscoElasticMaxwell(config=config_mx, fe_space=fe_space)
    propag_zn = visco_elastic_zener_propagator.ViscoElasticZener(config=config_zn, fe_space=fe_space)
    propag_inviscid = elastic_propagator.Elastic(config_inviscid, fe_space)

    propag_kv.initialize(cfl_factor=1.0)
    propag_mx.initialize(cfl_factor=1.0)
    propag_zn.initialize(cfl_factor=1.0)
    propag_inviscid.initialize(cfl_factor=1.0)

    return propag_kv.timestep, propag_mx.timestep, propag_zn.timestep, propag_inviscid.timestep


# Extracting slope in log scale from input data.
def get_log_slope(x, y):
    return np.polyfit(np.log(x), np.log(y), 1)[0]


# Type of comparaison
class CompareType(Enum):
    MESH_STEP = 1
    QFACTOR = 2


# Choosing comparison type.
compare_type = CompareType.MESH_STEP

if compare_type is CompareType.MESH_STEP:

    # Definition of target parameters.
    density = 8.0
    C = 288.0  # GPa
    D = 10.0  # GPa
    f = 4.0  # MHz
    w = 2.0 * np.pi * f

    cfl_kv = []
    cfl_mx = []
    cfl_zn = []
    cfl_inviscid = []
    npts = np.linspace(30, 200, num=25)

    for npt in npts:

        # Creating finite element space.
        fe_space = fe_sp.FiniteElementSpace(mesh=mesh.make_mesh_from_npt(0.0, 3.0, npt), fe_order=5, quad_order=5)

        # computing CFL
        dt_kv, dt_mx, dt_zn, dt_inviscid = get_model_cfls(density, C, D, w, fe_space)

        # Extracting CFL.
        cfl_kv.append(dt_kv)
        cfl_mx.append(dt_mx)
        cfl_zn.append(dt_zn)
        cfl_inviscid.append(dt_inviscid)

    h = 3.0 / (npts - 1)
    plt.loglog(h, cfl_inviscid, label="inviscd: " + str(get_log_slope(h, cfl_inviscid)))
    plt.loglog(h, cfl_kv, label="kv: " + str(get_log_slope(h, cfl_kv)))
    plt.loglog(h, cfl_mx, label="mx: " + str(get_log_slope(h, cfl_mx)))
    plt.loglog(h, cfl_zn, label="zn: " + str(get_log_slope(h, cfl_zn)))
    plt.legend(loc='lower right')
    plt.xlabel("mesh step (mm)")
    plt.ylabel("cfl condition (µs)")
    plt.show()

    np.savetxt('steps.txt', h)
    np.savetxt('cfl_inviscid_vs_steps.txt', cfl_inviscid)
    np.savetxt('cfl_kv_vs_steps.txt', cfl_kv)
    np.savetxt('cfl_mx_vs_steps.txt', cfl_mx)
    np.savetxt('cfl_zn_vs_steps.txt', cfl_zn)
    np.savetxt('slopes_vs_steps.txt', [get_log_slope(h, cfl_inviscid), get_log_slope(h, cfl_kv), get_log_slope(h, cfl_mx), get_log_slope(h, cfl_zn)])

elif compare_type is CompareType.QFACTOR:

    # Definition of target parameters.
    density = 8.0
    C = 288.0  # GPa
    Ds = np.linspace(10.0, C, num=100)  # GPa
    f = 4.0  # MHz
    w = 2.0 * np.pi * f

    cfl_kv = []
    cfl_mx = []
    cfl_zn = []
    cfl_inviscid = []
    fe_space = fe_sp.FiniteElementSpace(mesh=mesh.make_mesh_from_npt(0.0, 3.0, 30), fe_order=5, quad_order=5)

    for D in Ds:

        # computing CFL
        dt_kv, dt_mx, dt_zn, dt_inviscid = get_model_cfls(density, C, D, w, fe_space)

        # Extracting CFL.
        cfl_kv.append(dt_kv)
        cfl_mx.append(dt_mx)
        cfl_zn.append(dt_zn)
        cfl_inviscid.append(dt_inviscid)

    Qs = C / Ds
    plt.loglog(Qs, cfl_inviscid, label="inviscid")
    plt.loglog(Qs, cfl_kv, label="kv: " + str(get_log_slope(Qs, cfl_kv)))
    plt.loglog(Qs, cfl_mx, label="mx: " + str(get_log_slope(Qs, cfl_mx)))
    plt.loglog(Qs, cfl_zn, label="zn: " + str(get_log_slope(Qs, cfl_zn)))
    plt.xlabel("quality factor")
    plt.ylabel("cfl condition (µs)")
    plt.legend(loc='lower right')
    plt.show()

    np.savetxt('Qs.txt', Qs)
    np.savetxt('cfl_inviscid_vs_Qs.txt', cfl_inviscid)
    np.savetxt('cfl_kv_vs_Qs.txt', cfl_kv)
    np.savetxt('cfl_mx_vs_Qs.txt', cfl_mx)
    np.savetxt('cfl_zn_vs_Qs.txt', cfl_zn)
    np.savetxt('slopes_vs_Qs.txt', [get_log_slope(Qs, cfl_inviscid), get_log_slope(Qs, cfl_kv), get_log_slope(Qs, cfl_mx), get_log_slope(Qs, cfl_zn)])


