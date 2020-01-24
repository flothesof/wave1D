"""
This example proposes an implementation of an Newton-descent inversion scheme on a sample elastic configuration.

The sample configuration is like this:
S==============================|

S is the source of a wide-band pulse
| is our observation point, subject to free end conditions (Neumann)

An overview of the example would be:

- generate "true" synthetic data and extract observations
- perform Newton iterative descent based on the previous observations until convergence
"""
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.interpolate import interp1d

import wave1D.configuration as configuration
from wave1D.functional import ricker
import wave1D.finite_element_space as fe_sp
import wave1D.elastic_propagator as elastic_propagator
import wave1D.finite_element_operator as fe_op
import wave1D.mesh as mesh
import wave1D.mass_assembler as mass_assembler
import wave1D.stiffness_assembler as stiffness_assembler

# Step 0: defining the problem parameters
# =======================================

# "true" value of objective parameter
THETA_BAR = 2.
# observation window [0, TMAX]
TMAX = 5

# current_theta descent parameters
THETA_INIT = 1.45
CONVERGENCE_EPS = 1e-8
MAX_STEP = 10
JREGUL = 0.0


# Step 1: generate "true" synthetic data and extract observations
# ==============================================================

def alpha(x):
    return 1.0


def beta(x, theta):
    return theta


true_beta = partial(beta, theta=THETA_BAR)

# Creating configuration.
# We don't need to specify a right boundary condition since the natural boundary condition is zero derivative,
# which is what we want.
left_bc = configuration.BoundaryCondition(boundary_condition_type=configuration.BoundaryConditionType.ROBIN, param=0.0,
                                          value=lambda t: ricker(t - 2.4, 2.0))

config_theta_bar = configuration.Elastic(alpha=alpha, beta=true_beta, left_bc=left_bc)

fe_space = fe_sp.FiniteElementSpace(mesh=mesh.make_mesh_from_npt(0.0, 1.5, 200), fe_order=5, quad_order=5)

propag = elastic_propagator.Elastic(config=config_theta_bar, fe_space=fe_space,
                                    mass_assembly_type=fe_op.AssemblyType.LUMPED,
                                    stiffness_assembly_type=fe_op.AssemblyType.ASSEMBLED)

mass = mass_assembler.assemble_mass(fe_space, assembly_type=fe_op.AssemblyType.LUMPED)

propag.initialize()

# Computing observer values using forward model
observer = []
while propag.time < TMAX:
    propag.forward()
    # u0 is the newly computed value, u1 and u2 previous timesteps
    value = (propag.u0[-1] - propag.u2[-1]) / (2 * propag.timestep)
    observer.append((propag.time, value))
    propag.swap()

interp_observer = interp1d(np.array(observer)[:, 0],
                           np.array(observer)[:, 1],
                           bounds_error=False,
                           fill_value=0.)

fig, ax = plt.subplots()
ax.plot(*np.array(observer).T, label='observer values')
ax.legend(loc='upper right')
ax.set_xlabel('time')
ax.set_title('observable (velocity at $x=L$)')
plt.show()

# Step 2: perform Newton iterative descent
# ========================================

propag_builder = partial(elastic_propagator.Elastic, fe_space=fe_space,
                         mass_assembly_type=fe_op.AssemblyType.LUMPED,
                         stiffness_assembly_type=fe_op.AssemblyType.ASSEMBLED)

k = stiffness_assembler.assemble_stiffness(fe_space)
minv = mass_assembler.assemble_mass(fe_space, density=alpha, assembly_type=fe_op.AssemblyType.LUMPED)
fe_op.inv(minv)
minv.data = scipy.sparse.dia_matrix((minv.data, [0]), shape=k.data.shape)
minv_k = fe_op.make_from_data(minv.data * k.data, assembly_type=fe_op.AssemblyType.ASSEMBLED)

current_theta = THETA_INIT
convergence_reached = False
step = 0

while not convergence_reached:
    dthetaJ = JREGUL * current_theta
    d2thetaJ = JREGUL

    # Building the propagators for the current descent step
    current_beta = partial(beta, theta=current_theta)

    config_theta = configuration.Elastic(alpha=alpha, beta=current_beta, left_bc=left_bc)
    propag_theta = propag_builder(config=config_theta)


    def rhs_dtheta(fe_space, time):
        rhs = np.zeros(fe_space.get_ndof())
        fe_op.mlt_add(minv_k, propag_theta.u1, rhs, -1.0)
        return rhs


    config_dtheta = configuration.Elastic(alpha=alpha, beta=current_beta, rhs=rhs_dtheta)
    propag_dtheta = propag_builder(config=config_dtheta)


    def rhs_d2theta(fe_space, time):
        rhs = np.zeros(fe_space.get_ndof())
        fe_op.mlt_add(minv_k, propag_dtheta.u1, rhs, -2.0)
        return rhs


    config_d2theta = configuration.Elastic(alpha=alpha, beta=current_beta, rhs=rhs_d2theta)
    propag_d2theta = propag_builder(config=config_d2theta)

    propag_theta.initialize()
    propag_dtheta.initialize()
    propag_d2theta.initialize()


    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
    lines_theta = ax1.plot(propag_theta.u1)
    lines_dtheta = ax2.plot(propag_dtheta.u1)
    lines_d2theta = ax3.plot(propag_d2theta.u1)

    fig_innov, ax_innov = plt.subplots()
    lines_innov = ax_innov.plot([], [])

    i = 0
    innov_snapshots = []
    while propag_theta.time < TMAX:
        propag_theta.forward()
        propag_dtheta.forward()
        propag_d2theta.forward()

        # if i % 15 == 0:
        #     lines_theta[0].set_ydata(propag_theta.u0)
        #     lines_dtheta[0].set_ydata(propag_dtheta.u0)
        #     lines_d2theta[0].set_ydata(propag_d2theta.u0)
        #     plt.pause(0.01)
        # i += 1
        dt = propag_theta.timestep
        innov = interp_observer(propag_theta.time) - (propag_theta.u0[-1] - propag_theta.u2[-1]) / (2 * dt)
        innov_snapshots.append((propag_theta.time, innov))
        velocity_dtheta = (propag_dtheta.u0[-1] - propag_dtheta.u2[-1]) / (2 * dt)
        velocity_d2theta = (propag_d2theta.u0[-1] - propag_d2theta.u2[-1]) / (2 * dt)
        dthetaJ -= velocity_dtheta * innov * dt
        d2thetaJ += (velocity_dtheta ** 2 - velocity_d2theta * innov) * dt
        propag_theta.swap()
        propag_dtheta.swap()
        propag_d2theta.swap()

    #ax_innov.plot(*np.array(innov_snapshots).T)
    #plt.show()

    residual = 0.1 * dthetaJ # / d2thetaJ
    current_theta = current_theta - residual
    print(f"Finished step {step}, theta: {current_theta}, dthetaJ: {dthetaJ}, d2thetaJ: {d2thetaJ}, residual {residual:.3e}")
    step += 1
    convergence_reached = abs(residual) < CONVERGENCE_EPS or step >= MAX_STEP or current_theta <= 0

if step >= MAX_STEP:
    print(f"Algorithm has exited due to max number of steps reached. Residual: {residual:.3e}")
elif current_theta <= 0:
    print(f"Algorithm has exited due to a negative theta parameter. Residual: {residual:.3e}")
else:
    print(f"Algorithm has exited due to convergence (abs(residual) < {CONVERGENCE_EPS:.4e}. Residual: {residual:.3e}")