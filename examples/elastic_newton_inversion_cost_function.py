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
from tqdm import tqdm

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
THETA_INIT = 2.2
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


propag_builder = partial(elastic_propagator.Elastic, fe_space=fe_space,
                         mass_assembly_type=fe_op.AssemblyType.LUMPED,
                         stiffness_assembly_type=fe_op.AssemblyType.ASSEMBLED)

theta_grid = np.linspace(1, 3, num=50)

Jthetas = []
for current_theta in tqdm(theta_grid):
    Jtheta = 0.
    current_beta = partial(beta, theta=current_theta)
    config_theta = configuration.Elastic(alpha=alpha, beta=current_beta, left_bc=left_bc)
    propag_theta = propag_builder(config=config_theta)
    propag_theta.initialize()
    while propag_theta.time < TMAX:
        propag_theta.forward()
        innov = interp_observer(propag_theta.time) - (propag_theta.u0[-1] - propag_theta.u2[-1]) / (2 * propag_theta.timestep)
        Jtheta += 0.5 * innov ** 2
        propag_theta.swap()
    Jthetas.append(Jtheta)

fig, ax = plt.subplots()
ax.plot(theta_grid, Jthetas)
plt.show()