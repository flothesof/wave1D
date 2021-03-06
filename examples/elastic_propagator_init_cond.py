import matplotlib.pyplot as plt
import numpy as np
import wave1D.configuration as configuration
import wave1D.elastic_propagator as elastic_propagator
import wave1D.finite_element_space as fe_sp
import wave1D.mesh as mesh


def analytical_solution(k, x, t):
    return [np.sqrt(2.0) * np.cos(k * np.pi * t) * np.cos(k * np.pi * xx) for xx in x]


k = 5
nstep = 600
everynstep = 100

# Creating configuration.
config = configuration.Elastic(init_field=lambda x: np.sqrt(2.0) * np.cos(k * np.pi * x))

# Creating finite element space.
fe_space = fe_sp.FiniteElementSpace(mesh.make_mesh_from_npt(0.0, 1.0, 6), fe_order=5, quad_order=5)

# Creating propagator.
numerical_propagator = elastic_propagator.Elastic(config, fe_space,
                                                      init_cond_type=elastic_propagator.InitialConditionType.ORDERTWO)

# Initializing numerical propagator.
numerical_propagator.initialize()

# Analytical solution.
x = numerical_propagator.fe_space.get_dof_coords()
dt = numerical_propagator.timestep
analytical_sol = analytical_solution(k, x, 0)

# Runing.
fig, ax = plt.subplots()
numerical_lines = ax.plot(x, numerical_propagator.u0)
analytical_lines = ax.plot(x, analytical_sol)
ax.plot(x, np.zeros_like(x), '*')
ax.set_ylim((-2, 2))
for i in range(200):
    numerical_propagator.forward()
    analytical_sol = analytical_solution(k, x, (i + 1) * dt)
    numerical_lines[0].set_ydata(numerical_propagator.u1)
    analytical_lines[0].set_ydata(analytical_sol)
    ax.set_title('istep = {}'.format(i))
    plt.pause(0.1)
    numerical_propagator.swap()
plt.show()