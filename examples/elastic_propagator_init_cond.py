import matplotlib.pyplot as plt
import numpy as np
import wave1D.configuration as configuration
import wave1D.elastic_propagator as elastic_propagator
import wave1D.functional as functional
import wave1D.finite_element_space as fe_sp
import wave1D.finite_element_operator as fe_op
import wave1D.lagrange_polynomial as lag_poly
import wave1D.mesh as mesh
import wave1D.mass_assembler as mass_assembler


def analytical_solution(k, x, t):
    return [np.sqrt(2.0) * np.cos(k * np.pi * t) * np.cos(k * np.pi * xx) for xx in x]


def make_propagator(k, nelem, fe_order, fe_type, quad_order, quad_type, mass_assembly_type, stiffness_assembly_type):
    # Creating configuration.
    config = configuration.Elastic(init_field=lambda x: np.sqrt(2.0) * np.cos(k * np.pi * x))

    # Creating mesh.
    msh = mesh.make_mesh_from_npt(0.0, 1.0, nelem + 1)

    # Creating finite element space.
    fe_space = fe_sp.FiniteElementSpace(msh, fe_order, fe_type, quad_order, quad_type)

    # Creating propagator.
    return elastic_propagator.ElasticExplicitOrderTwo(config, fe_space, mass_assembly_type, stiffness_assembly_type,
                                                      init_cond_type=elastic_propagator.InitialConditionType.ORDERONE)


k = 6
nstep = 600
everynstep = 100

# Creating numerical propagator.
numerical_propagator = make_propagator(k, 300,
       4, lag_poly.PointDistributionType.GAUSS_LOBATTO,
       4, lag_poly.PointDistributionType.GAUSS_LOBATTO,
       fe_op.AssemblyType.LUMPED,
       fe_op.AssemblyType.ASSEMBLED)

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
ax.set_ylim((-2, 2))
for i in range(1000):
    numerical_propagator.forward()
    analytical_sol = analytical_solution(k, x, i * dt)
    numerical_lines[0].set_ydata(numerical_propagator.u0)
    analytical_lines[0].set_ydata(analytical_sol)
    ax.set_title('istep = {}'.format(i))
    plt.pause(0.01)
    numerical_propagator.swap()
plt.show()