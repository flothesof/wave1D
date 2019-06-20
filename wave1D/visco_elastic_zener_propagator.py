from enum import Enum
import numpy as np
import wave1D.configuration as configuration
import wave1D.finite_element_operator as fe_op
import wave1D.mass_assembler as mass_assembler
import wave1D.stiffness_assembler as stiffness_assembler
import wave1D.gradient_assembler as gradient_assembler


class InitialConditionType(Enum):
    """
    Definitions of initial condition types.
    """
    ORDERONE = 0
    ORDERTWO = 1
    NONE = 2


class ViscoElasticZener:
    """
    Definition of discrete propagators for visco elastic model with Zener constitutive law.
    """
    def __init__(self, config, fe_space, init_cond_type=InitialConditionType.NONE):
        """
        Constructor of discrete propagators.
        :param config: Elastic model configuration.
        :param fe_space: input finite element space.
        :param init_cond_type: type of initial condition.
        """
        self.config = config
        self.fe_space = fe_space
        self.init_cond_type = init_cond_type

        # Displacement unknown.
        self.u0 = None
        self.u1 = None
        self.u2 = None
        self.ustar = None

        # Displacement operators.
        self.operator1_u = None
        self.operator2_u = None
        self.rhs_operator_u = None
        self.inv_operator_u = None

        # Internal variable unknown.
        self.s0 = None
        self.s1 = None
        self.sstar = None

        # Internal variable operators.
        self.operator1_s = None
        self.inv_operator_s = None

        # Gradient operators.
        self.gradient = None
        self.transposed_gradient = None

        # Time values.
        self.timestep = None
        self.time = 0.0

    def initialize(self, timestep=None, cfl_factor=0.95):
        """
        Initializing discrete propagator.
        :param timestep: intput timestep, if timestep is None, time step is computed using CFL condition.
        :param cfl_factor: factor to be applied on CFL condition.
        """
        ndof_h1 = self.fe_space.get_ndof()
        ndof_l2 = self.fe_space.get_nelem() * self.fe_space.get_nlocaldof()

        # Allocating displacement unknown.
        self.u0 = np.zeros(ndof_h1)
        self.u1 = np.zeros(ndof_h1)
        self.u2 = np.zeros(ndof_h1)
        self.ustar = np.zeros(ndof_h1)

        # Allocating internal variable unknown.
        self.s0 = np.zeros(ndof_l2)
        self.s1 = np.zeros(ndof_l2)
        self.sstar = np.zeros(ndof_l2)

        # Extracting material parameteris.
        rho = self.config.density
        modulus = self.config.modulus
        eta = self.config.eta
        tau = self.config.tau

        # Definition of material parameters used in assembling procedures.
        def tau_s(x):
            return 1.0 / (tau(x) * (eta(x) - modulus(x)))

        def tau_prime_s(x):
            return 1.0 / (eta(x) - modulus(x))

        # Assembling displacement operators.
        mass_rho = mass_assembler.assemble_mass(self.fe_space, rho, fe_op.AssemblyType.LUMPED)
        stiffness_MR = stiffness_assembler.assemble_stiffness(self.fe_space, modulus)

        # Assembling internal variable operators.
        mass_tau_s = mass_assembler.assemble_discontinuous_mass(self.fe_space, tau_s, fe_op.AssemblyType.LUMPED)
        mass_tau_prime_s = mass_assembler.assemble_discontinuous_mass(self.fe_space, tau_prime_s, fe_op.AssemblyType.LUMPED)

        # Computing gradient operators.
        self.gradient = gradient_assembler.assemble_gradient(self.fe_space)
        self.transposed_gradient = gradient_assembler.assemble_transposed_gradient(self.fe_space)

        # Computing CFL or setting timestep
        if timestep is None:
            stiffness_k1 = stiffness_assembler.assemble_stiffness(self.fe_space, eta)
            cfl = 2.0 / np.sqrt(fe_op.spectral_radius(mass_rho, stiffness_k1))
            self.timestep = cfl_factor * cfl
        else:
            self.timestep = timestep

        # Computing operator applied on u1
        self.operator1_u = fe_op.linear_combination(2.0, mass_rho, -self.timestep ** 2, stiffness_MR)

        # Computing operator applied on u2
        self.operator2_u = fe_op.clone(-1.0, mass_rho)
        self.__add_boundary_contrib_operator2(self.config.left_boundary_condition, self.fe_space.get_left_idx())
        self.__add_boundary_contrib_operator2(self.config.right_boundary_condition, self.fe_space.get_right_idx())

        # Computing inv operator applied on displacement unknown.
        self.inv_operator_u = fe_op.clone(1.0, mass_rho)
        self.__add_boundary_contrib_inv_operator(self.config.left_boundary_condition, self.fe_space.get_left_idx())
        self.__add_boundary_contrib_inv_operator(self.config.right_boundary_condition, self.fe_space.get_right_idx())
        fe_op.inv(self.inv_operator_u)

        # Computing operator applied on s1.
        self.operator1_s = fe_op.linear_combination(1.0, mass_tau_prime_s, -self.timestep * 0.5, mass_tau_s)

        # Computing inv operator applied on internal variable.
        self.inv_operator_s = fe_op.linear_combination(1.0, mass_tau_prime_s, self.timestep * 0.5, mass_tau_s)
        fe_op.inv(self.inv_operator_s)

        # Computing rhs operator.
        if self.config.rhs is not None:
            raise NotImplementedError()

        # Applying initial conditions.
        if self.init_cond_type is not InitialConditionType.NONE:
            raise NotImplementedError()

    def forward(self):
        """
        Forwarding discrete solver.
        """
        # Initializing predictions.
        self.ustar.fill(0.)
        self.sstar.fill(0.)

        # Appending previous steps contributions into displacement prediction.
        fe_op.mlt_add(self.operator2_u, self.u2, self.ustar)
        fe_op.mlt_add(self.operator1_u, self.u1, self.ustar)
        fe_op.mlt_add(self.transposed_gradient, self.s1, self.ustar, coef=-self.timestep**2)

        # Appending potential boundary condition contributions.
        self.__add_boundary_contrib_prediction(self.config.left_boundary_condition, self.fe_space.get_left_idx())
        self.__add_boundary_contrib_prediction(self.config.right_boundary_condition, self.fe_space.get_right_idx())

        # Applying inverse operator on displacement prediction.
        fe_op.mlt(self.inv_operator_u, self.ustar, self.u0)

        # Appending previous step contribution into internal variable prediction.
        fe_op.mlt_add(self.operator1_s, self.s1, self.sstar)
        fe_op.mlt_add(self.gradient, self.u0, self.sstar, coef=1.0)
        fe_op.mlt_add(self.gradient, self.u1, self.sstar, coef=-1.0)

        # Applying inverse operator on internal variable predication.
        fe_op.mlt(self.inv_operator_s, self.sstar, self.s0)

    def swap(self):
        """
        Swapping DoF vectors.
        """
        # Swapping displacement unknown.
        u2_tmp = self.u2
        self.u2 = self.u1
        self.u1 = self.u0
        self.u0 = u2_tmp

        # Swapping internal variable unknown.
        s1_tmp = self.s1
        self.s1 = self.s0
        self.s0 = s1_tmp

        # Updating time value.
        self.time += self.timestep

    def reset(self):
        """
        Reseting propagator by setting zeros to every solution vectors and time value to 0.
        """
        self.u0.fill(0.)
        self.u1.fill(0.)
        self.u2.fill(0.)
        self.s0.fill(0.)
        self.s1.fill(0.)
        self.time = 0.

    def __add_boundary_contrib_operator2(self, bc, b_idx):
        """
        Appending contribution of boundary condition into operator 2.
        :param bc: boundary condition specifics.
        :param b_idx: index of boundary DoF.
        """
        if bc is not None:
            if bc.boundary_condition_type is configuration.BoundaryConditionType.ROBIN:
                fe_op.add_value(self.operator2_u, -0.5 * (self.timestep ** 2) * bc.param, b_idx, b_idx)
            elif bc.boundary_condition_type is configuration.BoundaryConditionType.ABSORBING:
                fe_op.add_value(self.operator2_u, 0.5 * self.timestep * bc.param, b_idx, b_idx)

    def __add_boundary_contrib_inv_operator(self, bc, b_idx):
        """
        Appending contribution of boundary condition into inv operator.
        :param bc: boudnary condition specifics.
        :param b_idx: index of boundary DoF.
        """
        if bc is not None:
            if bc.boundary_condition_type is configuration.BoundaryConditionType.DIRICHLET:
                fe_op.apply_pseudo_elimination(self.inv_operator_u, b_idx)
            elif bc.boundary_condition_type is configuration.BoundaryConditionType.ROBIN:
                fe_op.add_value(self.inv_operator_u, 0.5 * (self.timestep ** 2) * bc.param, b_idx, b_idx)
            elif bc.boundary_condition_type is configuration.BoundaryConditionType.ABSORBING:
                fe_op.add_value(self.inv_operator_u, 0.5 * self.timestep * bc.param, b_idx, b_idx)

    def __add_boundary_contrib_prediction(self, bc, b_idx):
        """
        Appending contribution of boundary condition into prediction vector.
        :param bc: boundary condition specifics.
        :param b_idx: index of boundary DoF.
        """
        if bc is not None:
            if bc.boundary_condition_type is configuration.BoundaryConditionType.DIRICHLET:
                self.ustar[b_idx] = bc.value(self.time)
            else:
                self.ustar[b_idx] += self.timestep * self.timestep * bc.value(self.time)

