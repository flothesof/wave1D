from enum import Enum
import numpy as np
import wave1D.configuration as configuration
import wave1D.finite_element_operator as fe_op
import wave1D.finite_element_space as fe_sp
import wave1D.mass_assembler as mass_assembler
import wave1D.stiffness_assembler as stiffness_assembler


class InitialConditionType(Enum):
    """
    Definitions of initial condition types.
    """
    ORDERONE = 0
    ORDERTWO = 1
    NONE = 2


class ElasticExplicitOrderTwo:
    """
    Definition of leap-frog like discrete propagators for elastic models.
    """
    def __init__(self, config, fe_space,
                 mass_assembly_type=fe_op.AssemblyType.LUMPED, stiffness_assembly_type=fe_op.AssemblyType.ASSEMBLED,
                 init_cond_type=InitialConditionType.NONE):
        """
        Constructor of discrete propagators.
        :param config: Elastic model configuration.
        :param fe_space: input finite element space.
        """
        self.config = config
        self.fe_space = fe_space
        self.mass_assembly_type = mass_assembly_type
        self.stiffness_assembly_type = stiffness_assembly_type
        self.init_cond_type = init_cond_type

        self.u0 = None
        self.u1 = None
        self.u2 = None
        self.ustar = None
        self.operator1 = None
        self.operator2 = None
        self.rhs_operator = None
        self.inv_operator = None
        self.timestep = None
        self.time = 0.0

    def initialize(self, timestep=None, cfl_factor=0.95):
        """
        Initializing discrete propagator.
        :param timestep: intput timestep, if timestep is None, time step is computed using CFL condition.
        :param cfl_factor: factor to be applied on CFL condition.
        """
        # Allocating and applying initial conditions.
        ndof = self.fe_space.get_ndof()
        self.u0 = np.zeros(ndof)
        self.u1 = np.zeros(ndof)
        self.u2 = np.zeros(ndof)
        self.ustar = np.zeros(ndof)

        # Assembling mass and stiffness operators.
        mass = mass_assembler.assemble_mass(self.fe_space, self.config.alpha, self.mass_assembly_type)
        stiffness = stiffness_assembler.assemble_stiffness(self.fe_space, self.config.beta, self.stiffness_assembly_type)

        # Computing CFL or setting timestep.
        if timestep is None:
            cfl = 2.0 / np.sqrt(fe_op.spectral_radius(mass, stiffness))
            self.timestep = cfl_factor * cfl
        else:
            self.timestep = timestep

        # Computing operator to apply on u1.
        self.operator1 = fe_op.linear_combination(2.0, mass, -self.timestep ** 2, stiffness)

        # Computing operator to apply on u2.
        self.operator2 = fe_op.clone(-1.0, mass)
        self.__add_boundary_contrib_operator2(self.config.left_boundary_condition, self.fe_space.get_left_idx())
        self.__add_boundary_contrib_operator2(self.config.right_boundary_condition, self.fe_space.get_right_idx())

        # Computing rhs operator.
        if self.config.rhs is not None:
            self.rhs_operator = mass_assembler.assemble_mass(lambda x: 1.0, self.fe_space, self.mass_assembly_type)

        # Computing inv operator.
        self.inv_operator = fe_op.clone(1.0, mass)
        self.__add_boundary_contrib_inv_operator(self.config.left_boundary_condition, self.fe_space.get_left_idx())
        self.__add_boundary_contrib_inv_operator(self.config.right_boundary_condition, self.fe_space.get_right_idx())
        fe_op.inv(self.inv_operator)

        # Applying initial conditions.
        if self.init_cond_type is InitialConditionType.ORDERONE:

            for i, x in enumerate(self.fe_space.get_dof_coords()):
                self.u2[i] = self.config.init_field(x)
                self.u1[i] = self.timestep * self.config.init_velocity(x) + self.u2[i]

        elif self.init_cond_type is InitialConditionType.ORDERTWO:

            for i, x in enumerate(self.fe_space.get_dof_coords()):
                self.u2[i] = self.config.init_field(x)

            fe_op.mlt(stiffness, self.u2, self.ustar)
            fe_op.inv(mass)
            fe_op.mlt(mass, self.ustar, self.u1)

            for i, x in enumerate(self.fe_space.get_dof_coords()):
                self.u1[i] = self.timestep * self.config.init_velocity(x) + self.u2[i] \
                             - 0.5 * (self.timestep ** 2) * self.u1[i]

    def forward(self):
        """
        Forwarding discrete solver.
        """
        # Setting potential rhs.
        if self.config.rhs is None:
            self.ustar.fill(0.)
        else:
            raise NotImplementedError()

        # Appending previous steps contributions.
        fe_op.mlt_add(self.operator2, self.u2, self.ustar)
        fe_op.mlt_add(self.operator1, self.u1, self.ustar)

        # Appending potential boundary condition contributions.
        self.__add_boundary_contrib_prediction(self.config.left_boundary_condition, self.fe_space.get_left_idx())
        self.__add_boundary_contrib_prediction(self.config.right_boundary_condition, self.fe_space.get_right_idx())

        # Applying invert operator.
        fe_op.mlt(self.inv_operator, self.ustar, self.u0)

    def swap(self):
        """
        Swapping DoF vectors.
        """
        u2_tmp = self.u2
        self.u2 = self.u1
        self.u1 = self.u0
        self.u0 = u2_tmp
        self.time += self.timestep

    def reset(self):
        """
        Reseting propagator by setting zeros to every solution vectors and time value to 0.
        """
        self.u0.fill(0.)
        self.u1.fill(0.)
        self.u2.fill(0.)
        self.time = 0.

    def __add_boundary_contrib_operator2(self, bc, b_idx):
        """
        Appending contribution of boundary condition into operator 2.
        :param bc: boundary condition specifics.
        :param b_idx: index of boundary DoF.
        """
        if bc is not None:
            if bc.boundary_condition_type is configuration.BoundaryConditionType.ROBIN:
                fe_op.add_value(self.operator2, -0.5 * (self.timestep ** 2) * bc.param, b_idx, b_idx)
            elif bc.boundary_condition_type is configuration.BoundaryConditionType.ABSORBING:
                fe_op.add_value(self.operator2, 0.5 * self.timestep * bc.param, b_idx, b_idx)

    def __add_boundary_contrib_inv_operator(self, bc, b_idx):
        """
        Appending contribution of boundary condition into inv operator.
        :param bc: boudnary condition specifics.
        :param b_idx: index of boundary DoF.
        """
        if bc is not None:
            if bc.boundary_condition_type is configuration.BoundaryConditionType.DIRICHLET:
                fe_op.apply_pseudo_elimination(self.inv_operator, b_idx)
            elif bc.boundary_condition_type is configuration.BoundaryConditionType.ROBIN:
                fe_op.add_value(self.inv_operator, 0.5 * (self.timestep ** 2) * bc.param, b_idx, b_idx)
            elif bc.boundary_condition_type is configuration.BoundaryConditionType.ABSORBING:
                fe_op.add_value(self.inv_operator, 0.5 * self.timestep * bc.param, b_idx, b_idx)

    def __add_boundary_contrib_prediction(self, bc, b_idx):
        """
        Appending contribution of boundary condition into prediction vector.
        :param bc: boudnary condition specifics.
        :param b_idx: index of boundary DoF.
        """
        if bc is not None:
            if bc.boundary_condition_type is configuration.BoundaryConditionType.DIRICHLET:
                self.ustar[b_idx] = bc.value(self.time)
            else:
                self.ustar[b_idx] += self.timestep * self.timestep * bc.value(self.time)
