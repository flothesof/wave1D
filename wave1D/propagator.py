import numpy as np
import configuration
import finite_element_operator as fe_op
import finite_element_space as fe_sp


class ElasticExplicitOrderTwo:
    """
    Definition of leap-frog like discrete propagators for elastic models.
    """
    def __init__(self, config=configuration.Elastic(), fe_space=fe_sp.FiniteElementSpace(),
                 mass_assembly_type=fe_op.AssemblyType.LUMPED,
                 stiffness_assembly_type=fe_op.AssemblyType.ASSEMBLED):
        """
        Constructor of discrete propagators.
        :param config: Elastic model configuration.
        :param fe_space: input finite element space.
        """
        self.config = config
        self.fe_space = fe_space
        self.mass_assembly_type = mass_assembly_type
        self.stiffness_assembly_type = stiffness_assembly_type

        self.u0 = None
        self.u1 = None
        self.u2 = None
        self.ustar = None
        self.operator1 = None
        self.operator2 = None
        self.rhs_operator = None
        self.inv_operator = None
        self.timestep = None

    def initialize(self, timestep):
        """
        Initializing discrete propagator.
        :param timestep: intput timestep.
        """
        self.timestep = timestep

        # Extracting number of DoF and index of boundary DoF.
        ndof = self.fe_space.get_ndof()
        left_idx = 0
        right_idx = ndof - 1

        # Allocating and applying initial conditions.
        self.u0 = np.zeros(ndof)
        self.u1 = np.zeros(ndof)
        self.u2 = np.zeros(ndof)
        self.ustar = np.zeros(ndof)

        # Assembling mass and stiffness operators.
        mass = fe_op.assemble_mass(self.config.alpha, self.fe_space, self.mass_assembly_type)
        stiffness = fe_op.assemble_stiffness(self.config.beta, self.fe_space, self.stiffness_assembly_type)

        # Computing operator to apply on u1.
        self.operator1 = fe_op.linear_combination(2.0, mass, -timestep ** 2, stiffness)

        # Computing operator to apply on u2.
        self.operator2 = fe_op.clone(-1.0, mass)
        self.__append_boundary_contribution_operator2(self.config.left_boundary_condition, left_idx)
        self.__append_boundary_contribution_operator2(self.config.right_boundary_condition, right_idx)

        # Computing rhs operator.
        if self.config.rhs is not None:
            self.rhs_operator = fe_op.assemble_mass(lambda x: 1.0, self.fe_space, self.mass_assembly_type)

        # Computing inv operator.
        self.inv_operator = mass
        self.__append_boundary_contribution_inv_operator(self.config.left_boundary_condition, left_idx)
        self.__append_boundary_contribution_inv_operator(self.config.right_boundary_condition, right_idx)
        fe_op.inv(self.inv_operator)

    def __append_boundary_contribution_operator2(self, bc, b_idx):
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

    def __append_boundary_contribution_inv_operator(self, bc, b_idx):
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

    def forward(self):
        """
        Forwarding discrete solver.
        """
        # Setting potential rhs.
        if self.config.rhs is None:
            self.ustar.fill(0.)

        # Appending previous steps contributions.
        fe_op.mlt_add(self.operator2, self.u2, self.ustar)
        fe_op.mlt_add(self.operator1, self.u1, self.ustar)

        # Appending potential boundary condition contributions.
        # TO DO !

        # Applying invert operator.
        fe_op.mlt(self.inv_operator, self.ustar, self.u0)



