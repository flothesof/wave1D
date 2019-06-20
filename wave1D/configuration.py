from enum import Enum


class BoundaryConditionType(Enum):
    """
    Definitions of boundary condition types.
    """
    DIRICHLET = 0
    ROBIN = 1
    ABSORBING = 2
    NONE = 3


class BoundaryCondition:
    """
    Definition of boundary conditions.
    """
    def __init__(self, boundary_condition_type=BoundaryConditionType.NONE, value=lambda t: 0.0, param=0.0):
        """
        Constructor of boundary conditions.
        :param boundary_condition_type: enum member of BoundaryConditionType.
        :param value: function of time.
        :param param: potential parameter  as positive constant value.
        """
        self.boundary_condition_type = boundary_condition_type
        self.value = value
        self.param = param
        if self.param < 0.0:
            raise ValueError("Negative value in boundary condition parameter not supported.")


class RightHandSide:
    """
    Definition of rhigt hand sides.
    """
    def __init__(self, space_time_func=lambda x, t: 0.0):
        """
        Constructor of right-hand sides.
        :param space_time_func: function of space and time representing the rhs value.
        """
        self.value = space_time_func


class Elastic:
    def __init__(self, alpha=lambda x: 1.0, beta=lambda x: 1.0, init_field=lambda x: 0.0, init_velocity=lambda x: 0.0,
                 left_bc=None, right_bc=None, rhs=None):
        """
        Constructor for Elastic configurations. Input arguments are expected to be lambda depending on space variable.
        :param alpha: is the inertia coefficient,
        :param beta: is the constitutive law parameter.
        :param init_field: field initial condition.
        :param init_velocity: velocity initial condition.
        :param left_bc: boundary condition on left point (x=0).
        :param right_bc: boudnary condition on right point (x=L).
        :param rhs: right-hand side in time scheme.
        """
        self.alpha = alpha
        self.beta = beta
        self.init_field = init_field
        self.init_velocity = init_velocity
        self.left_boundary_condition = left_bc
        self.right_boundary_condition = right_bc
        self.rhs = rhs


class ViscoElasticMaxwell:
    def __init__(self, density=lambda x: 1.0, modulus=lambda x: 1.0, eta=lambda x: 0.0, init_field=lambda x: 0.0,
                 init_stress=lambda x: 0.0, left_bc=None, right_bc=None, rhs=None):
        """
        Constructor for visco-elastic model with Maxwell constitutive law. Input arguments are expected to be a
        lambda depending on the space variable.
        :param density: is the mass density.
        :param modulus: is the elasticity coefficient.
        :param eta: is the attenuation coefficient.
        :param init_field: field initial condition.
        :param init_stress: stress initial condition.
        :param left_bc: boundary condition on left point (x=0).
        :param right_bc: boundary condition on right point (x=L).
        :param rhs: right-hand side in time scheme.
        """
        self.density = density
        self.modulus = modulus
        self.eta = eta
        self.init_field = init_field
        self.init_stress = init_stress
        self.left_boundary_condition = left_bc
        self.right_boundary_condition = right_bc
        self.rhs = rhs


class ViscoElasticKelvinVoigt:
    def __init__(self, density=lambda x: 1.0, modulus=lambda x: 1.0, eta=lambda x: 1.0, init_field=lambda x: 0.0,
                 init_velocity=lambda x: 0.0, left_bc=None, right_bc=None, rhs=None):
        """
        Constructor for visco-elastic model with Kelvin Voigt constitutive law. Input arguments are expected to be a
        lambda depending on the space variable.
        :param density: is the mass density.
        :param modulus: is the elasticity coefficient.
        :param eta: is the attenuation coefficient.
        :param init_field: field initial condition.
        :param init_velocity: velocity initial condition.
        :param left_bc: boundary condition on left point (x=0).
        :param right_bc: boundary condition on right point (x=L).
        :param rhs: right-hand side in time scheme.
        """
        self.density = density
        self.modulus = modulus
        self.eta = eta
        self.init_field = init_field
        self.init_velocity = init_velocity
        self.left_boundary_condition = left_bc
        self.right_boundary_condition = right_bc
        self.rhs = rhs


class ViscoElasticZener:
    def __init__(self, density=lambda x: 1.0, modulus=lambda x: 1.0, eta=lambda x: 1.0, tau=lambda x: 1.0,
                 init_field=lambda x: 0.0, init_viscous_stress=lambda x: 0.0, left_bc=None, right_bc=None, rhs=None):
        """
        Constructor for visco-elastic model with Kelvin Voigt constitutive law. Input arguments are expected to be a
        lambda depending on the space variable.
        :param density: is the mass density.
        :param modulus: is the first elasticity coefficient.
        :param eta: is the viscosity coefficient.
        :param tau: is the relaxation time.
        :param init_field: field initial condition.
        :param init_viscous_stress: initial condition for the viscous part of the stress.
        :param left_bc: boundary condition on left point (x=0).
        :param right_bc: boundary condition on right point (x=L).
        :param rhs: right-hand side in time scheme.
        """
        self.density = density
        self.modulus = modulus
        self.eta = eta
        self.tau = tau
        self.init_field = init_field
        self.init_viscous_stress = init_viscous_stress
        self.left_boundary_condition = left_bc
        self.right_boundary_condition = right_bc
        self.rhs = rhs
