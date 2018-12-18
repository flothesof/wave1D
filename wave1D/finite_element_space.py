import numpy as np
import lagrange_polynomial as lag_poly
import mesh as msh


class FiniteElementSpace:
    """
    Definition of 1D finite element spaces;
    """

    def __init__(self, mesh=msh.make_mesh_from_npt(),
                 fe_order=1, basis_type=lag_poly.PointDistributionType.GAUSS_LOBATTO,
                 quad_order=1, quad_type=lag_poly.PointDistributionType.GAUSS_LOBATTO):
        self.mesh = mesh
        self.local_dofs, _ = lag_poly.make_quadrature_formula(fe_order, basis_type)
        self.quad_pnts, self.quad_weights = lag_poly.make_quadrature_formula(quad_order, quad_type)
        self.basis = lag_poly.eval_lagrange_polynomials(self.local_dofs, self.quad_pnts)
        self.dbasis = lag_poly.eval_lagrange_polynomials_derivatives(self.local_dofs, self.quad_pnts)

    def get_ndof(self):
        """
        :return: number of Degrees of Freedom (DoF) in finite element space.
        """
        return self.get_nelem() * (self.get_nlocaldof() - 1) + 1

    def get_nelem(self):
        """
        :return: number of elements in a finite element space.
        """
        return len(self.mesh.pnts) - 1

    def get_elem_length(self, elem_idx):
        """
        :param: elem_idx: input element index.
        :return: return length of an element in the mesh.
        """
        return abs(self.mesh.pnts[elem_idx + 1] - self.mesh.pnts[elem_idx])

    def get_nlocaldof(self):
        """
        :return: number of Degrees of Freedom (DoF) in one element.
        """
        return len(self.local_dofs)

    def get_left_idx(self):
        """
        :return: index of far left DoF.
        """
        return 0

    def get_right_idx(self):
        """
        :return: index of far right DoF.
        """
        return self.get_ndof() - 1

    def get_coord(self, elem_idx, s):
        """
        Extracting coordinate in an element from a parametric coordinate in the reference element.
        :param elem_idx: element index in the mesh
        :param s: parametrix coordinate in the reference element.
        """
        return self.mesh.pnts[elem_idx] + s * self.get_elem_length(elem_idx)

    def locals_to_globals(self, elem_idx):
        """
        Extracting index mappings from local numbering to global numbering
        :param elem_idx: element index in the mesh.
        :return: a 1D array regrouping the global indexes.
        """
        offset = elem_idx * (self.get_nlocaldof() - 1)
        return np.linspace(0, self.get_nlocaldof() - 1, self.get_nlocaldof(), dtype=int) + offset

    def get_quadrature_weight(self, quad_pnt_idx):
        """
        Extracting value of a quadrature weight.
        :param quad_pnt_idx: index of a quadrature point.
        :return: the weight associated to a quadrature point.
        """
        return self.quad_weights[quad_pnt_idx]

    def eval_at_quadrature_pnts(self, func=lambda k, s: 0.0):
        """
        Evaluating function defined on the reference element at the quadrature points.
        :param func: a input function depending on the index of the quadrature points and a coordinate variable in the
        reference element
        :return: a vector gathering the evaluation of the function at the quadrature points.
        """
        result = np.zeros(len(self.quad_pnts))
        for iq in range(len(self.quad_pnts)):
            result[iq] = func(iq, self.quad_pnts[iq])
        return result

    def apply_basis_diag_basis(self, diag):
        """
        Computing the result of basis * D * basis^T, where basis are the Lagrange basis functions in the reference
        element evaluated at the quadrature points.
        :param diag: the diagonal of a matrix of dimension equals to the number of quadrature points
        in the reference element.
        :return: a matrix resulting from basis * D * basis^T
        """
        return np.dot(self.basis, np.dot(np.diag(diag), self.basis.transpose()))

    def apply_dbasis_diag_dbasis(self, diag):
        """
        Computing the result of dbasis * D * dbasis^T, where dbasis are the first derivative of the Lagrange basis
        functions in the reference element evaluated at the quadrature points.
        :param diag: the diagonal of a matrix of dimension equals to the number of quadrature points
        in the reference element.
        :return: a matrix resulting from dbasis * D * basis^T
        """
        return np.dot(self.dbasis, np.dot(np.diag(diag), self.dbasis.transpose()))
