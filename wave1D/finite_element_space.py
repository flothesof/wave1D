class FiniteElementSpace:
    """
    Definition of 1D finite element spaces;
    """
    def __init__(self):
        return self

    def get_ndof(self):
        """
        :return: number of Degrees of Freedom (DoF) in finite element space.
        """
        return 0

    def get_nelem(self):
        """
        :return: number of elements in a finite element space.
        """
        return 0

    def get_nlocaldof(self):
        """
        :return: number of Degrees of Freedom (DoF) in one element.
        """
        return 0