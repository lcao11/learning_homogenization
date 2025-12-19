import dolfin as dl
import hippylib as hp
from typing import Any


class Projector:
    """
    A class for projection in between function spaces.
    """

    def __init__(self, V_in: dl.FunctionSpace, V_out: dl.FunctionSpace, max_iter: int = 500,
                 rel_tol: float = 1.e-10) -> None:
        """
        :param V_in: The input function space
        :param V_out: The output function sapce
        :param max_iter: The maximum iteration required for Krylov solver in 3D
        :param rel_tol: The relative tolerance required for Krylov solver in 3D
        """
        out_trial = dl.TrialFunction(V_out)
        self._out_test = dl.TestFunction(V_out)
        lhs_matrix = dl.as_backend_type(dl.assemble(dl.inner(out_trial, self._out_test) * dl.dx))
        dim = V_out.mesh().geometric_dimension()
        if dim <= 2:
            self._solver = hp.PETScLUSolver(V_out.mesh().mpi_comm())
            self._solver.set_operator(lhs_matrix)
        else:
            self._solver = hp.PETScKrylovSolver(V_out.mesh().mpi_comm(), "cg", "jacobi")
            self._solver.set_operator(lhs_matrix)
            self._solver.parameters["maximum_iterations"] = max_iter
            self._solver.parameters["relative_tolerance"] = rel_tol
            self._solver.parameters["error_on_nonconvergence"] = True
            self._solver.parameters["nonzero_initial_guess"] = False
            # self._solver = hp.PETScLUSolver(V_out.mesh().mpi_comm(), method="mumps")
            # self._solver.set_operator(lhs_matrix)
        self._V_in = V_in
        self._V_out = V_out
        self._help_in = dl.Function(self._V_in)

    def project(self, u: Any) -> dl.Function:
        """
        :param u: the input vector, function, or expression
        :return: an output function
        """
        out = dl.Function(self._V_out)
        if isinstance(u, dl.cpp.la.PETScVector):
            self._help_in.vector().zero()
            self._help_in.vector().axpy(1., u)
            rhs = dl.assemble(dl.inner(self._help_in, self._out_test) * dl.dx)
        else:
            rhs = dl.assemble(dl.inner(u, self._out_test) * dl.dx)
        self._solver.solve(out.vector(), rhs)
        return out
