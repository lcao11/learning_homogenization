# Note: This file is almost entirely copied from hippylib library. Please cite hippylib if you use this code.
import ufl
import dolfin as dl
import hippylib as hp
import numpy as np
import numbers
from .periodic_boundary import PeriodicBoundary1D, PeriodicBoundary2D, PeriodicBoundary3D


class _BilaplacianR:
    """
    Operator that represent the action of the regularization/precision matrix
    for the Bilaplacian prior.
    """

    def __init__(self, A, Msolver):
        self.A = A
        self.Msolver = Msolver

        self.help1, self.help2 = dl.Vector(self.A.mpi_comm()), dl.Vector(self.A.mpi_comm())
        self.A.init_vector(self.help1, 0)
        self.A.init_vector(self.help2, 1)

    def init_vector(self, x, dim):
        self.A.init_vector(x, 1)

    def mpi_comm(self):
        return self.A.mpi_comm()

    def mult(self, x, y):
        self.A.mult(x, self.help1)
        self.Msolver.solve(self.help2, self.help1)
        self.A.mult(self.help2, y)


class _BilaplacianRsolver:
    """
    Operator that represent the action of the inverse the regularization/precision matrix
    for the Bilaplacian prior.
    """

    def __init__(self, Asolver, M):
        self.Asolver = Asolver
        self.M = M

        self.help1, self.help2 = dl.Vector(self.M.mpi_comm()), dl.Vector(self.M.mpi_comm())
        self.init_vector(self.help1, 0)
        self.init_vector(self.help2, 0)

    def init_vector(self, x, dim):
        self.M.init_vector(x, 1)

    def solve(self, x, b):
        nit = self.Asolver.solve(self.help1, b)
        self.M.mult(self.help1, self.help2)
        nit += self.Asolver.solve(x, self.help2)
        return nit


class Periodic_SqrtPrecisionPDE_Prior():
    """
    This class implement a prior model with covariance matrix
    :math:`C = A^{-1} M A^-1`,
    where A is the finite element matrix arising from discretization of sqrt_precision_varf_handler

    """

    def __init__(self, Vh, sqrt_precision_varf_handler, rel_tol=1e-12, max_iter=1000):
        """
        Construct the prior model.
        Input:
        - :code:`Vh`:              the finite element space for the parameter
        - :code:sqrt_precision_varf_handler: the PDE representation of the  sqrt of the covariance operator
        - :code:`mean`:            the prior mean
        """

        self.Vh = Vh
        dim = self.Vh.mesh().geometric_dimension()
        if dim == 1:
            pbc = PeriodicBoundary1D()
        elif dim == 2:
            pbc = PeriodicBoundary2D()
        elif dim == 3:
            pbc = PeriodicBoundary3D()

        trial = dl.TrialFunction(self.Vh)
        test = dl.TestFunction(self.Vh)

        varfM = ufl.inner(trial, test) * ufl.dx
        self.M = dl.assemble(varfM)
        self.Msolver = hp.PETScKrylovSolver(self.Vh.mesh().mpi_comm(), "cg", "jacobi")
        self.Msolver.set_operator(self.M)
        self.Msolver.parameters["maximum_iterations"] = max_iter
        self.Msolver.parameters["relative_tolerance"] = rel_tol
        self.Msolver.parameters["error_on_nonconvergence"] = True
        self.Msolver.parameters["nonzero_initial_guess"] = False

        self.A = dl.assemble(sqrt_precision_varf_handler(trial, test))
        self.Asolver = hp.PETScKrylovSolver(self.Vh.mesh().mpi_comm(), "cg", hp.amg_method())
        self.Asolver.set_operator(self.A)
        self.Asolver.parameters["maximum_iterations"] = max_iter
        self.Asolver.parameters["relative_tolerance"] = rel_tol
        self.Asolver.parameters["error_on_nonconvergence"] = True
        self.Asolver.parameters["nonzero_initial_guess"] = False

        old_qr = dl.parameters["form_compiler"]["quadrature_degree"]
        dl.parameters["form_compiler"]["quadrature_degree"] = -1
        qdegree = 2 * Vh._ufl_element.degree()
        metadata = {"quadrature_degree": qdegree}

        representation_old = dl.parameters["form_compiler"]["representation"]
        dl.parameters["form_compiler"]["representation"] = "quadrature"

        num_sub_spaces = self.Vh.num_sub_spaces()
        if num_sub_spaces <= 1:  # SCALAR PARAMETER
            element = dl.FiniteElement("Quadrature", self.Vh.mesh().ufl_cell(), qdegree, quad_scheme="default")
        else:  # Vector FIELD PARAMETER
            element = dl.VectorElement("Quadrature", self.Vh.mesh().ufl_cell(),
                                       qdegree, dim=num_sub_spaces, quad_scheme="default")
        Qh = dl.FunctionSpace(self.Vh.mesh(), element, constrained_domain=pbc)

        ph = dl.TrialFunction(Qh)
        qh = dl.TestFunction(Qh)
        Mqh = dl.assemble(ufl.inner(ph, qh) * ufl.dx(metadata=metadata))
        if num_sub_spaces <= 1:
            one_constant = dl.Constant(1.)
        else:
            one_constant = dl.Constant(tuple([1.] * num_sub_spaces))
        ones = dl.interpolate(one_constant, Qh).vector()
        dMqh = Mqh * ones
        Mqh.zero()
        dMqh.set_local(ones.get_local() / np.sqrt(dMqh.get_local()))
        Mqh.set_diagonal(dMqh)
        MixedM = dl.assemble(ufl.inner(ph, test) * ufl.dx(metadata=metadata))
        self.sqrtM = hp.MatMatMult(MixedM, Mqh)

        dl.parameters["form_compiler"]["quadrature_degree"] = old_qr
        dl.parameters["form_compiler"]["representation"] = representation_old

        self.R = _BilaplacianR(self.A, self.Msolver)
        self.Rsolver = _BilaplacianRsolver(self.Asolver, self.M)
        self.mean = dl.Function(Vh).vector()

    def init_vector(self, x, dim):
        """
        Inizialize a vector :code:`x` to be compatible with the range/domain of :math:`R`.

        If :code:`dim == "noise"` inizialize :code:`x` to be compatible with the size of
        white noise used for sampling.
        """
        if dim == "noise":
            self.sqrtM.init_vector(x, 1)
        else:
            self.A.init_vector(x, dim)

    def sample(self, noise, s, add_mean=True):
        """
        Given :code:`noise` :math:`\\sim \\mathcal{N}(0, I)` compute a sample :code:`s` from the prior.
        If :code:`add_mean == True` add the prior mean value to :code:`s`.
        """
        rhs = self.sqrtM * noise
        self.Asolver.solve(s, rhs)
    
    def cost(self,m):
        d = self.mean.copy()
        d.axpy(-1., m)
        Rd = dl.Vector(self.R.mpi_comm())
        self.init_vector(Rd,0)
        self.R.mult(d,Rd)
        return .5*Rd.inner(d)


def PeriodicBiLaplacianPrior(Vh, gamma, delta, Theta=None, rel_tol=1e-12, max_iter=1000):
    """
    This function construct an instance of :code"`SqrtPrecisionPDE_Prior`  with covariance matrix
    :math:`C = (\\delta I + \\gamma \\mbox{div } \\Theta \\nabla) ^ {-2}`.

    The magnitude of :math:`\\delta\\gamma` governs the variance of the samples, while
    the ratio :math:`\\frac{\\gamma}{\\delta}` governs the correlation lenght.

    Here :math:`\\Theta` is a SPD tensor that models anisotropy in the covariance kernel.

    Input:

    - :code:`Vh`:              the finite element space for the parameter
    - :code:`gamma` and :code:`delta`: the coefficient in the PDE (floats, dl.Constant, dl.Expression, or dl.Function)
    - :code:`Theta`:           the SPD tensor for anisotropic diffusion of the PDE
    - :code:`rel_tol`:         relative tolerance for solving linear systems involving covariance matrix
    - :code:`max_iter`:        maximum number of iterations for solving linear systems involving covariance matrix
    """
    if isinstance(gamma, numbers.Number):
        gamma = dl.Constant(gamma)

    if isinstance(delta, numbers.Number):
        delta = dl.Constant(delta)

    def sqrt_precision_varf_handler(trial, test):
        if Theta is None:
            varfL = gamma * ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx
        else:
            varfL = gamma * ufl.inner(Theta * ufl.grad(trial), ufl.grad(test)) * ufl.dx

        varfM = delta * ufl.inner(trial, test) * ufl.dx

        return varfL + varfM

    return Periodic_SqrtPrecisionPDE_Prior(Vh, sqrt_precision_varf_handler, rel_tol, max_iter)
