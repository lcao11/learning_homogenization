import numpy as np
import ufl
import dolfin as dl
import hippylib as hp
from .utils.visualization import extract_component
from .utils.projection import Projector
from .microstructure import MatrixInterpolation
from .utils.solver_utils import (
    print_compute_time,
    LiftingFunction,
    type_error_message,
    identity_4th,
    create_mesh,
)
import time
from typing import Any

# 6th-order finite difference coefficients for finite difference estimation of the memory kernel
coeff_6th_side = np.array([-49 / 20, 6, -15 / 2, 20 / 3, -15 / 4, 6 / 5, -1 / 6])
coeff_6th_central = np.array([-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60])
coeff_k = [1.0 / 6, 1.0 / 3, 1.0 / 3, 1.0 / 6]
coeff_s = [0, 0.5, 0.5, 1.0]
index_s = [2, 1, 1, 0]


def KelvinVoigtSettings():
    """Settings for Kelvin--Voigt viscoelasticity simulation in a unit cell."""

    settings = {"n_cells": 64, "dim": 2, "FE_order": 1, "direct_solver": True}
    return settings


def _tensor2array_index(d: int) -> np.ndarray:
    """Return the index map for flattening a symmetric tensor.

    This defines the ordering used throughout this module for the symmetric components.
    The ordering is the upper-triangular traversal

    - 2D: (11, 12, 22)
    - 3D: (11, 12, 13, 22, 23, 33)

    i.e. indices are assigned by increasing row `i` and then increasing column `j>=i`.

    Parameters
    ----------
    d:
        Spatial dimension.

    Returns
    -------
    np.ndarray
        Integer array `idx` of shape `(d, d)` such that `idx[i,j]` gives the position in a length-`d_sym`
        vector corresponding to the symmetric component `(min(i,j), max(i,j))`.
    """

    out = np.zeros((d, d)).astype(int)
    for i in range(d):
        for j in range(d):
            if i <= j:
                row, col = i, j
            else:
                row, col = j, i
            out[i, j] = (2 * d - row) * (row + 1) // 2 - d + col
    return out


def array2tensor_mapping(d: int) -> np.ndarray:
    r"""
        Kelvin--Mandel mapping from a symmetric-tensor vector representation to a symmetric tensor.

        Convention: the input array is in Kelvin--Mandel form, i.e. diagonal components are
        unchanged and off-diagonal (shear) components are scaled by $\sqrt{2}$.

        For example in 2D with ordering (11, 12, 22):
            eps_vec = [eps_11, sqrt(2)*eps_12, eps_22]
            eps_tensor[0,1] = eps_tensor[1,0] = eps_vec[1] / sqrt(2)

        :param d: the spatial dimension
        :return: transform tensor of shape (d, d, d_sym) such that eps = einsum(transform, eps_vec)
    """

    sym_d = int(d * (d + 1) // 2)
    transform = np.zeros((d, d, sym_d))
    idx = _tensor2array_index(d)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for i in range(d):
        for j in range(d):
            transform[i, j, idx[i, j]] = 1.0 if i == j else inv_sqrt2
    return transform


def tensor2array_mapping(d: int) -> np.ndarray:
    r"""
    Kelvin--Mandel mapping from a symmetric tensor to its Kelvin--Mandel vector representation.

    This is the inverse mapping of `array2tensor_mapping` (for symmetric tensors):
    diagonal entries are unchanged and off-diagonal entries are scaled by $\sqrt{2}$.

    :param d: the spatial dimension
    :return: transform tensor of shape (d, d, d_sym) such that eps_vec = einsum(transform, eps)
    """

    sym_d = int(d * (d + 1) // 2)
    transform = np.zeros((d, d, sym_d))
    idx = _tensor2array_index(d)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for i in range(d):
        for j in range(d):
            # For off-diagonal terms we use inv_sqrt2 in both (i,j) and (j,i)
            # so that summing over i,j yields sqrt(2)*eps_ij for symmetric tensors.
            transform[i, j, idx[i, j]] = 1.0 if i == j else inv_sqrt2
    return transform


def array2tensor(array: np.ndarray, map: np.ndarray, d: int) -> np.ndarray:
    """Map a Kelvin--Mandel (KM) vector to a symmetric tensor.

    Parameters
    ----------
    array:
        1D array of length `d_sym` representing a symmetric tensor in **Kelvin--Mandel Voigt** form
        (same ordering as `_tensor2array_index`, with shear scaled by $\sqrt{2}$).
    map:
        The transform returned by `array2tensor_mapping(d)`.
    d:
        Spatial dimension.

    Returns
    -------
    np.ndarray
        Symmetric tensor of shape `(d, d)` in physical tensor components.
    """

    out = np.einsum("ijk,k->ij", map, array) if d > 1 else np.expand_dims(array, axis=0)
    return out


def _variational_forms(
    mesh: dl.mesh,
    sigma_E: ufl.tensors,
    sigma_nu: ufl.tensors,
    test: dl.function.argument.Argument,
    symmetric_test: bool = False,
) -> tuple[ufl.Form, ufl.Form]:
    """
    A method for producing variational forms that are repeated used in the viscoelastic cell problem.
    :param sigma_E: an ufl tensor for the elastic stress
    :param sigma_nu: an ufl tensor for the viscous stress
    :param test: a test function for stress assembly.
    :param symmetric_test: a flag for using symmetric part of the test functions.
    :return: A tuple consists of variational forms
    """

    dx = dl.Measure("dx", mesh)
    grad_form = dl.sym(dl.grad(test)) if symmetric_test else dl.grad(test)
    varf_E = dl.inner(sigma_E, grad_form) * dx
    varf_nu = dl.inner(sigma_nu, grad_form) * dx
    return varf_E, varf_nu


class KelvinVoigtCellProblem:
    r"""
    Solver for Kelvin Voigt viscoelasticity cell problem.

    Given elastic (E) and viscous (nu) microstructure properties and imposed averaged strain and strain-rate
    trajectories, solve the periodic cell problem and compute the averaged stress trajectory.

        Notes
        -----
        - Strain/rate inputs and stress outputs are **Kelvin--Mandel Voigt** vectors of length `d_sym`.
            Diagonal components are unchanged and off-diagonals are scaled by $\sqrt{2}$.
        - All linear solves are posed on a mixed space `(u, lambda)` where `lambda` enforces a mean-zero
            constraint (via a Lagrange multiplier). This constraint is present in every solve.
        - This class does not support any legacy "solve viscous stress" mode; only the averaged stress
            trajectory in KM coordinates is exposed.
        """

    def __init__(self, n_cells=100, dim=2, FE_order=1, direct_solver=True) -> None:
        self._p, self._n_cells, self._dim, self._direct_solver = FE_order, n_cells, dim, direct_solver
        self._mesh, pbc = create_mesh(dim, n_cells)
        self._create_spaces(pbc)
        self._lift = LiftingFunction(self._Vh, self.dim)
        (
            self._A_E,
            self._A_nu,
            self._E,
            self._nu,
            self._E_bar,
            self._nu_bar,
            self._f_E,
            self._f_nu,
            self._rhs_mean,
            self._solver,
        ) = [None] * 10
        self._projector = Projector(self._Ph, self._Vh)
        self.parameters = {}
        self.parameters["verbose"] = False
        self.parameters["nt"] = 1000
        self.parameters["T"] = 1.0
        self.parameters["maximum_iterations"] = 200
        self.parameters["relative_tolerance"] = 1e-10

    def _create_spaces(self, pbc: object) -> None:
        """
        :param pbc: a periodic boundary with methods `inside()` and `map()`.
        """
        VCGp = dl.VectorElement("Lagrange", self._mesh.ufl_cell(), self.FE_order)
        CGp = dl.FiniteElement("Lagrange", self._mesh.ufl_cell(), self.FE_order)
        R = dl.VectorElement("R", self._mesh.ufl_cell(), 0)
        ME = dl.MixedElement([VCGp, R])

        self._Mh = dl.FunctionSpace(self._mesh, ME, constrained_domain=pbc)
        self._Vh = dl.FunctionSpace(self._mesh, VCGp)
        self._Sh = dl.FunctionSpace(self._mesh, CGp)
        self._Ph = dl.FunctionSpace(self._mesh, VCGp, constrained_domain=pbc)
        self._SPh = dl.FunctionSpace(self._mesh, CGp, constrained_domain=pbc)

        self._help_rhs = self.generate_vector(type="mixed")
        self._help_mixed = [self.generate_vector(type="mixed") for _ in range(2)]
        self._help_periodic = self.generate_vector(type="periodic")
        self._help_free = self.generate_vector(type="free")

    def FunctionSpace(self, type: str = None) -> dl.FunctionSpace:
        """
        :param type: type of function space to return
        :return: requested function space
        """
        if type is None or type == "free":
            return self._Vh
        elif type == "scalar":
            return self._Sh
        elif type == "scalar_periodic":
            return self._SPh
        elif type == "periodic":
            return self._Ph
        elif type == "mixed":
            return self._Mh
        else:
            raise Exception(type_error_message())

    def generate_vector(self, type: str = None) -> dl.Vector:
        """
        :param type: the type of dolfin vector to return
        :return: requested dolfin vector
        """
        if type is None or type == "free":
            return dl.Function(self._Vh).vector()
        elif type == "scalar":
            return dl.Function(self._Sh).vector()
        elif type == "periodic":
            return dl.Function(self._Ph).vector()
        elif type == "scalar_periodic":
            return dl.Function(self._SPh).vector()
        elif type == "mixed":
            return dl.Function(self._Mh).vector()
        else:
            raise Exception(type_error_message())

    def _assemble_system(
        self, E: Any, nu: Any
    ) -> tuple[
        dl.Matrix,
        dl.Matrix,
        dl.Matrix,
        list[dl.Vector],
        list[list[dl.Vector]],
        list[list[dl.Vector]],
    ]:
        r"""Assemble the matrices/vectors reused across solves.

        Parameters
        ----------
        E, nu:
            4th-order (UFL) elasticity/viscosity tensors for the microstructure.

        Returns
        -------
        A_lhs:
            Mixed-space constrained solve operator (viscous term + mean-zero constraint blocks).
        A_E:
            Mixed-space elastic operator used in the homogeneous evolution and RHS assembly.
        A_nu:
            Mixed-space viscous bilinear-form operator (displacement block only). Cached for evaluating
            $(\cdot,\cdot)_\nu$ inner products in `extract_memory_form()`.
        rhs_mean, f_E, f_nu:
            Preassembled vectors used to build RHS terms.
        """
        dx = dl.Measure("dx", self._mesh)
        A_lhs = dl.PETScMatrix(self._Mh.mesh().mpi_comm())
        A_E = dl.PETScMatrix(self._Mh.mesh().mpi_comm())
        A_nu = dl.PETScMatrix(self._Mh.mesh().mpi_comm())

        (u_trial, lam_trial) = dl.split(dl.TrialFunction(self._Mh))
        (u_test, lam_test) = dl.split(dl.TestFunction(self._Mh))

        eps_trial = dl.sym(dl.grad(u_trial))
        i, j, k, l = ufl.indices(4)
        sigma_E = ufl.as_tensor(E[i, j, k, l] * eps_trial[k, l], (i, j))
        sigma_nu = ufl.as_tensor(nu[i, j, k, l] * eps_trial[k, l], (i, j))

        varf_E, varf_nu = _variational_forms(self._mesh, sigma_E, sigma_nu, u_test)
        varf_lhs = varf_nu + dl.inner(lam_trial, u_test) * dx + dl.inner(u_trial, lam_test) * dx

        dl.assemble(varf_lhs, tensor=A_lhs)
        dl.assemble(varf_E, tensor=A_E)
        dl.assemble(varf_nu, tensor=A_nu)

        rhs_mean = [None for _ in range(self.dim)]
        f_E = [[None for _ in range(self.dim)] for _ in range(self.dim)]
        f_nu = [[None for _ in range(self.dim)] for _ in range(self.dim)]

        for ii in range(self.dim):
            one_half = [0.0 for _ in range(self.dim)]
            one_half[ii] = 0.5
            rhs_mean[ii] = dl.assemble(dl.inner(-dl.Constant(one_half), lam_test) * dx)

            for jj in range(self.dim):
                sigma_E = ufl.as_tensor(E[ii, jj, i, j], (i, j))
                sigma_nu = ufl.as_tensor(nu[ii, jj, i, j], (i, j))
                varf_E, varf_nu = _variational_forms(
                    self._mesh, sigma_E, sigma_nu, u_test, symmetric_test=True
                )
                f_E[ii][jj] = dl.assemble(varf_E)
                f_nu[ii][jj] = dl.assemble(varf_nu)

        return A_lhs, A_E, A_nu, rhs_mean, f_E, f_nu

    def _assemble_stress_map(self, E: ufl.tensors, nu: ufl.tensors) -> tuple[np.ndarray, np.ndarray]:
        """
        :param E: an ufl tensor
        :param nu: an ufl tensor
        :return: the spatial mean of E and nu components
        """
        dx = dl.Measure("dx", self._mesh)
        E_bar = np.zeros((self.dim, self.dim, self.dim, self.dim))
        nu_bar = np.zeros((self.dim, self.dim, self.dim, self.dim))
        dl.split(dl.TestFunction(self._Mh))

        for ii in range(self.dim):
            for jj in range(self.dim):
                for kk in range(self.dim):
                    for ll in range(self.dim):
                        E_bar[ii, jj, kk, ll] = dl.assemble(E[ii, jj, kk, ll] * dx)
                        nu_bar[ii, jj, kk, ll] = dl.assemble(nu[ii, jj, kk, ll] * dx)
        return E_bar, nu_bar

    def _assemble_rhs(self, z, strain, rate):
        """
        This function assemble the rhs matrix for RK4 scheme.
        :param z: the current state vector
        :param strain: the 3-point stencil values of strain tensor
        :param rate: the 3-point stencil values of strain rate tensor
        """
        self._help_rhs.zero()
        self._help_mixed[1].zero()

        for ii in range(self.dim):
            for jj in range(self.dim):
                self._help_rhs.axpy(strain[0, ii, jj] - strain[-1, ii, jj], self._rhs_mean[ii])

        for step in range(4):
            self._help_mixed[0].zero()
            self._help_mixed[0].axpy(-1, self._A_E * z)
            self._help_mixed[0].axpy(-coeff_s[step] * self.dt, self._A_E * self._help_mixed[1])
            for ii in range(self.dim):
                for jj in range(self.dim):
                    self._help_mixed[0].axpy(-strain[index_s[step], ii, jj], self._f_E[ii][jj])
                    self._help_mixed[0].axpy(-rate[index_s[step], ii, jj], self._f_nu[ii][jj])
            self._help_rhs.axpy(coeff_k[step] * self.dt, self._help_mixed[0])
            self._help_mixed[1].zero()
            self._help_mixed[1].axpy(1.0, self._help_mixed[0])

    def set_microstructure(self, E: Any, nu: Any) -> None:
        """
        :param E: the microstructure elastic property
        :param nu: the microstructure viscous property
        The microstructures must be given as one the following type:
        (1) dolfin vectors; (2) matrices; (3) functions; (4) an ufl matrix
        """
        microstructure = [None] * 2
        error_message = (
            "The size of the matrix does not match the scalar free function space nor the scalar periodic"
            "function space"
        )

        comm = self._Mh.mesh().mpi_comm()
        comm_size = comm.size

        for count, arg in enumerate([E, nu]):
            if isinstance(arg, dl.cpp.la.PETScVector):
                if arg.size() == self._Sh.dim():
                    func = hp.vector2Function(arg, self._Sh)
                elif arg.size() == self._SPh.dim():
                    func = hp.vector2Function(arg, self._SPh)
                else:
                    raise Exception(error_message)
                microstructure[count] = func * identity_4th(self.dim)

            elif isinstance(arg, np.ndarray):
                if comm_size > 1:
                    raise RuntimeError(
                        "Passing microstructure as a numpy array is not MPI-safe in this code path. "
                        "Pass a dolfin Function/Constant, or a distributed PETScVector on the correct FunctionSpace."
                    )
                assert self.FE_order == 1
                if arg.size == self._Sh.dim():
                    func = MatrixInterpolation(arg, self._Sh)
                elif arg.size == self._SPh.dim():
                    func = MatrixInterpolation(arg, self._SPh)
                else:
                    raise Exception(error_message)
                microstructure[count] = func * identity_4th(self.dim)

            elif isinstance(arg, dl.function.function.Function) or isinstance(arg, dl.function.constant.Constant):
                microstructure[count] = arg * identity_4th(self.dim)

            else:
                microstructure[count] = arg

        time0 = None
        if self.parameters["verbose"]:
            print("Assemble required matrices and vectors....")
            time0 = time.time()

        A_lhs, self._A_E, self._A_nu, self._rhs_mean, self._f_E, self._f_nu = self._assemble_system(
            microstructure[0], microstructure[1]
        )
        self._E_bar, self._nu_bar = self._assemble_stress_map(microstructure[0], microstructure[1])

        if self.use_direct_solver:
            if comm_size > 1:
                self._solver = hp.PETScLUSolver(comm, method="mumps")
            else:
                self._solver = hp.PETScLUSolver(comm)
        else:
            self._solver = hp.PETScKrylovSolver(comm, "gmres", "jacobi")
            self._solver.parameters["maximum_iterations"] = self.parameters["maximum_iterations"]
            self._solver.parameters["relative_tolerance"] = self.parameters["relative_tolerance"]
            self._solver.parameters["error_on_nonconvergence"] = True
            self._solver.parameters["nonzero_initial_guess"] = False

        self._solver.set_operator(A_lhs)

        if self.parameters["verbose"]:
            time1 = time.time()
            print("Finished. Took %1.2fs" % (time1 - time0))

        self._E, self._nu = microstructure[0], microstructure[1]

    def _update_stress(
        self,
        stress: np.ndarray,
        step_count: int,
        sigma_E: np.ndarray,
        sigma_nu: np.ndarray,
        strain: np.ndarray,
        rate: np.ndarray,
    ) -> None:
        """
        :param stress: a numpy array for stress history.
        :param step_count: an integer for step count, with t=0 has count 0.
        :param sigma_E: a numpy ndarray for elastic stress component for the periodic solution
        :param sigma_nu: a numpy ndarray for viscous stress component for the periodic solution
        :param strain: a numpy ndarray for the symmetric tensor form of strain history (up to 5 steps)
        :param rate: a numpy ndarray for the symmetric tensor form of strain rate history (up to 5 steps)
        """
        if 6 <= step_count <= 8:
            coeff = coeff_6th_side[::-1, np.newaxis, np.newaxis]
            offset, current = 6, -1
            stress[step_count - offset] = (
                np.sum(coeff / self.dt * sigma_nu, axis=0)
                + sigma_E[current]
                + np.einsum("ijkl,kl->ij", self._E_bar, strain[current])
                + np.einsum("ijkl,kl->ij", self._nu_bar, rate[current])
            )

        if step_count >= self.nt - 2:
            coeff = -coeff_6th_side[:, np.newaxis, np.newaxis]
            offset, current = 0, 0
            stress[step_count - offset] = (
                np.sum(coeff / self.dt * sigma_nu, axis=0)
                + sigma_E[current]
                + np.einsum("ijkl,kl->ij", self._E_bar, strain[current])
                + np.einsum("ijkl,kl->ij", self._nu_bar, rate[current])
            )

        if step_count >= 6:
            coeff = coeff_6th_central[::-1, np.newaxis, np.newaxis]
            offset, current = 3, 3
            stress[step_count - offset] = (
                np.sum(coeff / self.dt * sigma_nu, axis=0)
                + sigma_E[current]
                + np.einsum("ijkl,kl->ij", self._E_bar, strain[current])
                + np.einsum("ijkl,kl->ij", self._nu_bar, rate[current])
            )

    def _assign_displacement(self, displacement: dl.Vector, periodic: dl.Vector, strain: np.ndarray) -> None:
        """
        Build the full displacement by adding periodic deviation and affine lifting.

        Parameters
        ----------
        displacement:
            Output displacement vector in the free space.
        periodic:
            Periodic deviation vector (periodic space).
        strain:
            Symmetric strain tensor (shape `(dim, dim)`) used to generate the affine lifting.
        """
        temp = self._projector.project(periodic)
        varepsilon = self._lift.generate(strain)
        displacement.zero()
        displacement.axpy(1.0, temp.vector())
        displacement.axpy(1.0, varepsilon)

    def extract_memory_form(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Extract the memory-form homogenized model in Kelvin--Mandel Voigt form.

        The returned objects satisfy the convolution constitutive law

          sigma(t) = E' e(t) + nu' e_dot(t) - \int_0^t K(t-s) e(s)\,ds,

        where e, e_dot, sigma are Kelvin--Mandel vectors of length `d_sym`.

                KM vs tensor checklist
                ---------------------
                This method intentionally switches between two representations:

                - Kelvin--Mandel (KM) vectors: length `d_sym`, shear scaled by $\sqrt{2}$.
                    This is the representation of the *returned* operators `nu_prime`, `E_prime`, and `kernel`.
                - Physical symmetric tensors: shape `(dim, dim)`, with shear entries as true tensor components.
                    This is the representation required by the PDE / variational forms.

                Concretely in the implementation:

                - `a2t_map = array2tensor_mapping(dim)` maps KM vectors -> tensor components.
                - `t2a_map = tensor2array_mapping(dim)` maps tensor components -> KM vectors.
                - `basis_tensors[m]` is the physical-tensor representative of the KM unit vector $e_m$.
                    For shear modes this has off-diagonals equal to $1/\sqrt{2}$.
                - The RHS assembly loops over `i,j` and uses `e_tensor[i,j]` in tensor form.
                - `nu_prime[:, m]` and `E_prime[:, m]` are produced by mapping the computed *tensor* averages
                    back into KM coordinates using `t2a_map`.
                - `kernel[n, row, col]` is computed as a $\nu$-inner product on the mixed space and therefore
                    corresponds to the KM-basis pairing $(e_{row}, e_{col})$.

        This implementation follows the two static unit-cell problems for q_e^(1), q_e^(2)
        and the homogeneous evolution problem for phi_e(t) (see attached equations).
        All linear solves reuse the factorization/preconditioner already set up in `set_microstructure`.

        :return: (nu_prime, E_prime, kernel) with shapes:
                 nu_prime: (d_sym, d_sym)
                 E_prime:  (d_sym, d_sym)
                 kernel:   (nt+1, d_sym, d_sym) where kernel[n] corresponds to K(t_n).
        """

        if self._solver is None or self._E is None or self._nu is None:
            raise RuntimeError("Call set_microstructure(E, nu) before extract_memory_form().")

        time1 = time.time() if self.parameters["verbose"] else None
        if self.parameters["verbose"]:
            print("Extracting (nu', E', K(t)) memory form...")

        dim = self.dim
        d_sym = self.d_sym
        dt = self.dt
        times = np.linspace(0.0, self.T, self.nt + 1)

        a2t_map = array2tensor_mapping(dim)
        t2a_map = tensor2array_mapping(dim)

        nu_prime = np.zeros((d_sym, d_sym))
        E_prime = np.zeros((d_sym, d_sym))
        kernel = np.zeros((self.nt + 1, d_sym, d_sym))

        # Precompute physical-tensor representatives of the Kelvin--Mandel basis vectors.
        # Concretely, `basis_tensors[m]` is the symmetric tensor you get by mapping the KM unit vector e_m
        # through `array2tensor_mapping(dim)`.
        basis_tensors = [a2t_map[:, :, m].copy() for m in range(d_sym)]

        # Solve the two static cell problems for each basis direction.
        z2_list: list[dl.Vector] = []
        rhs = self.generate_vector(type="mixed")
        sol = self.generate_vector(type="mixed")

        for m in range(d_sym):
            e_tensor = basis_tensors[m]

            # The static cell problem solution in the viscous limit
            rhs.zero()
            for i in range(dim):
                for j in range(dim):
                    rhs.axpy(float(e_tensor[i, j]), self._f_nu[i][j])
            sol.zero()
            self._solver.solve(sol, rhs)
            z1 = self.generate_vector(type="mixed")
            z1.zero()
            z1.axpy(1.0, sol)

            # The static cell problem solution with the elastic contribution
            rhs.zero()
            # + <E psi^(1), sym grad v>
            rhs.axpy(1.0, self._A_E * z1)
            # - <E e, sym grad v>
            for i in range(dim):
                for j in range(dim):
                    rhs.axpy(-float(e_tensor[i, j]), self._f_E[i][j])
            sol.zero()
            self._solver.solve(sol, rhs)
            z2 = self.generate_vector(type="mixed")
            z2.zero()
            z2.axpy(1.0, sol)
            z2_list.append(z2)

            # Compute nu' * e_m = \int nu:(e_m + eps(psi^(1)))
            avg_nu_psi1 = np.zeros((dim, dim))
            for i in range(dim):
                for j in range(dim):
                    avg_nu_psi1[i, j] = self._f_nu[i][j].inner(z1)
            nu_stress = np.einsum("ijkl,kl->ij", self._nu_bar, e_tensor) - avg_nu_psi1
            nu_prime[:, m] = np.einsum("ij,ijm->m", nu_stress, t2a_map)

            # Compute E' * e_m = \int E:(e_m + eps(psi^(1))) + \int nu:eps(psi^(2))
            avg_E_psi1 = np.zeros((dim, dim))
            avg_nu_psi2 = np.zeros((dim, dim))
            for i in range(dim):
                for j in range(dim):
                    avg_E_psi1[i, j] = self._f_E[i][j].inner(z1)
                    avg_nu_psi2[i, j] = self._f_nu[i][j].inner(z2)
            E_stress = np.einsum("ijkl,kl->ij", self._E_bar, e_tensor) - avg_E_psi1 + avg_nu_psi2
            E_prime[:, m] = np.einsum("ij,ijm->m", E_stress, t2a_map)

        # The viscous bilinear form matrix on the *mixed* space (displacement block only)
        # is assembled once in set_microstructure() and stored as self._A_nu.
        if self._A_nu is None:
            raise RuntimeError("Internal error: viscous matrix was not assembled. Call set_microstructure() first.")

        # Time evolution for phi_e(t): A_lhs z_dot + A_E z = 0, with z(0)=q^(2).
        z = self.generate_vector(type="mixed")
        k1 = self.generate_vector(type="mixed")
        k2 = self.generate_vector(type="mixed")
        k3 = self.generate_vector(type="mixed")
        k4 = self.generate_vector(type="mixed")
        tmp = self.generate_vector(type="mixed")
        rhs_h = self.generate_vector(type="mixed")

        nu_prod = self.generate_vector(type="mixed")

        for col in range(d_sym):
            z.zero()
            z.axpy(1.0, z2_list[col])

            for n, _t in enumerate(times):
                # Compute kernel[:, col] at t_n via the nu-inner product on the mixed space:
                #   K_{row,col}(t_n) = ( q^(2)_row, phi_col(t_n) )_nu
                # This avoids extracting periodic components and keeps mean-zero constraints in every solve.
                self._A_nu.mult(z, nu_prod)
                for row in range(d_sym):
                    kernel[n, row, col] = z2_list[row].inner(nu_prod)

                if n == self.nt:
                    break

                # RK4 step for z_dot = -A_lhs^{-1} (A_E z)
                rhs_h.zero()
                rhs_h.axpy(-1.0, self._A_E * z)
                k1.zero()
                self._solver.solve(k1, rhs_h)

                tmp.zero()
                tmp.axpy(1.0, z)
                tmp.axpy(0.5 * dt, k1)
                rhs_h.zero()
                rhs_h.axpy(-1.0, self._A_E * tmp)
                k2.zero()
                self._solver.solve(k2, rhs_h)

                tmp.zero()
                tmp.axpy(1.0, z)
                tmp.axpy(0.5 * dt, k2)
                rhs_h.zero()
                rhs_h.axpy(-1.0, self._A_E * tmp)
                k3.zero()
                self._solver.solve(k3, rhs_h)

                tmp.zero()
                tmp.axpy(1.0, z)
                tmp.axpy(dt, k3)
                rhs_h.zero()
                rhs_h.axpy(-1.0, self._A_E * tmp)
                k4.zero()
                self._solver.solve(k4, rhs_h)

                z.axpy(dt / 6.0, k1)
                z.axpy(dt / 3.0, k2)
                z.axpy(dt / 3.0, k3)
                z.axpy(dt / 6.0, k4)

        if self.parameters["verbose"]:
            time2 = time.time()
            print("Finished memory form extraction. Took %1.2fs" % (time2 - time1))

        return nu_prime, E_prime, kernel

    def solve(
        self,
        strain_func: Any,
        rate_func: Any,
        return_displacement: bool = False,
        return_periodic: bool = False,
        file: dl.File = None,
        snapshots: list[int] | None = None,
    ):
        """Solve the time-dependent constrained cell problem.

        The inputs/outputs are Kelvin--Mandel Voigt vectors of length `d_sym`.

                KM vs tensor checklist
                ---------------------
                - `strain_func(t)` and `rate_func(t)` return KM vectors (length `d_sym`).
                - Internally these are mapped to physical symmetric tensors via `array2tensor(...)` before
                    assembling right-hand sides for the PDE.
                - Internal stress accumulation (`stress` array) is stored in tensor form with shape `(dim, dim)`.
                - The returned `stress_km` is obtained by mapping the tensor stresses back to KM vectors via
                    `tensor2array_mapping(dim)`.

        :param strain_func: callable returning a 1D numpy array of length `d_sym` for t in [0, T]
        :param rate_func: callable returning a 1D numpy array of length `d_sym` for t in [0, T]
        :param return_displacement: A flag for returning displacement as a list of dolfin vectors. Default to `False`
        :param return_periodic: A flag for returning periodic part of the displacement as a list of dolfin vectors. Default to `False`.
        :param file: the pvd file for saving the solution state
        :param snapshots: the numpy array for the time slices requested for return or saving the state evolution.
        Returns
        -------
        Depending on flags, this returns:

        - `(times, stress_km)`
        - `(times, stress_km, displacement_list)`
        - `(times, stress_km, periodic_list)`
        - `(times, stress_km, displacement_list, periodic_list)`

        where:
        - `times` has shape `(nt+1,)`.
        - `stress_km` has shape `(nt+1, d_sym)` and is in Kelvin--Mandel Voigt form.
        """
        clock_ebar, clock_solve, clock_stress = None, None, None
        if return_displacement and self.parameters["verbose"]:
            print("Computing displacement is done via L2 project and it can be slow.")

        solve_time, stress_time, ebar_time, time0 = 0, 0, 0, 0
        displacement_time = 0 if return_displacement else None
        if self.parameters["verbose"]:
            print("Solving the cell problem....")
            time0 = time.time()

        a2t_map = array2tensor_mapping(self.dim)
        t2a_map = tensor2array_mapping(self.dim)

        times = np.linspace(0, self.T, self.nt + 1)
        stress = np.zeros((self.nt + 1, self.dim, self.dim))

        strain_hist = np.zeros((7, self.dim, self.dim))
        rate_hist = np.zeros((7, self.dim, self.dim))

        strain_stencil = np.zeros((3, self.dim, self.dim))
        rate_stencil = np.zeros((3, self.dim, self.dim))

        z = self.generate_vector(type="mixed")
        sigma_E_hist = np.zeros((7, self.dim, self.dim))
        sigma_nu_hist = np.zeros((7, self.dim, self.dim))

        strain = strain_func(0.0)
        assert strain.size == self.d_sym
        strain_hist[1] = array2tensor(strain, a2t_map, self.dim)
        rate = rate_func(0.0)
        assert rate.size == self.d_sym
        rate_hist[1] = array2tensor(rate, a2t_map, self.dim)

        u_list, u_p_list = [], []

        if snapshots is None or np.isin(0, snapshots).all():
            if file is None:
                if return_displacement:
                    u_list.append(self.generate_vector("free"))
                if return_periodic:
                    u_p_list.append(self.generate_vector("periodic"))
            else:
                if return_displacement:
                    file.write(dl.Function(self._Vh), 0.0)
                if return_periodic:
                    file.write(dl.Function(self._Ph), 0.0)

        for step_count, t in enumerate(times[1:]):
            clock_new_iter = time.time() if self.parameters["verbose"] else None

            strain_hist[0] = array2tensor(strain_func(t), a2t_map, self.dim)
            rate_hist[0] = array2tensor(rate_func(t), a2t_map, self.dim)

            strain_stencil[0] = strain_hist[0]
            rate_stencil[0] = rate_hist[0]
            strain_stencil[1] = array2tensor(strain_func(t - 0.5 * self.dt), a2t_map, self.dim)
            rate_stencil[1] = array2tensor(rate_func(t - 0.5 * self.dt), a2t_map, self.dim)
            strain_stencil[2] = strain_hist[1]
            rate_stencil[2] = rate_hist[1]

            if self.parameters["verbose"]:
                clock_ebar = time.time()
                ebar_time += clock_ebar - clock_new_iter

            self._assemble_rhs(z, strain_stencil, rate_stencil)
            self._solver.solve(self._help_mixed[0], self._help_rhs)
            z.axpy(1.0, self._help_mixed[0])

            if self.parameters["verbose"]:
                clock_solve = time.time()
                solve_time += clock_solve - clock_ebar

            for ii in range(self.dim):
                for jj in range(self.dim):
                    sigma_E_hist[0, ii, jj] = self._f_E[ii][jj].inner(z)
                    sigma_nu_hist[0, ii, jj] = self._f_nu[ii][jj].inner(z)
            self._update_stress(stress, step_count + 1, sigma_E_hist, sigma_nu_hist, strain_hist, rate_hist)

            if self.parameters["verbose"]:
                clock_stress = time.time()
                stress_time += clock_stress - clock_solve

            if snapshots is None or np.isin(step_count + 1, snapshots).all():
                if return_displacement:
                    extract_component(self._help_periodic, z, self._Ph, self._Mh, component=0)
                    if file is None:
                        u_list.append(self.generate_vector("free"))
                        self._assign_displacement(u_list[-1], self._help_periodic, strain_hist[0])
                    else:
                        self._assign_displacement(self._help_free, self._help_periodic, strain_hist[0])
                        file.write(hp.vector2Function(self._help_free, self._Vh), t)
                    if self.parameters["verbose"]:
                        clock_displacement = time.time()
                        displacement_time += clock_displacement - clock_stress

                if return_periodic:
                    if file is None:
                        u_p_list.append(self.generate_vector("periodic"))
                        extract_component(u_p_list[-1], z, self._Ph, self._Mh, component=0)
                    else:
                        extract_component(self._help_periodic, z, self._Ph, self._Mh, component=0)
                        file.write(hp.vector2Function(self._help_periodic, self._Ph), t)

            strain_hist[1:] = strain_hist[:-1]
            rate_hist[1:] = rate_hist[:-1]
            sigma_E_hist[1:] = sigma_E_hist[:-1]
            sigma_nu_hist[1:] = sigma_nu_hist[:-1]

        if file is not None:
            file.close()

        out = np.einsum("ijk, jkl -> il", stress, t2a_map)

        if self.parameters["verbose"]:
            time1 = time.time()
            print_compute_time(
                time1 - time0,
                [solve_time, stress_time, ebar_time, displacement_time],
                label=["Solve", "Stress estimation", "Ebar and rate evaluation", "Solution to displacement mapping"],
            )

        if return_displacement and return_periodic and file is None:
            return times, out, u_list, u_p_list
        elif return_displacement and file is None:
            return times, out, u_list
        elif return_periodic and file is None:
            return times, out, u_p_list
        else:
            return times, out

    @property
    def E(self):
        return self._E

    @property
    def nu(self):
        return self._nu

    @property
    def T(self):
        return self.parameters["T"]

    @property
    def nt(self):
        return self.parameters["nt"]

    @property
    def dt(self):
        return self.parameters["T"] / self.parameters["nt"]

    @property
    def dim(self):
        return self._dim

    @property
    def d_sym(self):
        return self.dim * (self.dim + 1) // 2

    @property
    def n_cells(self):
        return self._n_cells

    @property
    def FE_order(self):
        return self._p

    @property
    def use_direct_solver(self):
        return self._direct_solver
