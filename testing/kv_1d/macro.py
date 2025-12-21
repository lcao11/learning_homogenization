import dolfin as dl
import numpy as np
import torch
import sys
from pathlib import Path

# Ensure project root (learning_homogenization/) is on sys.path when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modeling import MatrixInterpolation
import scipy.sparse as ss
import time
from datetime import timedelta

def PETScMatrix_to_torch(M: dl.PETScMatrix, device="cpu", dtype=torch.float32) -> torch.sparse:
    mat = dl.as_backend_type(M).mat()
    mat = ss.csr_matrix(mat.getValuesCSR()[::-1], shape=mat.size)
    mat = mat.tocoo()
    values = mat.data
    indices = np.vstack((mat.row, mat.col))

    i = torch.tensor(indices, dtype=torch.long, device=device)
    v = torch.tensor(values, dtype=dtype, device=device)
    shape = mat.shape

    return torch.sparse_coo_tensor(i, v, torch.Size(shape))

def boundary(x):
    return x[0] < dl.DOLFIN_EPS or x[0] > 1.0 - dl.DOLFIN_EPS

class MacroViscoelasticSolver1D():
    def __init__(self, n_periods=100, cells_per_periods=100) -> None:
        self._cells_per_periods, self._n_periods = cells_per_periods, n_periods
        self.parameters = {}
        self.parameters["verbose"] = False
        self.parameters["nt"] = 1000
        self.parameters["T"] = 1.0
        mesh = dl.UnitIntervalMesh(self._n_periods * self._cells_per_periods)
        self._Vh = dl.FunctionSpace(mesh, "CG", 1)
        self._bc0 = dl.DirichletBC(self._Vh, dl.Constant(0.0), boundary)
        (self._A_nu, self._A_E, self._A_f, self._E_epsilon, self._nu_epsilon, self._solver) = [None] * 6
    
    def _assemble_system(self, E_epsilon, nu_epsilon):
        u_trial = dl.TrialFunction(self._Vh)
        u_test = dl.TestFunction(self._Vh)
        # The LHS matrix of the variational form, RK4
        varf_nu = dl.inner(nu_epsilon*dl.grad(u_trial), dl.grad(u_test))*dl.dx
        varf_E = dl.inner(E_epsilon*dl.grad(u_trial), dl.grad(u_test))*dl.dx
        A_nu, A_E, A_f = dl.PETScMatrix(self._Vh.mesh().mpi_comm()), dl.PETScMatrix(self._Vh.mesh().mpi_comm()), dl.PETScMatrix(self._Vh.mesh().mpi_comm())
        tmp = dl.Function(self._Vh).vector()
        dl.assemble_system(varf_nu, u_test*dl.dx, self._bc0, A_tensor = A_nu, b_tensor = tmp)  # The matrix A_lhs
        dl.assemble_system(varf_E, u_test*dl.dx, self._bc0, A_tensor = A_E, b_tensor = tmp)  # The matrix A_E
        dl.assemble_system(u_trial*u_test*dl.dx, u_test*dl.dx, self._bc0, A_tensor = A_f, b_tensor = tmp)  # The matrix A_E
        return A_nu, A_E, A_f

    def set_microstructure(self, E: np.ndarray, nu: np.ndarray) -> None:
        E_epsilon_array =  np.zeros(self._Vh.dim())
        nu_epsilon_array = np.zeros(self._Vh.dim())
        for ii in range(self._n_periods):
            left = ii*(self._cells_per_periods)
            right = left + self._cells_per_periods + 1
            E_epsilon_array[left:right] = E
            nu_epsilon_array[left:right] = nu
        E_epsilon = MatrixInterpolation(E_epsilon_array, self._Vh)
        nu_epsilon = MatrixInterpolation(nu_epsilon_array, self._Vh)
        self._A_nu, self._A_E, self._A_f = self._assemble_system(E_epsilon, nu_epsilon)
        self._solver = dl.PETScLUSolver(self._Vh.mesh().mpi_comm(), self._A_nu, method="mumps")

    def solve(self, forcing_expr):
        u_list = [dl.Function(self._Vh) for _ in range(self.parameters["nt"]+1)]
        rhs = dl.Function(self._Vh).vector()
        dt = self.parameters["T"]/self.parameters["nt"]
        times = np.linspace(0, self.parameters["T"], self.parameters["nt"]+1)
        for ii in range(self.parameters["nt"]):
            forcing_expr.t = times[ii+1]
            f = dl.interpolate(forcing_expr, self._Vh)
            rhs.zero()
            rhs.axpy(dt, self._A_f*f.vector())
            rhs.axpy(-dt, self._A_E*u_list[ii].vector())
            rhs.axpy(1.0, self._A_nu*u_list[ii].vector())
            self._bc0.apply(rhs)
            self._solver.solve(u_list[ii+1].vector(), rhs)
        return u_list
        
        

class HomogenizedViscoelasticSolver1D():
    def __init__(self, n_cells=100, FE_order=1) -> None:
        self._p, self._n_cells = FE_order, n_cells
        self.parameters = {}
        self.parameters["verbose"] = False
        self.parameters["nt"] = 1000
        self.parameters["T"] = 1.0
        self.parameters["maximum_iterations"] = 200
        self.parameters["relative_tolerance"] = 1e-10
        mesh = dl.UnitIntervalMesh(self._n_cells)
        self._Vh = dl.FunctionSpace(mesh, "CG", 1)
        self._bc0 = dl.DirichletBC(self._Vh, dl.Constant(0.0), boundary)
        u_trial = dl.TrialFunction(self._Vh)
        u_test = dl.TestFunction(self._Vh)
        varf_h1semi = dl.inner(dl.grad(u_trial), dl.grad(u_test))*dl.dx
        varf_l2 = dl.inner(u_trial, u_test)*dl.dx
        self._A_h1semi, self._A_l2 = dl.PETScMatrix(self._Vh.mesh().mpi_comm()), dl.PETScMatrix(self._Vh.mesh().mpi_comm())
        tmp = dl.Function(self._Vh).vector()
        dl.assemble_system(varf_h1semi, u_test*dl.dx, self._bc0, A_tensor=self._A_h1semi, b_tensor=tmp)
        dl.assemble_system(varf_l2, u_test*dl.dx, self._bc0, A_tensor=self._A_l2, b_tensor=tmp)

    def set_microstructure(self, E_prime, nu_prime, memory_kernel):
        self._E_prime = E_prime
        self._nu_prime = nu_prime
        self._memory_kernel = memory_kernel
        self._solver = dl.PETScLUSolver(self._Vh.mesh().mpi_comm(), self._nu_prime*self._A_h1semi, method="mumps")
    
    def solve(self, forcing_expr):
        assert self._memory_kernel.shape[0] == self.parameters["nt"]+1
        u_list = [dl.Function(self._Vh) for _ in range(self.parameters["nt"]+1)]
        rhs = dl.Function(self._Vh).vector()
        dt = self.parameters["T"]/self.parameters["nt"]
        times = np.linspace(0, self.parameters["T"], self.parameters["nt"]+1)
        for ii in range(self.parameters["nt"]):
            forcing_expr.t = times[ii+1]
            f = dl.interpolate(forcing_expr, self._Vh)
            rhs.zero()
            rhs.axpy(dt, self._A_l2*f.vector())
            rhs.axpy(-self._E_prime*dt, self._A_h1semi*u_list[ii].vector())
            rhs.axpy(self._nu_prime, self._A_h1semi*u_list[ii].vector())
            for jj in range(ii):
                rhs.axpy(self._memory_kernel[jj]*dt**2, self._A_h1semi*u_list[ii-jj].vector())
            rhs.axpy(-0.5*self._memory_kernel[0]*dt**2, self._A_h1semi*u_list[ii].vector())
            rhs.axpy(-0.5*self._memory_kernel[ii]*dt**2, self._A_h1semi*u_list[0].vector())
            self._bc0.apply(rhs)
            self._solver.solve(u_list[ii+1].vector(), rhs)
        return u_list

class RNOViscoelasticSolver1D():
    def __init__(self, n_cells=100, FE_order=1, device="cuda") -> None:
        self._p, self._n_cells = FE_order, n_cells
        self.parameters = {}
        self.parameters["verbose"] = False
        self.parameters["nt"] = 1000
        self.parameters["T"] = 1.0
        self.parameters["maximum_iterations"] = 200
        self.parameters["relative_tolerance"] = 1e-10

        mesh = dl.UnitIntervalMesh(self._n_cells)
        self._Vh = dl.FunctionSpace(mesh, "CG", 1)
        self._Gh = dl.VectorFunctionSpace(mesh, "DG", 0)
        self._bc0 = dl.DirichletBC(self._Vh, dl.Constant(0.0), boundary)
        u_trial, u_test = dl.TrialFunction(self._Vh), dl.TestFunction(self._Vh)
        s_trial, s_test = dl.TrialFunction(self._Gh), dl.TestFunction(self._Gh)
        A_sigma, _ = dl.assemble_system(dl.dot(s_trial, dl.grad(u_test))*dl.dx, u_test*dl.dx, self._bc0)
        self._A_f, _ = dl.assemble_system(dl.inner(u_trial, u_test)*dl.dx, u_test*dl.dx, self._bc0)
        A_norm, _ = dl.assemble_system(dl.inner(u_trial, u_test)*dl.dx + dl.inner(dl.grad(u_trial), dl.grad(u_test))*dl.dx, u_test*dl.dx, self._bc0)
        self._A_sigma = PETScMatrix_to_torch(A_sigma, device=device, dtype=torch.float32)
        self._device = device

        # Torch sparse version of the L2 mass matrix (used to build RHS on GPU).
        self._A_f_torch = PETScMatrix_to_torch(self._A_f, device=self._device, dtype=torch.float32).coalesce()

        # Analytic CG1 -> DG0 gradient operator for a uniform 1D UnitIntervalMesh.
        # Maps nodal values u (n_nodes = n_cells+1) to cellwise gradients du (n_cells).
        n_cells = self._n_cells
        n_nodes = self._Vh.dim()
        n_cells_dg = self._Gh.dim()
        if n_nodes != n_cells + 1 or n_cells_dg != n_cells:
            raise ValueError(
                f"Unexpected 1D dimensions: Vh.dim={n_nodes}, Gh.dim={n_cells_dg}, n_cells={n_cells}"
            )
        h = 1.0 / float(n_cells)
        rows = torch.arange(n_cells, device=self._device, dtype=torch.long)
        cols_left = rows
        cols_right = rows + 1
        grad_indices = torch.stack(
            [torch.cat([rows, rows]), torch.cat([cols_left, cols_right])], dim=0
        )
        grad_values = torch.cat(
            [
                torch.full((n_cells,), -1.0 / h, device=self._device, dtype=torch.float32),
                torch.full((n_cells,), 1.0 / h, device=self._device, dtype=torch.float32),
            ]
        )
        self._grad_tensor = torch.sparse_coo_tensor(
            grad_indices, grad_values, (n_cells, n_nodes)
        ).coalesce()

        # Residual weighting: avoid forming inv(A_norm). Use Cholesky factor for fast solves.
        A_norm_t = PETScMatrix_to_torch(A_norm, device=self._device, dtype=torch.float32)
        with torch.no_grad():
            self._A_norm_chol = torch.linalg.cholesky(A_norm_t.to_dense())

        # Cache constant tensors used each timestep.
        self._zero_pad = torch.zeros(1, 1, device=self._device, dtype=torch.float32)
        mask = torch.ones(n_nodes, 1, device=self._device, dtype=torch.float32)
        mask[0, 0] = 0.0
        mask[-1, 0] = 0.0
        self._mask = mask
    
    def set_microstructure(self, E, nu):
        # Shape convention matches learning.response: batch dimension = self._Gh.dim().
        E_base = torch.as_tensor(E, device=self._device, dtype=torch.float32).unsqueeze(0)
        nu_base = torch.as_tensor(nu, device=self._device, dtype=torch.float32).unsqueeze(0)
        microstructure = torch.empty(self._Gh.dim(), 2, E_base.shape[1], device=self._device, dtype=torch.float32)
        microstructure[:, 0, :] = E_base.expand(self._Gh.dim(), -1)
        microstructure[:, 1, :] = nu_base.expand(self._Gh.dim(), -1)
        self._microstructure = microstructure
    
    def set_rno(self, F, G, n_internal = 5):
        F.eval()
        G.eval()
        self._F = F.to(device=self._device)
        self._G = G.to(device=self._device)
        self._n_internal = n_internal
    
    def solve(self, forcing_expr, return_internal=False, precompute_rhs: bool = True):
        u_list = [dl.Function(self._Vh) for _ in range(self.parameters["nt"] + 1)]
        dt = self.parameters["T"] / self.parameters["nt"]
        times = np.linspace(0, self.parameters["T"], self.parameters["nt"] + 1)
        if return_internal:
            xi_list = np.zeros((self.parameters["nt"] + 1, self._Gh.dim(), self._n_internal))
        xi = torch.zeros(self._Gh.dim(), self._n_internal).to(dtype=torch.float32, device=self._device)
        du_0 = torch.zeros(self._Gh.dim(), 1).to(dtype=torch.float32, device=self._device)
        zero_pad = self._zero_pad
        mask = self._mask
        forcing_function = dl.Function(self._Vh)

        rhs_all = None
        if precompute_rhs:
            # Precompute nodal RHS vectors for all timesteps on CPU, then transfer once to device.
            # This avoids per-step CPU->GPU transfers of rhs.
            rhs_np = np.zeros((self.parameters["nt"], self._Vh.dim()), dtype=np.float32)
            rhs_vec = dl.Function(self._Vh).vector()
            for ii in range(self.parameters["nt"]):
                forcing_expr.t = times[ii + 1]
                forcing_function.assign(dl.interpolate(forcing_expr, self._Vh))
                rhs_vec.zero()
                rhs_vec.axpy(1.0, self._A_f * forcing_function.vector())
                rhs_np[ii, :] = rhs_vec.get_local().astype(np.float32, copy=False)
            rhs_all = torch.from_numpy(rhs_np).to(device=self._device).unsqueeze(-1)

        # Keep interior DOFs as a persistent torch tensor across all timesteps.
        # This avoids FEniCS->NumPy->Torch copies for the initial guess each step.
        u_var = torch.zeros(self._Vh.dim() - 2, 1, device=self._device, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.LBFGS([u_var], line_search_fn="strong_wolfe", max_iter=1000)

        def force(u_var_local, xi_local, du0_local):
            u = torch.cat([zero_pad, u_var_local, zero_pad], dim=0)
            du = torch.sparse.mm(self._grad_tensor, u)
            sigma = self._F(self._microstructure, du, (du - du0_local) / dt, xi_local)
            return torch.sparse.mm(self._A_sigma, sigma), du

        t0 = time.time()
        for ii in range(self.parameters["nt"]):
            if rhs_all is not None:
                rhs = rhs_all[ii]
            else:
                forcing_expr.t = times[ii + 1]
                forcing_function.assign(dl.interpolate(forcing_expr, self._Vh))
                f_vec = torch.as_tensor(
                    forcing_function.vector().get_local(), device=self._device, dtype=torch.float32
                ).unsqueeze(1)
                rhs = torch.sparse.mm(self._A_f_torch, f_vec)

            with torch.no_grad():
                xi = self._G(self._microstructure, du_0, xi)*dt + xi
            if return_internal:
                xi_list[ii+1] = xi.detach().cpu().numpy()

            def residual_norm(force_vec, rhs_vec):
                r = force_vec - rhs_vec
                # y = A_norm^{-1} r via Cholesky solve
                y = torch.cholesky_solve(r, self._A_norm_chol)
                return torch.einsum("ij, ij", r * mask, y)

            def closure():
                optimizer.zero_grad()
                out, _ = force(u_var, xi, du_0)
                objective = residual_norm(out, rhs)
                objective.backward()
                return objective
            
            optimizer.step(closure)
            with torch.no_grad():
                _, du_0 = force(u_var, xi, du_0)
                u = torch.cat([zero_pad, u_var, zero_pad], dim=0)
            u_list[ii+1].vector().set_local(u.detach().cpu().numpy())
            progress_every = max(1, self.parameters["nt"] // 10)
            if (ii + 1) % progress_every == 0:
                t_ii = time.time()
                per_sample_time = (t_ii - t0) / (ii + 1)
                remaining = (self.parameters["nt"] - ii-1)*per_sample_time
                print("%d steps finished with %1.2fs per step. Estimated remaining time:" % (
                    ii + 1, per_sample_time) + str(timedelta(seconds=remaining)))
                sys.stdout.flush()
        if return_internal:
            return u_list, xi_list
        else:
            return u_list


def compute_error(ref_list, pred_list, Vh_ref, Vh_pred):
    
    if Vh_ref.dim() > Vh_pred.dim():
        u_trial, u_test = dl.TrialFunction(Vh_ref), dl.TestFunction(Vh_ref)
        tmp = dl.Function(Vh_ref).vector()
    elif Vh_ref.dim() < Vh_pred.dim():
        u_trial, u_test = dl.TrialFunction(Vh_pred), dl.TestFunction(Vh_pred)
        tmp = dl.Function(Vh_pred).vector()
    else:
        u_trial, u_test = dl.TrialFunction(Vh_ref), dl.TestFunction(Vh_ref)
        tmp = dl.Function(Vh_ref).vector()
    M = dl.assemble(dl.inner(u_trial, u_test)*dl.dx)
    error = np.zeros(len(ref_list))
    ref_norm = np.zeros(len(ref_list))
    dt = 1.0/(len(ref_list)-1)
    for ii, u1, u2 in zip(np.arange(len(ref_list)),ref_list, pred_list):
        if Vh_ref.dim() > Vh_pred.dim():
            u2_proj = dl.project(u2, Vh_ref)
            tmp.zero()
            tmp.axpy(1., u2_proj.vector())
            tmp.axpy(-1., u1.vector())
            ref_norm[ii] = u1.vector().inner(M*u1.vector())
        elif Vh_ref.dim() < Vh_pred.dim():
            u1_proj = dl.project(u1, Vh_pred)
            tmp.zero()
            tmp.axpy(1., u2.vector())
            tmp.axpy(-1., u1_proj.vector())
            ref_norm[ii] = u1_proj.vector().inner(M*u1_proj.vector())
        else:
            tmp.zero()
            tmp.axpy(1., u2.vector())
            tmp.axpy(-1., u1.vector())
            ref_norm[ii] = u1.vector().inner(M*u1.vector())
        error[ii] = tmp.inner(M*tmp)
    return np.sqrt(np.trapz(error, dx=dt)/np.trapz(ref_norm, dx=dt))