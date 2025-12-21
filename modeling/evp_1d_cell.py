import numpy as np
import dolfin as dl
import hippylib as hp
from .microstructure import MatrixInterpolation
from .utils.solver_utils import (type_error_message, create_mesh)
from typing import Any


def ElastoViscoplasticity1DSettings():
    """"
        Settings for Elasto-Visco-Plasticity 1D simulation in unit cell.
    """
    settings = {"n_cells": 100,
                "FE_order": 1,
                "nt": 1000,
                "T": 1.0}
    return settings

class ElastoViscoplasticity1DCellProblem:
    def __init__(self, n_cells=100, FE_order=1) -> None:
        self._p, self._n_cells = FE_order, n_cells
        self._mesh, pbc = create_mesh(1, n_cells)
        self._create_spaces(pbc)
        self.parameters = {}
        self.parameters["verbose"] = False
        self.parameters["nt"] = 1000
        self.parameters["T"] = 1.0
    
    def _create_spaces(self, pbc: object) -> None:
        """
        :param pbc: a periodic boundary with methods `inside()` and `map()`.
        """
        CGp = dl.FiniteElement("Lagrange", self._mesh.ufl_cell(), self.FE_order)
        self._Vh = dl.FunctionSpace(self._mesh, CGp)  # Scalar
        self._Ph = dl.FunctionSpace(self._mesh, CGp, constrained_domain=pbc)  # Periodic Vector

        # Set up helper vectors
        self._help_periodic = self.generate_vector(type='periodic')
        self._help_free = self.generate_vector(type='free')

    def FunctionSpace(self, type: str = None) -> dl.FunctionSpace:
        """
        :param type: type of function space to return
        :return: requested function space
        """
        if type is None or type == "free":
            return self._Vh
        elif type == "periodic":
            return self._Ph
        else:
            raise Exception(type_error_message())

    def generate_vector(self, type: str = None) -> dl.Vector:
        """
        :param type: the type of dolfin vector to return
        :return: requested dolfin vector
        """
        if type is None or type == "free":
            return dl.Function(self._Vh).vector()
        elif type == "periodic":
            return dl.Function(self._Ph).vector()
        else:
            raise Exception(type_error_message())

    def set_microstructure(self, youngs_modulus: Any, rate_constant: Any, yield_stress: Any, rate_exponent: Any) -> None:
        """
        The microstructures must be given as one the following type:
        (1) dolfin vectors; (2) matrices; (3) functions; (4) an ufl matrix
        """
        microstructure = [youngs_modulus, rate_constant, yield_stress, rate_exponent]
        error_message = ("The size of the matrix does not match the free function space nor the periodic"
                         "function space")
        for count, arg in enumerate(microstructure):
            if isinstance(arg, dl.cpp.la.PETScVector):
                if arg.size() == self._Vh.dim():
                    func = hp.vector2Function(arg, self._Vh)
                elif arg.size() == self._Ph.dim():
                    func = hp.vector2Function(arg, self._Ph)
                else:
                    raise Exception(error_message)
                microstructure[count] = func
            elif isinstance(arg, np.ndarray):
                assert self.FE_order == 1
                if arg.size == self._Vh.dim():
                    func = MatrixInterpolation(arg, self._Vh)
                elif arg.size == self._Ph.dim():
                    func = MatrixInterpolation(arg, self._Ph)
                else:
                    raise Exception(error_message)
                microstructure[count] = func
            else:
                microstructure[count] = arg
        self._youngs_modulus, self._rate_constant, self._yield_stress, self._rate_exponent = microstructure[0], microstructure[1], microstructure[2], microstructure[3]
        self._E_inv_mean = dl.assemble(1./self._youngs_modulus*dl.dx)
    
    def solve(self, strain_func: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        :param strain_func: the strain function
        :return: the time points and stress
        """
        times = np.linspace(0, self.T, self.nt+1)
        stress = np.zeros(self.nt+1)
        plastic_strain = np.zeros(self.nt + 1)

        strain = strain_func(times).squeeze()
        assert strain.size == self.nt + 1

        for step_count, t in enumerate(times[1:]):
            term_1 = self._yield_stress**(-self._rate_exponent)
            term_2 = dl.Constant(self._E_inv_mean)**(-self._rate_exponent)
            term_3 = dl.Constant(np.abs(strain[step_count] - plastic_strain[step_count]))**self._rate_exponent
            plastic_strain_rate = np.sign(strain[step_count] - plastic_strain[step_count])*dl.assemble(self._rate_constant*term_1*term_2*term_3*dl.dx)
            
            plastic_strain[step_count+1] = plastic_strain[step_count] + self.dt * plastic_strain_rate
            stress[step_count+1] = (strain[step_count+1] - plastic_strain[step_count+1])/self._E_inv_mean
            
        return times, stress, plastic_strain
    
    @property
    def youngs_modulus(self):
        return self._youngs_modulus

    @property
    def yield_stress(self):
        return self._yield_stress

    @property
    def rate_constant(self):
        return self._rate_constant

    @property
    def rate_exponent(self):
        return self._rate_exponent

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
    def n_cells(self):
        return self._n_cells

    @property
    def FE_order(self):
        return self._p
