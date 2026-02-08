import numpy as np
from scipy.integrate import simpson
from scipy.linalg import eigh
import math


def _flip(angles, index):
    for ii in range(index.shape[0]):
        if ii % 2 == 0:
            angles[index[ii, 0]:index[ii + 1, 0], index[ii, 1]] = math.pi - angles[index[ii, 0]:index[ii + 1, 0],
                                                                            index[ii, 1]]
    return angles


class KelvinVoigtHomogenizedModel:
    def __init__(self, viscous=None, elastic=None, kernel=None, T=1.0):
        self._nu_p = viscous
        self._E_p = elastic
        self._kernel = kernel
        self._t = np.linspace(0.0, T, kernel.shape[0])
        self._dt = np.mean(self._t[1:] - self._t[:-1])

    def predict(self, ebar, rate):
        if isinstance(ebar, np.ndarray) and isinstance(rate, np.ndarray):
            ebar_array = ebar
            rate_array = rate
        else:
            ebar_array = ebar(self._t)
            rate_array = rate(self._t)
        stress = np.zeros((self._t.size, self._nu_p.shape[0]))
        stress[0] = self._nu_p @ rate_array[0]
        for ii, time in enumerate(self._t[1:]):
            step = ii + 1
            stress[step] = self._nu_p @ rate_array[step] + self._E_p @ ebar_array[step]
            kernel_flip = np.flip(-self._kernel[:step + 1], 0)
            integrand = np.einsum("ijk, ik -> ij", kernel_flip, ebar_array[:step + 1])
            stress[step] += simpson(integrand, dx=self._dt, axis=0)
        return self._t, stress

    def eigen(self, type=None):
        if type == "viscous":
            s, basis = eigh(self._nu_p)
            angles = np.arccos(np.diagonal(np.abs(basis)))
            return s, angles
        elif type == "elastic":
            s, basis = eigh(self._E_p)
            angles = np.arccos(np.diagonal(np.abs(basis)))
            return s, angles
        elif type == "kernel":
            angles = np.zeros((self.nt + 1, 3))
            s = np.zeros((self.nt + 1, 3))
            for ii, time in enumerate(self._t):
                s[ii], basis = eigh(self._kernel[ii])
                angles[ii] = np.arccos(np.diagonal(np.abs(basis)))
            index = self._check_for_reflection(angles)
            angles = _flip(angles, index)
            return s, angles
        else:
            raise Exception("Wrong type. Must be viscous, elastic or kernel.")

    def _check_for_reflection(self, angles):
        angels_1st_rate = np.gradient(angles, self.dt, axis=0, edge_order=2)
        angels_2nd_rate = np.gradient(angels_1st_rate, self.dt, axis=0, edge_order=2)
        index = np.vstack(
            np.where(np.abs(angels_2nd_rate - np.mean(angels_2nd_rate, axis=0)) > 5 * np.std(angels_2nd_rate))).T[1::3]
        unique, count = np.unique(index[:, 1], return_counts=True)
        for ii, id in enumerate(unique):
            if count[ii] % 2 != 0:
                index = np.vstack((index, np.array([0, id])))
        for ii in range(2):
            index = index[index[:, ii].argsort()]
        return index

    @property
    def viscous(self):
        return self._nu_p

    @property
    def elastic(self):
        return self._E_p

    @property
    def kernel(self):
        return self._kernel

    @property
    def T(self):
        return self._t[-1]

    @property
    def dt(self):
        return self._dt

    @property
    def nt(self):
        return self._t.size - 1

    @property
    def times(self):
        return self._t
