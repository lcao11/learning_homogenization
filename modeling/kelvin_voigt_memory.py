import numpy as np
from scipy.integrate import simpson

class KelvinVoigtHomogenizedModel:
    def __init__(self, viscous=None, elastic=None, kernel=None, T=1.0):
        self._nu_p = viscous
        self._E_p = elastic
        self._kernel = kernel
        self._t = np.linspace(0.0, T, kernel.shape[0])
        self._dt = np.mean(self._t[1:] - self._t[:-1])

    def predict(self, ebar, rate):
        """Predict stress via the memory-form constitutive law.

        Uses the convention

            sigma(t) = E' e(t) + nu' e_dot(t) - \int_0^t K(t-s) e(s) ds,

        where `e`, `e_dot`, and `sigma` are Kelvin--Mandel Voigt vectors of length `d_sym`.
        """

        if isinstance(ebar, np.ndarray) and isinstance(rate, np.ndarray):
            ebar_array = np.asarray(ebar)
            rate_array = np.asarray(rate)
        else:
            ebar_array = np.asarray(ebar(self._t))
            rate_array = np.asarray(rate(self._t))

        # Coerce 1D trajectories to (nt+1, 1)
        if ebar_array.ndim == 1:
            ebar_array = ebar_array[:, None]
        if rate_array.ndim == 1:
            rate_array = rate_array[:, None]

        # Accept either (nt+1, d) or (d, nt+1)
        if ebar_array.shape[0] != self._t.size and ebar_array.shape[-1] == self._t.size:
            ebar_array = ebar_array.T
        if rate_array.shape[0] != self._t.size and rate_array.shape[-1] == self._t.size:
            rate_array = rate_array.T

        if ebar_array.shape[0] != self._t.size or rate_array.shape[0] != self._t.size:
            raise ValueError(
                "Trajectory length mismatch: expected first dimension to be nt+1=%d (times), got ebar %s and rate %s"
                % (self._t.size, ebar_array.shape, rate_array.shape)
            )

        d = int(self._nu_p.shape[0])
        if ebar_array.shape[1] != d or rate_array.shape[1] != d:
            raise ValueError(
                "Trajectory component mismatch: expected d_sym=%d, got ebar %s and rate %s"
                % (d, ebar_array.shape, rate_array.shape)
            )

        stress = np.zeros((self._t.size, d))

        # t = 0: integral term is zero.
        stress[0] = self._nu_p @ rate_array[0] + self._E_p @ ebar_array[0]

        for ii, _time in enumerate(self._t[1:]):
            step = ii + 1
            stress[step] = self._nu_p @ rate_array[step] + self._E_p @ ebar_array[step]

            # Discrete approximation of \int_0^{t_step} K(t_step - s) e(s) ds.
            # With uniform time grid, this is a convolution with the kernel sampled at t_k.
            kernel_flip = np.flip(self._kernel[: step + 1], 0)
            integrand = np.einsum("ijk,ik->ij", kernel_flip, ebar_array[: step + 1])
            stress[step] -= simpson(integrand, dx=self._dt, axis=0)

        return self._t, stress

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
