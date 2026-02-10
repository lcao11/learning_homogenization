import math
import numpy as np
import dolfin as dl
import hippylib as hp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
import random
from typing import Any


class RandomTrajectory:
    """
    This class produces 1D gaussian random fields for a random trajectory.
    """

    def __init__(self, correlation_time: list[float], marginal_std: list[float], T: float = 1.0,
                 nt_min: int = 1000) -> None:
        """
        :param correlation_time: the correlation time for the random trajectory
        :param marginal_std: the marginal std for the random trajectory
        :param T: the end time
        :param nt_min: the minimum number of time steps. The actual number of time steps will depend on
        the correlation time
        """
        if isinstance(correlation_time, list):
            correlation_time = np.array(correlation_time)
        if isinstance(marginal_std, list):
            marginal_std = np.array(marginal_std)
        assert len(correlation_time) == len(marginal_std)
        self._dim = len(correlation_time)
        self._nt = max(nt_min, math.floor(1. / np.min(correlation_time) * 8))
        self._T = T
        self._mesh = dl.IntervalMesh(self._nt, 0.0, self._T)  # Here is where we define a unit interval mesh
        if self._dim == 1:
            self._Vh = dl.FunctionSpace(self._mesh, 'Lagrange', 1)
        else:
            self._Vh = dl.VectorFunctionSpace(self._mesh, 'Lagrange', 1, dim=self._dim)
        delta = 1.0 / (marginal_std * np.sqrt(correlation_time))
        gamma = correlation_time ** 2 * delta
        if self._dim == 1:
            self._GaussianMeasure = hp.BiLaplacianPrior(self._Vh, gamma[0], delta[0], robin_bc=True)
        else:
            self._GaussianMeasure = hp.VectorBiLaplacianPrior(self._Vh, gamma, delta, robin_bc=True)
        self._function = dl.Function(self._Vh)
        self._noise = dl.Vector(self._Vh.mesh().mpi_comm())
        self._GaussianMeasure.init_vector(self._noise, "noise")

    def sample(self) -> None:
        """
        Generate a random time sequence
        """
        hp.parRandom.normal(1., self._noise)
        self._GaussianMeasure.sample(self._noise, self._function.vector())

    def __call__(self, *args) -> np.ndarray:
        """
        :param args: the first argument must be the time query point (float)
        :return: a numpy array of function values at the given time.
        """
        if len(args) > 2:
            raise Exception("Cannot be called with more than two argument: time (scalar) and value (array).\n\
            Note that the random process does not depend on the value argument")
        t = args[0]
        if t > self._T:
            assert abs(t - self._T) < 1.0e-6 * self._T / float(self._nt)
            t = self.T
        if t < 0.0:
            assert abs(t - 0) < 1.0e-6 * self._T / float(self._nt)
            t = 0
        if self._dim == 1:
            return np.expand_dims(self._function(t), axis=0)
        else:
            return self._function(t)

    def consume_random(self):
        """
        Consume random time sequence
        """
        hp.parRandom.normal(1., self._noise)

    def plot(self, component: int = None, **kwargs):
        """
        This function plot the stored random time series
        :param component: the component for visualization
        :param kwargs: the arguments to be handed to a typical matplotlib function.
        :return: the plot object
        """
        if component is None and self._dim > 1:
            raise ("Assign the component intended for visualization")
        if self.dim > 1:
            return dl.plot(self._function[component], **kwargs)
        elif self.dim == 1:
            dl.plot(self._function, **kwargs)

    def time_sequence(self) -> np.ndarray:
        """
        This function return the store time sequence with shape (nt, dim)
        """
        return np.linspace(0, self._T, self._nt), self._function.compute_vertex_values().reshape((self._nt, self._dim))

    @property
    def T(self):
        return self._T

    @property
    def nt(self):
        return self._nt

    @property
    def dim(self):
        return self._dim


class ODETrajectory:
    """
    This class creates a function as a time integration of another function.
    """

    def __init__(self, fun: Any, initial_value: Any = None, T: float = 1.0) -> None:
        """
        :param fun: the function used for integration, must return time derivative values when input times and values
        :param initial_value: the list of initial values, also accept numpy arrays
        :param T: the end time
        """
        if isinstance(initial_value, list):
            initial_value = np.array(initial_value)
        if initial_value is None:
            self._dim = len(fun(0.0))
            initial_value = np.zeros(self._dim)
        else:
            self._dim = len(initial_value)
        self._fun = fun
        output = solve_ivp(fun, (0, T), initial_value, dense_output=True, rtol=1.e-8, atol=1.e-12)
        self._function = ODESolutionWrapper(output.sol)
        self._y = output.y
        self._t = output.t

    def __call__(self, *args):
        return self._function(args[0])

    def __neg__(self, *args):
        return -1 * self.__call__(args)

    def plot(self, component=None, **kwargs):
        if component is None and self._dim > 1:
            raise Exception("Assign the component intended for visualization")
        if self._dim == 1:
            return plt.plot(self._t, self._y[0], **kwargs)
        else:
            return plt.plot(self._t, self._y[component], **kwargs)

    def time_sequence(self):
        return self._t, self._y


class ODESolutionWrapper():
    def __init__(self, function):
        self._function = function

    def __call__(self, *args):
        if len(args) > 2:
            raise Exception("Cannot be called with more than two argument: time (scalar) and value (array).\n\
            Note that the random process does not depend on the value argument")
        t = args[0]
        return self._function(t)


class PchipTrajectory:
    def __init__(self, t_list, y_list, T: float = 1.0) -> None:
        """
        :param t_list: A list of numpy array of t values for interpolation
        :param y_list: A list of numpy array of y values for interpolation
        :param dydt_list: A list of numpuy array of dydt values for interpolation
        :param T: The end time
        """
        self._dim = len(t_list)
        assert len(y_list) == self._dim
        is_sorted = lambda a: np.all(a[:-1] <= a[1:])
        self._T = T
        self._func = []
        for points, values, in zip(t_list, y_list):
            if not is_sorted(points):
                raise Exception("Input sorted interpolation points with increasing values")
            assert points[0] == 0.0
            assert values[0] == 0.0
            self._func.append(PchipInterpolator(points, values))

    def __call__(self, t):
        return np.array([self._func[i](t) for i in range(self._dim)]).T

    def plot(self, component=None, **kwargs):
        t = np.linspace(0, self._T, 1000)
        if component is None and self._dim > 1:
            raise Exception("Assign the component intended for visualization")
        if self._dim == 1:
            return plt.plot(t, self._func[0](t), **kwargs)
        else:
            return plt.plot(t, self._func[component](t), **kwargs)

    def rate(self):
        """
        :return: A rate function that allows for evaluations and plots.
        """
        return RateWrapper(self._func, self._T)


class RateWrapper:
    """
    The class is a wrapper for the derivative of the cubic Hermite polynomial that allows for derivative evaluations
    """

    def __init__(self, func: list, T: float) -> None:
        self._dim = len(func)
        self._T = T
        self._func = [f.derivative() for f in func]

    def __call__(self, t):
        return np.array([self._func[i](t) for i in range(self._dim)]).T

    def plot(self, component=None, **kwargs):
        t = np.linspace(0, self._T, 1000)
        if component is None and self._dim > 1:
            raise Exception("Assign the component intended for visualization")
        if self._dim == 1:
            return plt.plot(t, self._func[0](t), **kwargs)
        else:
            return plt.plot(t, self._func[component](t), **kwargs)


def generate_trajectory(n_interval, dim, low=-1.0, high=1.0, T=1.0):
    """
    This is a function to generate a random CubicHermite trajectories with values bounded by the specified upper and
    lower values.
    :param n_interval: the number of intervals
    :param high: the upper bound
    :param low: the lower bound
    :param dim: the dimension of the trajectory
    :param T: the end time
    :return: the trajectory object, the list of time points, the list of values, the list of derivative values
    """
    t_list = []
    value_list = []
    assert low < high
    assert n_interval >= 2

    for ii in range(dim):
        n_points = np.random.randint(3, high=n_interval+2)
        time = np.zeros(n_points)
        time[-1] = T
        time[1:-1] = np.sort(np.random.choice(np.linspace(0, T, 51)[1:-1], n_points-2, replace=False))
        value = np.zeros(n_points)
        for jj in range(n_points-1):
            sign = random.choice([-1, 1])
            max_value = high-value[jj] if sign > 0 else value[jj]-low            
            value[jj+1] = value[jj] + sign*max_value*np.sqrt(time[jj+1]-time[jj])
        t_list.append(time)
        value_list.append(value)

    trajectory = PchipTrajectory(t_list, value_list)

    return trajectory, t_list, value_list