# Author: Lianghao Cao
# Date: Jan. 12, 2023
import numpy as np
import dolfin as dl
import hippylib as hp
import ufl
from scipy.spatial import cKDTree
from typing import Any
from .utils.solver_utils import identity_4th, trace_4th
from .utils.random_field import PeriodicBiLaplacianPrior
from .utils.periodic_boundary import PeriodicBoundary1D, PeriodicBoundary2D, PeriodicBoundary3D
from .utils.projection import Projector
from .utils.periodic_kdtree import PeriodicCKDTree


def MatrixInterpolation(matrix: np.ndarray, Vh: dl.FunctionSpace) -> dl.cpp.la.PETScVector:
    """
        This is a method that convert the vertex values in pixel orders into the dof values if FEM order.
        :param matrix:  numpy array containing vertex values. Can be a 1d, 2d, 3d, or flattened array.
        :param Vh: the dolfin function space for the interpolation
        :return: a dolfin vector for dof values
    """
    if not Vh.dim() == matrix.size:
        raise Exception(
            "The function space has dimension of %d and the matrix has %d elements" % (Vh.dim(), matrix.size))
    try:
        v2d = dl.dof_to_vertex_map(Vh)
    except:
        raise Exception(
            "Cannot assign vertex values. Possibly high order polynomial is used for FE. Consider using P1 FE or use "
            "`dolfin.Expression` or `dolfin.UserExpression` for interpolation")
    out = dl.Function(Vh)
    out.vector().set_local(matrix.flatten()[v2d])
    return out


def MatrixExtraction(x: dl.cpp.la.PETScVector, Vh: dl.FunctionSpace) -> np.ndarray:
    """
        This is a method that convert the dof values if FEM order into the vertex values in pixel orders.
        :param x:  a dolfin vector for dof values
        :param Vh: the dolfin function space for the interpolation
        :return: a numpy array for matrix representation of the vertex values
    """
    if not Vh.dim() == x.size():
        raise Exception(
            "The function space has dimension of %d and the vector has %d" % (Vh.dim(), x.size))
    try:
        d2v = dl.vertex_to_dof_map(Vh)
    except:
        raise Exception(
            "Cannot assign vertex values. Possibly high order polynomial is used for FE. Consider using P1 FE or use "
            "`dolfin.Expression` or `dolfin.UserExpression` for interpolation")
    dim = Vh.mesh().geometric_dimension()
    n_x = int(Vh.dim() ** (1. / dim))
    shape = tuple([n_x] * dim)
    out = x.get_local()[d2v].reshape(shape)
    return out


def TensorExtraction(Vh, tensor, order=4):
    """
    :param Vh: A scalar valued function space
    :param tensor: A ufl form for the tensor
    :param order: the order of the tensor, default is 4.
    :return: a numpy array with first `dim` axis for the spatial position and last `order` axis for the tensor.
    """
    dim = Vh.mesh().geometric_dimension()
    V_tensor = dl.TensorFunctionSpace(Vh.mesh(), "CG", 1, shape=tuple([dim] * order))
    function = dl.project(tensor, V_tensor)
    d2v = dl.vertex_to_dof_map(V_tensor)
    n_x = int(Vh.dim() ** (1. / dim))
    shape = [n_x] * dim
    shape.extend([dim] * order)
    out = function.vector().get_local()[d2v].reshape(shape)
    return out


class Ellipse(dl.UserExpression):
    def __init__(self, center, inv, values, degree=3):
        self._center = center
        self._inv = inv
        self._values = values
        super(Ellipse, self).__init__(degree=degree)

    def _check_inclusion(self, x):
        return np.linalg.norm(self._inv @ (x - self._center)) <= 1

    def eval(self, value, x):
        if self._check_inclusion(x):
            value[0] = self._values[0]
        else:
            value[0] = self._values[1]

    def value_shape(self):
        return ()


def EllipseMaterial(Vh: dl.FunctionSpace, center=None, matrix=None, values=None):
    assert len(center) == Vh.mesh().geometric_dimension()
    assert len(values) == 2
    assert matrix.shape[0] == matrix.shape[1] and matrix.shape[0] == Vh.mesh().geometric_dimension()
    if isinstance(center, list):
        center = np.array(center)
    inv = np.linalg.inv(matrix)
    material_init = Ellipse(center, inv, values, degree=3)
    material = dl.Function(Vh)
    material.interpolate(material_init)
    return material.vector()


class RandomMaterial:
    """
    This is a class that can be used to generate random field material properties. The material is represented as `E(
    x) = (a-b)*(erf(s(x)) + 1)/2 + b`, where a and b are the upper and lower bound, and s is a Gaussian random field
    with bi-Laplacian inverse covariance and robin boundary.
    """

    def __init__(self, Vh: dl.FunctionSpace, bound: list[float, float] = None,
                 correlation_length: float = 0.01, pointwise_std: float = 0.5, anisotropic_angles: [float, ...] = None,
                 anisotropic_scalings: [float, ...] = None, mean: Any = None, max_iter: int = 500,
                 rel_tol: float = 1.e-10):
        """
        :param Vh: the material function space
        :param bound: a list of the lower and upper bounds
        :param correlation_length: the spatial correlation length for the Gaussian random field
        :param pointwise_std: a pointwise standard deviation for the Gaussian random field
        :param mean: the mean function for the Gaussian random field
        :param max_iter: the maximum iteration for Krylov solver in 3D
        :param rel_tol: the relative tolerance for Krylov solver in 3D
        """
        if bound is None:
            bound = [0.0, 1.0]
        assert bound[0] < bound[1]
        self._upper_bound = bound[1]
        self._lower_bound = bound[0]
        self._diff = self._upper_bound - self._lower_bound

        self._Vh = Vh
        dim = Vh.mesh().geometric_dimension()
        if not mean is None:
            if isinstance(mean, dl.cpp.la.PETScVector):
                self._mean_func = hp.vector2Function(mean, Vh)
            elif isinstance(mean, np.ndarray):
                self._mean_func = MatrixInterpolation(mean, Vh)
            else:
                self._mean_func = mean
        else:
            self._mean_func = dl.Constant(0.0)

        delta = None
        if dim == 1:
            delta = 1.0 / (pointwise_std * np.sqrt(correlation_length))
        elif dim == 2:
            delta = 1.0 / (pointwise_std * correlation_length)
        elif dim == 3:
            delta = 1.0 / (pointwise_std * correlation_length ** 1.5)
        gamma = correlation_length ** 2 * delta
        anis_diff = None
        if anisotropic_angles is not None and anisotropic_scalings is not None:
            if dim == 2:
                assert len(anisotropic_scalings) == 2
                if isinstance(anisotropic_angles, float): anisotropic_angles = [anisotropic_angles]
                anis_diff = dl.CompiledExpression(hp.ExpressionModule.AnisTensor2D(), degree=1)
                anis_diff.set(anisotropic_scalings[0], anisotropic_scalings[1], anisotropic_angles[0])
        self._GaussianMeasure = hp.BiLaplacianPrior(self._Vh, gamma, delta, Theta=anis_diff, max_iter=max_iter,
                                                    rel_tol=rel_tol)
        self._help = dl.Vector(self._Vh.mesh().mpi_comm())
        self._noise = dl.Vector(self._Vh.mesh().mpi_comm())
        self._GaussianMeasure.init_vector(self._noise, "noise")
        self._GaussianMeasure.init_vector(self._help, 0)
        self._projector = Projector(Vh, Vh, max_iter=max_iter, rel_tol=rel_tol)

    def _map(self, m_func: dl.Function) -> dl.cpp.la.PETScVector:
        """
        :param m: the dolfin vector for a Gaussian random field sample
        :return: transformed random field sample
        """
        out = self._projector.project(
            dl.Constant(0.5 * self._diff) * (ufl.erf(m_func + self._mean_func) + dl.Constant(1.0)) + dl.Constant(
                self._lower_bound))
        return out.vector()

    def sample(self) -> dl.cpp.la.PETScVector:
        """
        :return: a random field sample
        """
        hp.parRandom.normal(1., self._noise)
        self._GaussianMeasure.sample(self._noise, self._help)
        return self._map(hp.vector2Function(self._help, self._Vh))

    def consume_random(self):
        """
        consume one random sample
        """
        hp.parRandom.normal(1., self._noise)


class PeriodicRandomMaterial(RandomMaterial):
    """
    This is a class that can be used to generate periodic random field material properties. The material is
    represented as `E(x) = (a-b)*(erf(s(x)) + 1)/2 + b`, where a and b are the upper and lower bound, and s is a
    periodic Gaussian random field with bi-Laplacian inverse covariance and periodic boundaries.
    """

    def __init__(self, Vh: dl.FunctionSpace, bound: list[float, float] = None, correlation_length: float = 0.01,
                 pointwise_std: float = 0.5, anisotropic_angles: [float, ...] = None,
                 anisotropic_scalings: [float, ...] = None, mean: Any = None,
                 max_iter: int = 500, rel_tol: float = 1.e-10) -> None:
        super(PeriodicRandomMaterial, self).__init__(Vh, bound=bound, correlation_length=correlation_length,
                                                     pointwise_std=pointwise_std, anisotropic_angles=anisotropic_angles,
                                                     anisotropic_scalings=anisotropic_scalings, mean=mean,
                                                     max_iter=max_iter,
                                                     rel_tol=rel_tol)
        """
        :param Vh: the material function space
        :param bound: a list of the lower and upper bounds
        :param correlation_length: the spatial correlation length for the Gaussian random field
        :param pointwise_std: a pointwise standard deviation for the Gaussian random field
        :param anisotropic_angles: the angles for the anisotropic tensor. (1 angle for 2d and 3 angles for 3d)
        :param anisotropic_scalings: the diagonal scalings for the anisotropic tensor. (2 scalings for 2d and 3 scalings for 3d)
        :param mean: the mean function for the Gaussian random field, must coincide with `Vh`
        :param max_iter: the maximum iteration for Krylov solver in 3D
        :param rel_tol: the relative tolerance for Krylov solver in 3D
        """
        dim = Vh.mesh().geometric_dimension()
        pbc = None
        if dim == 1:
            pbc = PeriodicBoundary1D()
        elif dim == 2:
            pbc = PeriodicBoundary2D()
        elif dim == 3:
            pbc = PeriodicBoundary3D()
        self._Ph = dl.FunctionSpace(Vh.mesh(), Vh.ufl_element(), constrained_domain=pbc)

        delta = None
        if dim == 1:
            delta = 1.0 / (pointwise_std * np.sqrt(correlation_length))
        elif dim == 2:
            delta = 1.0 / (pointwise_std * correlation_length)
        elif dim == 3:
            delta = 1.0 / (pointwise_std * correlation_length ** 1.5)
        gamma = correlation_length ** 2 * delta

        anis_diff = None
        if anisotropic_angles is not None and anisotropic_scalings is not None:
            if dim == 2:
                assert len(anisotropic_scalings) == 2
                if isinstance(anisotropic_angles, float): anisotropic_angles = [anisotropic_angles]
                anis_diff = dl.CompiledExpression(hp.ExpressionModule.AnisTensor2D(), degree=1)
                anis_diff.set(anisotropic_scalings[0], anisotropic_scalings[1], anisotropic_angles[0])
        self._GaussianMeasure = PeriodicBiLaplacianPrior(self._Ph, gamma, delta, Theta=anis_diff, max_iter=max_iter,
                                                         rel_tol=rel_tol)
        self._help = dl.Vector(self._Ph.mesh().mpi_comm())
        self._noise = dl.Vector(self._Ph.mesh().mpi_comm())
        self._GaussianMeasure.init_vector(self._noise, "noise")
        self._GaussianMeasure.init_vector(self._help, 0)
        self._projector = Projector(self._Ph, self._Vh, max_iter=max_iter, rel_tol=rel_tol)

    def sample(self) -> dl.cpp.la.PETScVector:
        """
        :return: a random field sample
        """
        hp.parRandom.normal(1., self._noise)
        self._GaussianMeasure.sample(self._noise, self._help)
        return self._map(hp.vector2Function(self._help, self._Ph))


class PC1D(dl.UserExpression):
    """
    A dolfin expression for interpolating a 1D piecewise constant material with given discontinuity points and values
    """

    def __init__(self, points, values, degree, periodic):
        """
        :param points: the discontinuity points, must be sorted
        :param values: the piecewise constant values, must be sorted
        :param degree: the finite element polynomial degree for interpolation
        :param periodic: whether to the material is periodic. Default is True.
        """
        self._points = points
        self._values = values
        self._periodic = periodic
        if periodic:
            assert len(self._values) == len(self._points)
        else:
            assert len(self._values) == len(self._points) + 1
        super(PC1D, self).__init__(degree=degree)

    def eval(self, value, x):
        idx = np.digitize(x[0], self._points)
        if self._periodic:
            value[0] = self._values[idx - 1] if idx < self._points.size else self._values[-1]
        else:
            value[0] = self._values[idx]

    def value_shape(self):
        return ()


def PC1DMaterial(Vh: dl.FunctionSpace, points: np.ndarray, values: np.ndarray, periodic: bool = True,
                 degree: int = 3) -> dl.cpp.la.PETScVector:
    """
    A function for generating piecewise constant material in 1D
    :param Vh: the function space intended for interpolation. Must be a scalar function space
    :param points: the discontinuity points. The points must be sorted from 0 to 1
    :param values: the values of the piecewise constant function. The values are assign to pieces starting from left to right
    :param periodic: whether to return a piecewise constant material
    :param degree: the degree of material interpolation
    :return: a dolfin vector representing the material dofs
    """
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    if not is_sorted(points):
        raise Exception("Input sorted discontinuity points with increasing values")
    # assert all(values > 0.0)
    assert all(points > 0.0)
    assert all(points < 1.0)
    material_init = PC1D(points, values, periodic=periodic, degree=degree)
    material = dl.Function(Vh)
    material.interpolate(material_init)
    return material.vector()


class Voronoi(dl.UserExpression):
    def __init__(self, points, values, degree, periodic):
        self._dim = points.shape[1]
        self._values = values
        self._periodic = periodic
        if self._periodic:
            self._voronoi_kdtree = PeriodicCKDTree(points)
        else:
            self._voronoi_kdtree = cKDTree(points)
        super(Voronoi, self).__init__(degree=degree)

    def _find_inclusion(self, x):
        position = [[x[i] for i in range(self._dim)]]
        _, idx = self._voronoi_kdtree.query(np.array(position))
        return idx

    def eval(self, value, x):
        idx = self._find_inclusion(x)
        value[0] = self._values[idx]

    def value_shape(self):
        return ()


def VoronoiMaterial(Vh: dl.FunctionSpace, points: np.ndarray, values: np.ndarray,
                    degree: int = 5, periodic=True) -> dl.cpp.la.PETScVector:
    """
    A function for generating Voronoi material in general dimensions of 1, 2, and 3
    :param Vh: the function space intended for interpolation. Must be a scalar function space
    :param points: the points that generates the Voronoi. Must have shape (n_point, n_dim).
    :param values: the values of Voronoi pieces. Must have shape (n_point,)
    :param degree: the degree of material interpolation
    :param periodic: whether to generate periodic Voronoi. Default is True
    :return: a dolfin vector representing the material dofs
    """
    assert values.size == points.shape[0]
    assert points.shape[1] == Vh.mesh().geometric_dimension()
    material_init = Voronoi(points, values, degree=degree, periodic=periodic)
    material = dl.Function(Vh)
    material.interpolate(material_init)
    return material.vector()


def AssembleSPDTensor(angle: Any, d1: Any, d2: Any, Vh: dl.FunctionSpace = None) -> ufl.tensoralgebra.Dot:
    """
    This function assembles a SPD tensor for scalar 2D problem.
    :param angle: the angle in the rotation matrix.
    :param d1: the first diagonal element in the diagonal matrix
    :param d2: the second diagonal element in the diagonal matrix
    :param Vh: the function space for matrix interpolation and vector to function operations. Not
    needed if all the properties a given in dolfin.Function or ufl.Forms
    :return: the ufl forms for the SPD tensor.
    """
    property_forms = [None] * 3
    for i, arg in enumerate([angle, d1, d2]):
        if isinstance(arg, dl.cpp.la.PETScVector):
            if Vh is None:
                raise Exception("Provide the function space through Vh=your_space")
            property_forms[i] = hp.vector2Function(arg, Vh)
        elif isinstance(arg, np.ndarray):
            if Vh is None:
                raise Exception("Provide the function space through Vh=your_space")
            property_forms[i] = MatrixInterpolation(arg, Vh)
        else:
            property_forms[i] = arg
    R = dl.as_matrix([[dl.cos(property_forms[0]), dl.Constant(-1.0) * dl.sin(property_forms[0])],
                      [dl.sin(property_forms[0]), dl.cos(property_forms[0])]])
    D = dl.as_matrix([[property_forms[1], 0], [0, property_forms[2]]])
    DRt = dl.dot(D, ufl.transpose(R))
    RDRt = dl.dot(R, DRt)
    return RDRt


def AssembleIsotropicTensor(Vh: dl.FunctionSpace, modulus: Any, ratio: Any,
                            type: str = "plane_strain") -> ufl.algebra.Sum:
    """
    :param modulus: the Young's modulus of the material
    :param ratio: the poisson ratio
    :param Vh: the function space
    :param type: the type of Isotropic tensor. Plane stress and plane strain conditions are available
    :return: the ufl forms for the 4th order isotropic tensor.
    """
    forms = [0.0 for i in range(2)]
    for i, arg in enumerate([modulus, ratio]):
        if isinstance(arg, dl.cpp.la.PETScVector):
            if Vh is None:
                raise Exception("Provide the function space through Vh=your_space")
            forms[i] = hp.vector2Function(arg, Vh)
        elif isinstance(arg, np.ndarray):
            if Vh is None:
                raise Exception("Provide the function space through Vh=your_space")
            forms[i] = MatrixInterpolation(arg, Vh)
        else:
            forms[i] = arg
    mu = forms[0] / (2 * (1 + forms[1]))
    lmbda = forms[0] * forms[1] / ((1 + forms[1]) * (1 - 2 * forms[1]))
    if type == "plane_stress":
        lmbda = 2 * mu * lmbda / (lmbda + 2 * mu)
    elif type != "plane_strain":
        raise Exception("Can only accept type='plane_stress' and type='plane_strain'.")
    dim = Vh.mesh().geometric_dimension()
    I, T = identity_4th(dim), trace_4th(dim)
    return lmbda * T + 2.0 * mu * I


def voronoi_function(x, points, values, periodic=True):
    """
    This function is used to analytically evaluate a voronoi tessellated spatial function
    :param x: the positions for evaluation. Can do batch evaluations.
    :param points: the points for voronoi. Must have shape (n_points, dim)
    :param values: the values for voronoi.
    :param periodic: whether to use periodic voronoi. Default to be true
    :return: the analytical values of the voronoi function
    """
    dim = points.shape[1]
    assert dim <= 3
    if periodic:
        voronoi_kdtree = PeriodicCKDTree(points)
    else:
        voronoi_kdtree = cKDTree(points)
    _, idx = voronoi_kdtree.query(x)
    return values[idx]
