import dolfin as dl
import ufl
from .periodic_boundary import PeriodicBoundary1D, PeriodicBoundary2D, PeriodicBoundary3D


def create_mesh(dim: int, ncells: int) -> tuple[dl.Mesh, object]:
    """
    :param dim: The dimension of the problem
    :param ncells: The number of cells in each direction
    :return: A mesh on the unit domain and the periodic boundary condition.
    """
    if dim == 1:
        mesh = dl.UnitIntervalMesh(ncells)
        pbc = PeriodicBoundary1D()
    elif dim == 2:
        mesh = dl.UnitSquareMesh(ncells,
                                 ncells)
        pbc = PeriodicBoundary2D()
    elif dim == 3:
        mesh = dl.UnitCubeMesh(ncells,
                               ncells,
                               ncells)
        pbc = PeriodicBoundary3D()
    else:
        raise ValueError("Wrong dimensions provided. Should be 1, 2, or 3.")
    return mesh, pbc


def print_compute_time(total_time, time_list, label):
    assert len(time_list) == len(label)
    print("=========================================")
    print("Total time: %1.2fs" % (total_time))
    print("=========================================")
    for i in range(len(time_list)):
        if not time_list[i] is None:
            print(label[i] + ": %1.2fs (%1.2f%%)" % (time_list[i], time_list[i] / total_time * 100))
    print("=========================================")


def type_error_message():
    return "The give type is not supported. Choose from `free` (default), `periodic`, `scalar`, `scalar_periodic`, or `mixed`."


class LiftingFunction:
    def __init__(self, Vh: dl.FunctionSpace, d: int) -> None:
        self._Vh = Vh
        self._d = d
        str_x = ["x[%d]" % (i) for i in range(self._d)]
        self.shift = [[None for ii in range(self._d)] for jj in range(self._d)]
        for i in range(self._d):
            for j in range(self._d):
                expr_list = ["0.0"] * self._d
                expr_list[j] = str_x[i]
                expr = dl.Expression(tuple(expr_list), degree=5)
                self.shift[i][j] = dl.interpolate(expr, self._Vh)

    def generate(self, ebar):
        out = dl.Function(self._Vh).vector()
        for i in range(self._d):
            for j in range(self._d):
                out.axpy(ebar[i, j], self.shift[i][j].vector())
        return out


def identity_4th(d):
    delta = ufl.Identity(d)
    i, j, k, l = ufl.indices(4)
    return ufl.as_tensor(0.5 * (delta[i, k] * delta[j, l] + delta[i, l] * delta[j, k]), (i, j, k, l))


def trace_4th(d):
    delta = ufl.Identity(d)
    i, j, k, l = ufl.indices(4)
    return ufl.as_tensor(delta[i, j] * delta[k, l], (i, j, k, l))
