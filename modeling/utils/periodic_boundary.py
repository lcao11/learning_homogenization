import dolfin as dl


class PeriodicBoundary1D(dl.SubDomain):
    """
    Creating a periodic boundary in the unit cell
    """

    def inside(self, x, on_boundary):
        return bool(- dl.DOLFIN_EPS < x[0] < dl.DOLFIN_EPS and on_boundary)

    def map(self, x, y):
        y[0] = x[0] - 1.


class PeriodicBoundary2D(dl.SubDomain):
    """
    Creating a periodic boundary in the unit cell
    """

    def inside(self, x, on_boundary):
        return bool((dl.near(x[0], 0) or dl.near(x[1], 0)) and
                    (not ((dl.near(x[0], 0) and dl.near(x[1], 1)) or
                          (dl.near(x[0], 1) and dl.near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        if dl.near(x[0], 1) and dl.near(x[1], 1):
            y[0] = x[0] - 1.
            y[1] = x[1] - 1.
        elif dl.near(x[0], 1):
            y[0] = x[0] - 1.
            y[1] = x[1]
        else:
            y[0] = x[0]
            y[1] = x[1] - 1.


class PeriodicBoundary3D(dl.SubDomain):
    """
    See https://fenicsproject.discourse.group/t/3d-periodic-boundary-condition/6199
    """

    def inside(self, x, on_boundary):
        # The origin is inside
        return bool((dl.near(x[0], 0.0) or dl.near(x[1], 0.0) or dl.near(x[2], 0.0)) and
                    (not ((dl.near(x[0], 1.0) and dl.near(x[2], 1.0)) or
                          (dl.near(x[0], 1.0) and dl.near(x[1], 1.0)) or
                          (dl.near(x[1], 1.0) and dl.near(x[2], 1.0)))) and on_boundary)

    def map(self, x, y):
        # Mappings of corners
        if dl.near(x[0], 1.0) and dl.near(x[1], 1.0) and dl.near(x[2], 1.0):
            y[0] = x[0] - 1.0
            y[1] = x[1] - 1.0
            y[2] = x[2] - 1.0
        # Mappins of edges
        elif dl.near(x[0], 1.0) and dl.near(x[2], 1.0):
            y[0] = x[0] - 1.0
            y[1] = x[1]
            y[2] = x[2] - 1.0
        elif dl.near(x[1], 1.0) and dl.near(x[2], 1.0):
            y[0] = x[0]
            y[1] = x[1] - 1.0
            y[2] = x[2] - 1.0
        elif dl.near(x[0], 1.0) and dl.near(x[1], 1.0):
            y[0] = x[0] - 1.0
            y[1] = x[1] - 1.0
            y[2] = x[2]
        # Mapping of left to right faces
        elif dl.near(x[0], 1.0):
            y[0] = x[0] - 1.0
            y[1] = x[1]
            y[2] = x[2]
        # Mapping of front to back faces
        elif dl.near(x[1], 1.0):
            y[0] = x[0]
            y[1] = x[1] - 1.0
            y[2] = x[2]
        # Mapping of top to bottom faces
        elif dl.near(x[2], 1.0):
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2] - 1.0
        else:
            y[0] = -1000
            y[1] = -1000
            y[2] = -1000
