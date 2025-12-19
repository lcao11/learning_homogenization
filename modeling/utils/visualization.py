import dolfin as dl

def vector2Function(vec, Vh):
    func = dl.Function(Vh)
    func.vector().axpy(1., vec)
    return func

def plot_vec(vec, Vh, **kwargs):
    return dl.plot(vector2Function(vec, Vh), **kwargs)

def extract_component(vec, mixed_vec, Uh, Zh, component=0):
    func = dl.Function(Zh)
    func_sub = dl.Function(Uh)
    func.vector().zero()
    func_sub.vector().zero()
    func.vector().axpy(1., mixed_vec)
    dl.assign(func_sub, func.sub(component))
    vec.zero()
    vec.axpy(1., func_sub.vector())
