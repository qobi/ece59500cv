class bundle:
    def __init__(self, prim, tg):
        self.prim = prim
        self.tg = tg
    def __add__(self, y):
        return plus(self, y)
    def __radd__(self, x):
        return plus(x, self)
    def __mul__(self, y):
        return times(self, y)
    def __rmul__(self, x):
        return times(x, self)

def bun(x, x_tangent):
    return bundle(x, x_tangent)

def prim(x):
    if isinstance(x, bundle):
        return x.prim
    else:
        return x

def tg(x):
    if isinstance(x, bundle):
        return x.tg
    else:
        return 0

def lift_real_to_real(f, dfdx):
    def me(x):
        if isinstance(x, bundle):
            return bun(me(prim(x)), dfdx(prim(x))*tg(x))
        else:
            return f(x)
    return me

def lift_real_cross_real_to_real(f, dfdx1, dfdx2):
    def me(x1, x2):
        if isinstance(x1, bundle) or isinstance(x2, bundle):
            return bun(me(prim(x1), prim(x2)),
                       (dfdx1(prim(x1), prim(x2))*tg(x1)
                        +dfdx2(prim(x1), prim(x2))*tg(x2)))
        else:
            return f(x1, x2)
    return me

plus = lift_real_cross_real_to_real(lambda x1, x2: x1+x2,
                                    lambda x1, x2: 1,
                                    lambda x1, x2: 1)

times = lift_real_cross_real_to_real(lambda x1, x2: x1*x2,
                                     lambda x1, x2: x2,
                                     lambda x1, x2: x1)

def D(f):
    def me(x):
        return tg(f(bun(x, 1)))
    return me
