import math

class bundle:
    def __init__(self, prim, tg):
        self.prim = prim
        self.tg = tg
    def __pos__(self): return self
    def __neg__(self): return 0-self
    def __add__(self, y): return plus(self, y)
    def __radd__(self, x): return plus(x, self)
    def __sub__(self, y): return minus(self, y)
    def __rsub__(self, x): return minus(x, self)
    def __mul__(self, y): return times(self, y)
    def __rmul__(self, x): return times(x, self)
    def __div__(self, y): return divide(self, y)
    def __rdiv__(self, x): return divide(x, self)
    def __truediv__(self, y): return divide(self, y)
    def __rtruediv__(self, x): return divide(x, self)
    def __lt__(self, x): return lt(self, x)
    def __le__(self, x): return le(self, x)
    def __gt__(self, x): return gt(self, x)
    def __ge__(self, x): return ge(self, x)
    def __eq__(self, x): return eq(self, x)
    def __ne__(self, x): return ne(self, x)

def bun(x, x_tangent): return bundle(x, x_tangent)

def prim(x):
    if isinstance(x, bundle): return x.prim
    else: return x

def tg(x):
    if isinstance(x, bundle): return x.tg
    else: return 0

def lift_real_to_real(f, dfdx):
    def me(x):
        if isinstance(x, bundle): return bun(me(prim(x)), dfdx(prim(x))*tg(x))
        else: return f(x)
    return me

def lift_real_cross_real_to_real(f, dfdx1, dfdx2):
    def me(x1, x2):
        if isinstance(x1, bundle) or isinstance(x2, bundle):
            return bun(me(prim(x1), prim(x2)),
                       (dfdx1(prim(x1), prim(x2))*tg(x1)
                        +dfdx2(prim(x1), prim(x2))*tg(x2)))
        else: return f(x1, x2)
    return me

def lift_real_cross_real_to_boolean(f):
    def me(x1, x2):
        if isinstance(x1, bundle): return me(prim(x1), x2)
        elif isinstance(x2, bundle): return me(x1, prim(x2))
        else: return f(x1, x2)
    return me

plus = lift_real_cross_real_to_real(lambda x1, x2: x1+x2,
                                    lambda x1, x2: 1,
                                    lambda x1, x2: 1)


minus = lift_real_cross_real_to_real(lambda x1, x2: x1-x2,
                                     lambda x1, x2: 1,
                                     lambda x1, x2: -1)

times = lift_real_cross_real_to_real(lambda x1, x2: x1*x2,
                                     lambda x1, x2: x2,
                                     lambda x1, x2: x1)

divide = lift_real_cross_real_to_real(lambda x1, x2: x1/x2,
                                      lambda x1, x2: 1/x2,
                                      lambda x1, x2: -x1/(x2*x2))

lt = lift_real_cross_real_to_boolean(lambda x1, x2: x1<x2)

le = lift_real_cross_real_to_boolean(lambda x1, x2: x1<=x2)

gt = lift_real_cross_real_to_boolean(lambda x1, x2: x1>x2)

ge = lift_real_cross_real_to_boolean(lambda x1, x2: x1>=x2)

eq = lift_real_cross_real_to_boolean(lambda x1, x2: x1==x2)

ne = lift_real_cross_real_to_boolean(lambda x1, x2: x1!=x2)

exp = lift_real_to_real(math.exp, lambda x: exp(x))

def derivative(f): return lambda x: tg(f(bun(x, 1)))

def replace_ith(x, i, xi):
    return [xi if j==i else x[j] for j in range(len(x))]

def partial_derivative(f, i):
    return lambda x: derivative(lambda xi: f(replace_ith(x, i, xi)))(x[i])

def gradient(f):
    return lambda x: [partial_derivative(f, i)(x) for i in range(len(x))]
