import math

class cobundle:
    def __init__(self, prim, tape):
        self.prim = prim
        self.tape = tape
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

class tape:
    def __init__(self, factors, tapes, fanout, cotg):
        self.factors = factors
        self.tapes = tapes
        self.fanout = fanout
        self.cotg = cotg

def cobun(x, factors, tapes): return cobundle(x, tape(factors, tapes, 0, 0))

def variable(x): return cobun(x, [], [])

def determine_fanout(tape):
    tape.fanout += 1
    if tape.fanout==1:
        for tape in tape.tapes: determine_fanout(tape)

def initialize_cotg(tape):
    tape.cotg = 0
    tape.fanout -= 1
    if tape.fanout==0:
        for tape in tape.tapes: initialize_cotg(tape)

def reverse_sweep(cotg, tape):
    tape.cotg += cotg
    tape.fanout -= 1
    if tape.fanout==0:
        cotg = tape.cotg
        for factor, tape in zip(tape.factors, tape.tapes):
            reverse_sweep(cotg*factor, tape)

def cotg(y, x):
    if isinstance(y, cobundle):
        determine_fanout(y.tape)
        initialize_cotg(y.tape)
        determine_fanout(y.tape)
        reverse_sweep(1, y.tape)
        return cotg(y.prim, x)
    else:
        if isinstance(x, list): return [xi.tape.cotg for xi in x]
        else: return x.tape.cotg

def lift_real_to_real(f, dfdx):
    def me(x):
        if isinstance(x, cobundle):
            return cobun(me(x.prim), [dfdx(x.prim)], [x.tape])
        else: return f(x)
    return me

def lift_real_cross_real_to_real(f, dfdx1, dfdx2):
    def me(x1, x2):
        if isinstance(x1, cobundle):
            if isinstance(x2, cobundle):
                return cobun(me(x1.prim, x2.prim),
                             [dfdx1(x1.prim, x2.prim),
                              dfdx2(x1.prim, x2.prim)],
                             [x1.tape, x2.tape])
            else:
                return cobun(me(x1.prim, x2), [dfdx1(x1.prim, x2)], [x1.tape])
        else:
            if isinstance(x2, cobundle):
                return cobun(
                    me(x1, x2.prim), [dfdx2(x1, x2.prim)], [x2.tape])
            else: return f(x1, x2)
    return me

def lift_real_cross_real_to_boolean(f):
    def me(x1, x2):
        if isinstance(x1, cobundle): return me(x1.prim, x2)
        elif isinstance(x2, cobundle): return me(x1, x2.prim)
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

def derivative(f):
    def me(x):
        if isinstance(x, list): x_reverse = [variable(xi) for xi in x]
        else: x_reverse = variable(x)
        return cotg(f(x_reverse), x_reverse)
    return me

def partial_derivative(f, i): return lambda x: gradient(f)(x)[i]

gradient = derivative
