class cobundle:
    def __init__(self, prim, tape):
        self.prim = prim
        self.tape = tape
    def __add__(self, y):
        return plus(self, y)
    def __radd__(self, x):
        return plus(x, self)
    def __mul__(self, y):
        return times(self, y)
    def __rmul__(self, x):
        return times(x, self)

class tape:
    def __init__(self, factors, tapes, fanout, cotg):
        self.factors = factors
        self.tapes = tapes
        self.fanout = fanout
        self.cotg = cotg

def cobun(x, factors, tapes):
    return cobundle(x, tape(factors, tapes, 0, 0))

def variable(x):
    return cobun(x, [], [])

def determine_fanout(tape):
    tape.fanout += 1
    if tape.fanout==1:
        for tape in tape.tapes:
            determine_fanout(tape)

def initialize_cotg(tape):
    tape.cotg = 0
    tape.fanout -= 1
    if tape.fanout==0:
        for tape in tape.tapes:
            initialize_cotg(tape)

def reverse_sweep(cotg, tape):
    tape.cotg = tape.cotg+cotg
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
        return x.tape.cotg

def lift_real_to_real(f, dfdx):
    def me(x):
        if isinstance(x, cobundle):
            return cobun(me(x.prim), [dfdx(x.prim)], [x.tape])
        else:
            return f(x)
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
                return new_cobun(
                    me(x1, x2.prim), [dfdx2(x1, x2.prim)], [x2.tape])
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
        x_reverse = variable(x)
        return cotg(f(x_reverse), x_reverse)
    return me
