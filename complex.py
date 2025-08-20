class complx:
    def __init__(self, a, b):
        self.a = a
        self.b = b
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
    def __eq__(self, x): return eq(self, x)
    def __ne__(self, x): return not eq(self, x)
    def __repr__(self): return complx_to_str(self)

def real_part(x):
    if isinstance(x, complx): return x.a
    else: return x

def imag_part(x):
    if isinstance(x, complx): return x.b
    else: return 0

def lift(x):
    if isinstance(x, complx): return x
    else: return complx(x, 0)

def plus(x, y):
    if isinstance(x, complx):
        if isinstance(y, complx):
            a = real_part(x)
            b = imag_part(x)
            c = real_part(y)
            d = imag_part(y)
            return complx(a+c, b+d)
        else: return x+lift(y)
    else:
        if isinstance(y, complx): return lift(x)+y
        else: return x+y

def minus(x, y):
    if isinstance(x, complx):
        if isinstance(y, complx):
            a = real_part(x)
            b = imag_part(x)
            c = real_part(y)
            d = imag_part(y)
            return complx(a-c, b-d)
        else: return x-lift(y)
    else:
        if isinstance(y, complx): return lift(x)-y
        else: return x-y

def times(x, y):
    if isinstance(x, complx):
        if isinstance(y, complx):
            a = real_part(x)
            b = imag_part(x)
            c = real_part(y)
            d = imag_part(y)
            return complx(a*c-b*d, a*d+b*c)
        else: return x*lift(y)
    else:
        if isinstance(y, complx): return lift(x)*y
        else: return x*y

# (a+bi)/(c+di)
# =((a+bi)(c-di))/((c+di)(c-di))
# =((a+bi)(c-di))/(c^2-d^2)
# =((ac+bd)/(c^2-d^2))+((bc-ad)/(c^2-d^2))i

def divide(x, y):
    if isinstance(x, complx):
        if isinstance(y, complx):
            a = real_part(x)
            b = imag_part(x)
            c = real_part(y)
            d = imag_part(y)
            return complx((a*c+b*d)/(c*c-d*d), (b*c-a*d)/(c*c-d*d))
        else: return x/lift(y)
    else:
        if isinstance(y, complx): return lift(x)/y
        else: return x/y

def eq(x, y):
    if isinstance(x, complx):
        if isinstance(y, complx):
            a = real_part(x)
            b = imag_part(x)
            c = real_part(y)
            d = imag_part(y)
            return a==c and b==d
        else: return x==lift(y)
    else:
        if isinstance(y, complx): return lift(x)==y
        else: return x==y

def complx_to_str(x):
    return "complx(%s, %s)"%(str(real_part(x)), str(imag_part(x)))
