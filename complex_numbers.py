class complex_number:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __pos__(self): return self
    def __add__(self, y): return plus(self, y)
    def __radd__(self, x): return plus(x, self)
    def __mul__(self, y): return times(self, y)
    def __rmul__(self, x): return times(x, self)
    def __repr__(self): return complex_number_to_str(self)

def real_part(x):
    if isinstance(x, complex_number): return x.a
    else: return x

def imaginary_part(x):
    if isinstance(x, complex_number): return x.b
    else: return 0

def lift(x):
    if isinstance(x, complex_number): return x
    else: return complex_number(x, 0)

def plus(x, y):
    if isinstance(x, complex_number):
        if isinstance(y, complex_number):
            a = real_part(x)
            b = imaginary_part(x)
            c = real_part(y)
            d = imaginary_part(y)
            return complex_number(a+c, b+d)
        else: return x+lift(y)
    else:
        if isinstance(y, complex_number): return lift(x)+y
        else: return x+y

def times(x, y):
    if isinstance(x, complex_number):
        if isinstance(y, complex_number):
            a = real_part(x)
            b = imaginary_part(x)
            c = real_part(y)
            d = imaginary_part(y)
            return complex_number(a*c-b*d, a*d+b*c)
        else: return x*lift(y)
    else:
        if isinstance(y, complex_number): return lift(x)*y
        else: return x*y

def complex_number_to_str(x):
    a = real_part(x)
    b = imaginary_part(x)
    if b>=0:
        return "%s+%s*i"%(str(a), str(b))
    else:
        return "%s-%s*i"%(str(a), str(-b))

i = complex_number(0, 1)
