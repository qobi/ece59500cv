class dual_number:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __pos__(self): return self
    def __add__(self, y): return plus(self, y)
    def __radd__(self, x): return plus(x, self)
    def __mul__(self, y): return times(self, y)
    def __rmul__(self, x): return times(x, self)
    def __repr__(self): return dual_number_to_str(self)

def primal(x):
    if isinstance(x, dual_number): return x.a
    else: return x

def tangent(x):
    if isinstance(x, dual_number): return x.b
    else: return 0

def lift(x):
    if isinstance(x, dual_number): return x
    else: return dual_number(x, 0)

def plus(x, y):
    if isinstance(x, dual_number):
        if isinstance(y, dual_number):
            a = primal(x)
            b = tangent(x)
            c = primal(y)
            d = tangent(y)
            return dual_number(a+c, b+d)
        else: return x+lift(y)
    else:
        if isinstance(y, dual_number): return lift(x)+y
        else: return x+y

def times(x, y):
    if isinstance(x, dual_number):
        if isinstance(y, dual_number):
            a = primal(x)
            b = tangent(x)
            c = primal(y)
            d = tangent(y)
            return dual_number(a*c, a*d+b*c)
        else: return x*lift(y)
    else:
        if isinstance(y, dual_number): return lift(x)*y
        else: return x*y

def dual_number_to_str(x):
    a = primal(x)
    b = tangent(x)
    if b>=0:
        return "%s+%s*e"%(str(a), str(b))
    else:
        return "%s-%s*e"%(str(a), str(-b))

e = dual_number(0, 1)

# def derivative(f, x):
#     return tangent(f(x+e))

# def f(x): return 3*x*x+4*x+2

# derivative(f, 3)

def derivative(f):
    return lambda x: tangent(f(x+1*e))

# derivative(f)
# derivative(f)(3)

def replace_ith(x, i, xi):
    return [xi if j==i else x[j] for j in range(len(x))]

def partial_derivative(f, i):
    return lambda x: derivative(lambda xi: f(replace_ith(x, i, xi)))(x[i])

def gradient(f):
    return lambda x: [partial_derivative(f, i)(x) for i in range(len(x))]

# def f(x):
#     u, v = x
#     return 3*u*u+4*u*v+6*v*v+7*u+8*v+5

# gradient(f)([3, 4])
# dfdu = lambda u, v: 6*u+4*v+7
# dfdu(3, 4)
# dfdv = lambda u, v: 4*u+12*v+8
# dfdv(3, 4)

def root(f, x0, n):
    x = x0
    for _ in range(n):
        x = x-f(x)/(derivative(f)(x))
    return x

# def naive_gradient_descent(f, x0, n, eta):
#     x = x0
#     for _ in range(n):
#         x = x-eta*derivative(f)(x)
#     return x

def naive_gradient_descent(f, x0, n, eta):
    x = x0
    for _ in range(n):
        x = [xi-eta*dfdxi for xi, dfdxi in zip(x, gradient(f)(x))]
    return x

# def f(x):
#     u, v = x
#     return 3*u*u+6*v*v

# f([0, 0])
# f([2, 2])

# naive_gradient_descent(f, [2, 2], 10, 0.1)
# naive_gradient_descent(f, [2, 2], 100, 0.1)

# import torch
# def f(x): return 3*x*x+4*x+2
# torch.autograd.functional.jvp(f, torch.tensor(3.0))
