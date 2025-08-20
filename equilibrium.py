from nestable_forward_mode import *

def root(f, x0, n):
    x = x0
    for i in range(n): x = x-f(x)/derivative(f)(x)
    return x

def argmax(f, x0, n): return root(derivative(f), x0, n)

def gameA(a, b):
    price = 20-0.1*a-0.1*b
    costs = a*(10-0.05*a)
    return a*price-costs

def gameB(a, b):
    price = 20-0.1*b-0.0999*a
    costs = b*(10.005-0.05*b)
    return b*price-costs

def equilibrium(A, B, a0, b0, n):
    def f(aprime):
        def g(a):
            def h(b): return B(aprime, b)
            return A(a, argmax(h, b0, n))
        return argmax(g, aprime, n)-aprime
    astar = root(f, a0, n)
    def h(b): return B(astar, b)
    bstar = argmax(h, b0, n)
    return astar, bstar

print(equilibrium(gameA, gameB, 0.0, 0.0, 10))
