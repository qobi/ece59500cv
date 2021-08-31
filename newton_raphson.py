from forward_mode import *

def root(f, x0):
    x = x0
    for i in range(100):
        x = x-f(x)/derivative(f)(x)
    return x
