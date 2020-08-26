def f(a):
    (x, y) = a
    def g(b):
        (u, v) = b
        return x+y+u+v
    return g

def my_sum(l):
    s = 0
    for x in l:
        s = s + x
    return s

def my_product(l):
    s = 1
    for x in l:
        s = s * x
    return s

def my_g(g, l, id):
    s = id
    for x in l:
        s = g(s, x)
    return s

def f_list(f):
    def g(l):
        return [f(x) for x in l]
    return g

f_list1 = lambda f: lambda l: [f(x) for x in l]

sqr = lambda x: x*x
