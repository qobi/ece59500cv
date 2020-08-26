six = 6
five = 5
seven = 7

def f(x, y):
    return six*x*y+five*x*x+seven*y*y

def f(x): return 6*x*x
f = lambda x: 6*x*x

def f(a):
    (x, y) = a
    return 6*x*y

# not valid
f = lambda a:
        (x, y) = a
        return 6*x*y

lambda a:
  (x, y) = a
  return lambda b:
             (u, v) = b
             return x+y+u+v

def f(a):
    (x, y) = a
    def g(b):
        (u, v) = b
        return x+y+u+v
    return g
