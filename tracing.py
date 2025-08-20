count = 0

class leaf:
    def __init__(self, x): self.x = x
    def __add__(self, y): return plus(self, y)
    def __radd__(self, x): return plus(x, self)
    def __mul__(self, y): return times(self, y)
    def __rmul__(self, x): return times(x, self)
    def __str__(self): return str(self.x)
    def eval(self): return self.x
    def trace(self):
        global count
        var = "t"+str(count)
        count += 1
        return var, [var+" = "+str(self.x)]

class vertex:
    def __init__(self, operator, x1, x2):
        self.operator = operator
        self.x1 = x1
        self.x2 = x2
    def __add__(self, y): return plus(self, y)
    def __radd__(self, x): return plus(x, self)
    def __mul__(self, y): return times(self, y)
    def __rmul__(self, x): return times(x, self)
    def __str__(self):
        return "("+str(self.operator)+", "+str(self.x1)+", "+str(self.x2)+")"
    def eval(self):
        if self.operator=="+": return self.x1.eval()+self.x2.eval()
        elif self.operator=="*": return self.x1.eval()*self.x2.eval()
        else: raise RuntimeError("invalid operator")
    def trace(self):
        (var1, code1) = self.x1.trace()
        (var2, code2) = self.x2.trace()
        global count
        var = "t"+str(count)
        count += 1
        return var, code1+code2+[var+" = "+var1+self.operator+var2]

def plus(x1, x2): return vertex("+", x1, x2)

def times(x1, x2): return vertex("*", x1, x2)

def print_code(v):
    var, code = v.trace()
    for statement in code: print(statement)
    print("return "+var)

def f(x, y): return leaf(6)*x*x+leaf(3)*y
