import torch

def derivative(f):
    def me(x):
        x_tensor = torch.tensor(float(x), requires_grad = True)
        y_tensor = f(x_tensor)
        y_tensor.backward()
        return x_tensor.grad.tolist()
    return me

def f(x): return 3*x*x

print derivative(f)(3.0)

#print derivative(derivative(f))(3.0)

# Siskind & Pearlmutter (IFL 2005) Equation (2)
print derivative(lambda x: x*derivative(lambda y: x+y)(1))(1)

# Siskind & Pearlmutter (HOSC 2008) pages 363-364
print derivative(lambda x: x*derivative(lambda y: x*y)(2))(1)
