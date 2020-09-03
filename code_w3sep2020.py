# y=mx+b
# a.x+b=0
# a1*x1+x2*x2+...+an*xn+b=0

# a.x+b>=0
# a.x+b<0

def two_layer_perceptron(point, weights1, biases1, weights2, biases2):
    hidden = fc_layer(point, weights1, biases1)
    return fc_layer(hidden, weights2, biases2)

# weight2*(weights1*point+biases1)+biases2
# (weight2*weights1)*point+(weights1*biases1+biases2)
