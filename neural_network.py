def neuron():
    # inputs: x                 x_reverse
    # output: y                 y_reverse
    # weights: w                w_reverse
    # intermediates: t1, t2     t1_reverse, t2_reverse

    # forward sweep
    for i in range(len(x)):
        t1[i] = w[i]*x[i]
    t2 = 0
    for i in range(len(t1)):
        t2 = t2+t1[i]
    y = sigmoid(t2)

    # initialize reverse variables
    y_reverse = 0
    t2_reverse = 0
    for i in range(len(t1)):
        t1_reverse[i] = 0
    for i in range(len(w)):
        w_reverse[i] = 0
    for i in range(len(x)):
        x_reverse[i] = 0

    # reverse sweep
    y_reverse += 1
    t2_reverse += y_reverse*sigmoid_derivative(t2)
    for i in range(len(t1)):
        t1_reverse[i] += t2_reverse
    for i in range(len(w)):
        w_reverse[i] += t1_reverse[i]*x[i]
    for i in range(len(x)):
        x_reverse[i] += t1_reverse[i]*w[i]

def slp():
    # inputs: x                 x_reverse
    # output: y                 y_reverse
    # weights: w                w_reverse
    # intermediates: t1, t2     t1_reverse, t2_reverse

    # forward sweep
    for j in range(len(y)):
        for i in range(len(x)):
            t1[j, i] = w[j, i]*x[i]
    for j in range(len(y)):
        t2[j] = 0
    for j in range(len(y)):
        for i in range(len(t1[j])):
            t2[j] = t2[j]+t1[j, i]
    for j in range(len(y)):
        y[j] = sigmoid(t2[j])

    # initialize reverse variables
    for j in range(len(y)):
        y_reverse[j] = 0
    for j in range(len(y)):
        t2_reverse[j] = 0
    for j in range(len(y)):
        for i in range(len(t1[j])):
            t1_reverse[j, i] = 0
    for j in range(len(y)):
        for i in range(len(w[j])):
            w_reverse[j, i] = 0
    for i in range(len(x)):
        x_reverse[i] = 0

    # reverse sweep
    for j in range(len(y)):
        y_reverse[j] += 1
    for j in range(len(y)):
        t2_reverse[j] += y_reverse[j]*sigmoid_derivative(t2[j])
    for j in range(len(y)):
        for i in range(len(t1[j])):
            t1_reverse[j, i] += t2_reverse[j]
    for j in range(len(y)):
        for i in range(len(w[j])):
            w_reverse[j, i] += t1_reverse[j, i]*x[i]
    for j in range(len(y)):
        for i in range(len(x)):
            x_reverse[i] += t1_reverse[j, i]*w[j, i]

def mlp():
    # inputs: x                 x_reverse
    # output: y                 y_reverse
    # weights: w                w_reverse
    # intermediates: t1, t2     t1_reverse, t2_reverse

    # forward sweep
    for k in range(layers):
        if k>0:
            for j in range(len(y[k])):
                x[k, j] = y[k-1, j]
        for j in range(len(y[k])):
            for i in range(len(x[k])):
                t1[k, j, i] = w[k, j, i]*x[k, i]
        for j in range(len(y[k])):
            t2[k, j] = 0
        for j in range(len(y[k])):
            for i in range(len(t1[k, j])):
                t2[k, j] = t2[k, j]+t1[k, j, i]
        for j in range(len(y[k])):
            y[k, j] = sigmoid(t2[k, j])

    # initialize reverse variables
    for k in range(layers-1, -1, -1):
        for j in range(len(y[k])):
            y_reverse[k, j] = 0
        for j in range(len(y[k])):
            t2_reverse[k, j] = 0
        for j in range(len(y[k])):
            for i in range(len(t1[k, j])):
                t1_reverse[k, j, i] = 0
        for j in range(len(y[k])):
            for i in range(len(w[k, j])):
                w_reverse[k, j, i] = 0
        for i in range(len(x[k])):
            x_reverse[k, i] = 0

    # reverse sweep
    for k in range(layers-1, -1, -1):
        if k<layers-1:
            for j in range(len(y[k])):
                y_reverse[k, j] += x_reverse[k+1, j]
        else:
            for j in range(len(y[k])):
                y_reverse[k, j] += 1
        for j in range(len(y[k])):
            t2_reverse[k, j] += y_reverse[k, j]*sigmoid_derivative(t2[k, j])
        for j in range(len(y[k])):
            for i in range(len(t1[k, j])):
                t1_reverse[k, j, i] += t2_reverse[k, j]
        for j in range(len(y[k])):
            for i in range(len(w[k, j])):
                w_reverse[k, j, i] += t1_reverse[k, j, i]*x[k, i]
        for j in range(len(y[k])):
            for i in range(len(x[k])):
                x_reverse[k, i] += t1_reverse[k, j, i]*w[k, j, i]
