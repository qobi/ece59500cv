import torch

class two_layer_perceptron(torch.nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super(two_layer_perceptron, self).__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(inputs, hidden),
                                          torch.nn.Sigmoid(),
                                          torch.nn.Linear(hidden, outputs),
                                          torch.nn.Sigmoid())
    def forward(self, x):
        return self.layers(x)

xor_training_set = [([0, 0], [0]),
                    ([0, 1], [1]),
                    ([1, 0], [1]),
                    ([1, 1], [0])]

one_bit_adder_training_set = [([0, 0, 0], [0, 0]),
                              ([0, 0, 1], [0, 1]),
                              ([0, 1, 0], [0, 1]),
                              ([0, 1, 1], [1, 0]),
                              ([1, 0, 0], [0, 1]),
                              ([1, 0, 1], [1, 0]),
                              ([1, 1, 0], [1, 0]),
                              ([1, 1, 1], [1, 1])]

training_set = [(torch.tensor(list(map(float, input))),
                 torch.tensor(list(map(float, label))))
                for input, label in xor_training_set]

net = two_layer_perceptron(2, 4, 1)

optimizer = getattr(torch.optim, "SGD")(net.parameters(), lr = 1e-1)

for epoch in range(100000):
    total_loss = 0
    for input, label in training_set:
        output = net(input)
        loss = torch.nn.functional.mse_loss(output, label, reduction="mean")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.tolist()
    if epoch%1000==0:
        print("epoch %d, loss %g"%(epoch, total_loss))
