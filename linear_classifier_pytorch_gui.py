from gui import *
import torch

points = []
labels = []
model = None
optimizer = None

class classifier(torch.nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.fc = torch.nn.Linear(2, 2)
    def forward(self, x):
        x = self.fc(x)
        return x

def loss(points, labels, model):
    input = torch.tensor(points, dtype = torch.float)
    target = torch.tensor(labels, dtype = torch.long)
    output = model(input)
    loss = torch.nn.functional.cross_entropy(output, target)
    return loss.tolist()

def all_labels(labels):
    red = False
    blue = False
    for label in labels:
        if label==0: red = True
        else: blue = True
    return red and blue

def initialize():
    global model, optimizer
    model = classifier()
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-1)

def step():
    model.train()
    input = torch.tensor(points, dtype = torch.float, requires_grad = True)
    target = torch.tensor(labels, dtype = torch.long)
    output = model(input)
    loss = torch.nn.functional.cross_entropy(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train():
    initialize()
    for i in range(100): step()

def classify(point, model):
    model.eval()
    input = torch.tensor([point], dtype = torch.float)
    output = model(input)
    _, prediction = output.data.max(1)
    return prediction[0]

def redisplay():
    get_axes().clear()
    for i in range(len(points)):
        if labels[i]==0: get_axes().plot([points[i][0]], [points[i][1]], "r+")
        else: get_axes().plot([points[i][0]], [points[i][1]], "b+")
    redraw()

def clear_command():
    global points, labels, model, optimizer
    points = []
    labels = []
    model = None
    optimizer = None
    message("")
    get_axes().clear()
    redraw()

def initialize_command():
    def internal():
        initialize()
        message("{:.3f}".format(loss(points, labels, model)))
        redisplay()
    if not all_labels(labels): message("Missing class")
    else:
        message("Training")
        get_window().after(10, internal)

def step_command():
    def internal():
        step()
        message("{:.3f}".format(loss(points, labels, model)))
        redisplay()
    if not all_labels(labels): message("Missing class")
    else:
        message("Training")
        get_window().after(10, internal)

def train_command():
    def internal():
        train()
        message("{:.3f}".format(loss(points, labels, model)))
        redisplay()
    if not all_labels(labels): message("Missing class")
    else:
        message("Training")
        get_window().after(10, internal)

def all_command():
    resolution = 50
    scale = 1.0/resolution
    for y in range(resolution+1):
        for x in range(resolution+1):
            label = classify([scale*x, scale*y], model)
            if label==0: get_axes().plot([scale*x], [scale*y], "r.")
            else: get_axes().plot([scale*x], [scale*y], "b.")
    redraw()

def click(x, y):
    message("")
    if mode()==0:
        points.append([x, y])
        labels.append(0)
        get_axes().plot([x], [y], "r+")
        redraw()
    elif mode()==1:
        points.append([x, y])
        labels.append(1)
        get_axes().plot([x], [y], "b+")
        redraw()
    else:
        if model is None: message("Train first")
        else:
            label = classify([x, y], model)
            if label==0: message("Red")
            else: message("Blue")

add_button(0, 0, "Clear", clear_command, nothing)
mode = add_radio_button_group([[0, 1, "Red", 0],
                               [0, 2, "Blue", 1],
                               [0, 3, "Classify", 2]],
                              lambda: False)
add_button(0, 4, "Initialize", initialize_command, nothing)
add_button(0, 5, "Step", step_command, nothing)
add_button(0, 6, "Train", train_command, nothing)
add_button(0, 7, "All", all_command, nothing)
add_button(0, 8, "Exit", done, nothing)
message = add_message(1, 0, 9)
add_click(click)
start_fixed_size_matplotlib(7, 7, 2, 9)
