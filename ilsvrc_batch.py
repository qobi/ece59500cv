# python3 -m venv venv
# venv/bin/pip3 install torch
# venv/bin/pip3 install torchvision

import torch
import os
import torchvision
import random

class alexnet(torch.nn.Module):
    def __init__(self, outputs):
        super(alexnet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.AdaptiveAvgPool2d((6, 6)))
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256*6*6, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True),
            torch.nn.Linear(4096, outputs))
    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))

class vgg19(torch.nn.Module):
    def __init__(self, outputs):
        super(vgg19, self).__init__()
        self.features = torch.nn.Sequential(
            # first block
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # second block
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # third block
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # fourth block
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # fifth block
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # tail
            torch.nn.AdaptiveAvgPool2d((7, 7)))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512*7*7, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, outputs))
    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))

classes =  [("dog", "n02106662"),
            ("cat", "n02124075"),
            ("elephant", "n02504458"),
            ("panda", "n02510455")]

directory = "/aux/qobi/Imagenet/ILSVRC2012_img_train"

training_set = [((torchvision.io.read_image(directory+"/"+
                                            class_id+"/"+
                                            filename)/255.0).cuda(),
                 torch.tensor(list(map(lambda c: c[0],
                                       classes)).index(class_name)).cuda())
                for class_name, class_id in classes
                for filename in os.listdir(directory+"/"+class_id)]

training_set = [(image, label)
                for image, label in training_set
                if image.size(0)==3]

net = alexnet(4).cuda()

training_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

optimizer = getattr(torch.optim, "SGD")(net.parameters(), lr = 1e-2)

criterion = torch.nn.CrossEntropyLoss().cuda()

batch_size = 30

net.train()
for epoch in range(20):
    random.shuffle(training_set)
    total_loss = 0
    for i in range(0, len(training_set), batch_size):
        inputs = torch.stack(
            [training_transforms(image)
             for image, label in training_set[i:i+batch_size]])
        labels = torch.stack(
            [label for image, label in training_set[i:i+batch_size]])
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.tolist()
    print("epoch %d, loss %g"%(epoch, total_loss))

net.eval()
total_loss = 0
for input, label in training_set:
    output = net(torch.stack([test_transforms(input)]))[0]
    if output.tolist().index(max(output.tolist()))==label.tolist():
        print("correct")
    else:
        print("incorrect")
    loss = criterion(output, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss += loss.tolist()
print("test, loss %g"%total_loss)
