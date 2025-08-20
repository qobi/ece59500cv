import torch

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
        return self.classifier(torch.flatten(self.features(x), 0))
