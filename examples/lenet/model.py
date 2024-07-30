from torch.nn import Module
from torch import nn


class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        
        #self.conv1.bias.data.fill_(0.0)
        y = self.conv1(x)
        
        print(f"conv1 result shape = {y.shape}")
        print("self.conv1.bias")
        
        print(self.conv1.bias)
        
        print("conv1 result")
        print(y)
        y = self.relu1(y)

        y = self.pool1(y)
        y = self.conv2(y)
        print(f"conv2 result shape = {y.shape}")
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        print(f"fc1 input shape = {y.shape}")
        y = self.fc1(y)
        print(f"fc1  result = {y.shape}")
        y = self.relu3(y)
        y = self.fc2(y)
        print(f"fc2  result = {y.shape}")
        y = self.relu4(y)
        y = self.fc3(y)
        print(f"fc3 result = {y.shape}")
        y = self.relu5(y)
        return y
