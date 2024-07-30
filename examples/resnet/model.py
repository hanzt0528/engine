import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
def conv3x3(in_planes,out_planes,stride=1):
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,padding=1,bias=False)


class CustomBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(CustomBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # 可学习的参数 gamma 和 beta
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        # 运行均值和方差
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.training = False

    def forward(self, x):
        if self.training:
            # 计算均值和方差
            mean = x.mean([0, 2, 3], keepdim=True)
            var = x.var([0, 2, 3], keepdim=True, unbiased=False)
        
            # 更新运行均值和方差
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            # 使用运行均值和方差
            mean, var = self.running_mean, self.running_var
        
        # 批量归一化
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        # 缩放和平移
        out = self.gamma.view(1, -1, 1, 1) * x_normalized + self.beta.view(1, -1, 1, 1)
        
        return out
    
class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes,stride)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.bn1=CustomBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
 
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
 
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)


        out = self.bn2(out)
        print(f'bn2 out shape = {out.shape}')
        out += self.downsample(x)
        out = self.relu(out)
        return out
 
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        print(self.bn1)
        #self.bn1 = CustomBatchNorm2d(64)
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion, num_classes)
 
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
 
    def forward(self, x):
        out = self.conv1(x)
        print(f'conv1 = {out}')
        print(f'bn1.mean = {self.bn1.running_mean}')
        mean = self.bn1.running_mean.view(1, self.bn1.num_features, 1, 1)
        
        sub = x
        print(f'out - mean = {sub}')
        
     
        sub = sub.flatten()
        sub = sub.tolist()
        
         
        with open('output-x.txt', 'w') as file:
            # 遍历数组并将每个元素写入文件
            for num in sub:
                file.write(f"{num}\n")
        
        var = self.bn1.running_var.view(1, self.bn1.num_features, 1, 1)
            
        x_normalized = (out - mean) / torch.sqrt(var + 0.000001)
        
        
        out = self.bn1.weight.view(1, self.bn1.num_features, 1, 1) * x_normalized + self.bn1.bias.view(1, self.bn1.num_features, 1, 1)
        
        print(f'bn1 out = {out}')
        #out = self.bn1(out)
        
      
        print(f'bn1 = {out}')
        out = self.relu(out)
        out = self.maxpool(out)
        print(f"maxpool shape = {out.shape}")
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        print(f" 4 layer shape = {out.shape}")
        out = self.avgpool(out)
        print(f" avgpool shape = {out.shape}")
        out = out.view(out.size(0), -1)
        print(f" viewed shape = {out.shape}")
        out = self.fc(out)
        return out
 
# 使用BasicBlock定义ResNet18
def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2],1000)
 
# 实例化模型
model = ResNet18()
print(model)