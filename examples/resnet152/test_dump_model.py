import torch
from model import ResNet50
import time
from torchviz import make_dot

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

if torch.cuda.is_available():
    model.to('cuda')
    
make_dot(model, (torch.zeros(1, 3, 224, 224).to('cuda')))  # 创建图形
