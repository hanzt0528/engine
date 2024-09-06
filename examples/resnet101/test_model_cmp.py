import torch
from model import ResNet101
import time
from torchvision.models import resnet101, ResNet101_Weights

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'


#model = model.to(device)
# 加载预训练的state_dict
state_dict = torch.load('/data/hanzt1/he/codes/engine/examples/resnet101/resnet101.state_dict')
#print(state_dict)
# 更新模型参数


# 使用自己定义的模型,在model.py文件中定义
model1 = ResNet101()
model = resnet101()
model1.load_state_dict(state_dict)
model.load_state_dict(state_dict)
# 比较模型参数
for param_a, param_b in zip(model1.parameters(), model.parameters()):
    if torch.equal(param_a, param_b):
        print("相同的权重:", param_a)
    else:
        print("不同的权重:", param_a, param_b)
        
