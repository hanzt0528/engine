import torch

import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.io import read_image

from torch.autograd import profiler
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的ResNet50模型
model = models.resnet50(pretrained=True)
model = model.to(device)

# 将模型设置为评估模式
model.eval()

# 定义图片的预处理方式
preprocess = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize(256),
    transforms.CenterCrop(224),
#    transforms.ToTensor(),
    
])

# 读取图片，这里需要替换为你的图片路径
img_path = 'kitten.jpg'  # 替换为你的图片路径
img = read_image(img_path)
print(type(img))
# img = np.array(img)
# # 预处理图片
img = img.float()

img_t = preprocess(img).unsqueeze(0)  # 增加一个批次维度



# 将图片数据移动到设备上（GPU或CPU）
img_t = img_t.to(device)

with profiler.profile(use_cuda=True) as prof:
    # 模拟你的模型或算子的执行
    output = model(img_t)

# 打印统计信息
print(prof)