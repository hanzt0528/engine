import torch
import urllib
from PIL import Image
from torchvision import transforms
import struct
import sys
import numpy as np
from model import AlexNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 创建AlexNet模型
model = AlexNet()
#model = model.to(device)
# 加载预训练的state_dict
state_dict = torch.load('alexnet.state_dict')
#print(state_dict)
# 更新模型参数
model.load_state_dict(state_dict)

# 将模型设置为评估模式
print(model)


model.eval()
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "./dog.jpg")
# try: urllib.URLopener().retrieve(url, filename)
# except: urllib.request.urlretrieve(url, filename)

input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)

input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model




# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')



with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
#print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
#print(probabilities)

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
    
    

