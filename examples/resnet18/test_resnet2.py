import torch
from model import ResNet18

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
# 创建AlexNet模型
model = ResNet18()
#model = model.to(device)
# 加载预训练的state_dict
state_dict = torch.load('/data/hanzt1/he/codes/engine/examples/resnet/resnet18.state_dict')
#print(state_dict)
# 更新模型参数
model.load_state_dict(state_dict)
model.eval()
print(model)
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "/data/hanzt1/he/codes/engine/examples/resnet/dog.jpg")
#try: urllib.URLopener().retrieve(url, filename)
#except: urllib.request.urlretrieve(url, filename)

# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)

# input = input_tensor.flatten()
# input = input.tolist()

# with open('input.txt', 'w') as file:
#     # 遍历数组并将每个元素写入文件
#     for num in input:
#         file.write(f"{num}\n")
                
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

#print(f'input_batch = {input_batch}')
# move the input and model to GPU for speed if available
# if torch.cuda.is_available():
#     input_batch = input_batch.to('cuda')
#     model.to('cuda')

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