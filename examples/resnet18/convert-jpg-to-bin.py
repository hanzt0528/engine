import torch
import urllib
from PIL import Image
from torchvision import transforms
import struct
import sys
import numpy as np
model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)

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
data = input_tensor.squeeze().numpy()
 
fout = open("dog.bin", "wb")
data = data.astype(np.float32)
# data
# print("data:")
# print(data)

data.tofile(fout)
fout.close()
