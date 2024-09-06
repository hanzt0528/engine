#使用torchvision.models中的模型，和预训练的权值

import torch
from torchvision.models import resnet152, ResNet152_Weights


# Step 1: Initialize model with the best available weights
weights = ResNet152_Weights.IMAGENET1K_V2
model = resnet152(weights=weights)


print(model)

torch.save(model.state_dict(),"resnet152.state_dict")
print("save model to vgg.state_dict Done.")