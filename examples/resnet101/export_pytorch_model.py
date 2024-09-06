#使用torchvision.models中的模型，和预训练的权值

import torch
from torchvision.models import resnet101, ResNet101_Weights


# Step 1: Initialize model with the best available weights
weights = ResNet101_Weights.IMAGENET1K_V2
model = resnet101(weights=weights)


print(model)

torch.save(model.state_dict(),"resnet101.state_dict")
print("save model to vgg.state_dict Done.")