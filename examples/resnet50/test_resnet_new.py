#使用torchvision.models中的模型，和预训练的权值

import torch
from torchvision.models import resnet50, ResNet50_Weights

from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights

img = read_image("dog.jpg")

# Step 1: Initialize model with the best available weights
weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights)
model.eval()

print(model)
# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
print(f'score = {score}')
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")