import torch
import torch.nn as nn

# 假设我们有一个特征维度为3的模型
num_features = 3

# 创建BatchNorm2d层
batch_norm = nn.BatchNorm2d(num_features=num_features)

# 创建一些模拟数据，例如一个3x3x3的张量，代表3个通道，3个批次，每个批次3个数据点
data = torch.randn(3, num_features, 3, 3)

print("input:")
print(data)
# 将数据通过BatchNorm2d层
output = batch_norm(data)
print("BN:")
print(output)