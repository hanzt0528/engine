import torch
import torch.nn as nn

# 创建卷积层，使用偏置项
conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)


# 创建一个随机的输入张量，例如一个单通道28x28图像
input_tensor = torch.ones(1, 1, 28, 28)

conv_layer.weight.data.fill_(1.0)

print("weight:")
print(conv_layer.weight)

conv_layer.bias.data.fill_(0.0)
# 前向传播，得到卷积层的输出
output_tensor = conv_layer(input_tensor)

print("result:")
print(output_tensor)