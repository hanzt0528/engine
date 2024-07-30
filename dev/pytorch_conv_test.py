import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个卷积层，其中卷积核的权重被初始化为1
class OnesConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(OnesConv2d, self).__init__()
        # 创建一个权重张量，其值全部为1
        weight = torch.ones((out_channels, in_channels, kernel_size, kernel_size), dtype=torch.float32)*2.5
        
        print("kernel shape:")
        print(weight.shape)
        

        print(weight)
        
        
        # 创建卷积层，使用自定义的权重
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.conv.weight.data = weight

    def forward(self, x):
        return self.conv(x)

# 创建模型实例
model = OnesConv2d(in_channels=10, out_channels=10,kernel_size=3,stride=1,padding=1)

# 创建一个9x9的输入张量，其所有值都被初始化为1
input_tensor = torch.ones(1, 10, 6, 8, dtype=torch.float32)*1.5

# 应用卷积操作
output = model(input_tensor)

# 打印输出结果
print("Convolution output:\n", output)
print("Convolution output.shape:\n", output.shape)