import torch
# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)

model.eval()
input_tensor = torch.randn(1, 3, 224, 224)

torch.onnx.export(model,               # 模型
                  input_tensor,              # 输入张量
                  "resnet18.onnx",           # 输出文件名
                  export_params=True,        # 是否导出参数
                  opset_version=11,          # ONNX的操作集版本
                  do_constant_folding=True,  # 是否执行常量折叠优化
                  input_names=['input'],     # 输入张量的名称
                  output_names=['output']   # 输出张量的名称
                  )  # 可变长度轴
