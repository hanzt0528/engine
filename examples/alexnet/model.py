
import torch
from torch.nn import Module
from torch import nn

__all__ = ["AlexNet", "AlexNet_Weights", "alexnet"]


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
     
        print("x.shape=")
        print(x.shape)
        #print(x)
        x = self.features(x)
        #print(x)
        x = self.avgpool(x)
        print("avgpool.shape=")
        print(x.shape)
        x = torch.flatten(x, 1)
        print("flattened:")
        print(x.shape)
        x = self.classifier(x)
        return x
    def load_state_dict(self, state_dict):
        # 可能的自定义加载逻辑
        super(AlexNet, self).load_state_dict(state_dict)  # 确保调用父类