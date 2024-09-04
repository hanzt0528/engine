import torch
import torch.nn as nn
import torch.nn.functional as F

       
class BasicBlock(nn.Module):
    expansion = 4

    def __init__(self, first_channels,second_channels ,third_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(first_channels ,second_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(second_channels)
        
        self.conv2 = nn.Conv2d(second_channels, second_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(second_channels)
        

        self.conv3 = nn.Conv2d(second_channels,third_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(third_channels)
        

        self.relu = nn.ReLU(inplace=True)
                 
        # if downsample == None:
        #     self.downsample = nn.Sequential()
        # else:
        self.downsample = downsample
        self.index = 0

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        print(f'out shape = {out.shape}')


              
                
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
 
        # dump_data = out

        
        # dump_data = dump_data.flatten()
        # dump_data = dump_data.tolist()
        
         
        # with open('/data/hanzt1/he/codes/engine/examples/resnet50/output-model-layer2-block0-bn3.txt', 'w') as file:
        #     # 遍历数组并将每个元素写入文件
        #     for num in dump_data:
        #         file.write(f"{num}\n")
 
        if self.downsample is not None:
            identity = self.downsample(x)
        

                
        out+=identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 256, layers[0])
        self.layer2 = self._make_layer(block, 512, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 1024, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 2048, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        in_channels = int(out_channels/4)

        downsample = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels , kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels ),
        )

        layers = []
        layers.append(block( self.in_channels,in_channels,out_channels, stride, downsample))
        
        for _ in range(1, blocks):
            layers.append(block(out_channels,in_channels,out_channels))
        self.in_channels = out_channels
        return nn.Sequential(*layers)
    

    def forward(self, x):
        out = self.conv1(x)
        

        
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        



        out = self.layer1(out)
        


                
        out = self.layer2(out)
        
        out = self.layer3(out)
        out = self.layer4(out)

        # dump_data = out

        
        # dump_data = dump_data.flatten()
        # dump_data = dump_data.tolist()
        
         
        # with open('/data/hanzt1/he/codes/engine/examples/resnet50/output-model-layer4.txt', 'w') as file:
        #     # 遍历数组并将每个元素写入文件
        #     for num in dump_data:
        #         file.write(f"{num}\n")
        # return out
      


        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
    
    
 # 使用BasicBlock定义ResNet50
def ResNet50():
    return ResNet(BasicBlock,[3,4,6,3],1000)


print(ResNet50())