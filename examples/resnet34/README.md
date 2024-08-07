https://pytorch.org/hub/pytorch_vision_resnet/

### ResNet34 网络结构
网络定义，见model.py
```
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential()
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential()
    )
    (2): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential()
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential()
    )
    (2): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential()
    )
    (3): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential()
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential()
    )
    (2): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential()
    )
    (3): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential()
    )
    (4): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential()
    )
    (5): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential()
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential()
    )
    (2): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential()
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=1000, bias=True)
)

```
### export resnet34 of pytorch  to state_dict
```
 python export_pytorch_model.py 
```

### Convert resnet to ggml
```
$python convert-resnet-to-ggml.py resnet18.state_dict
conv1.weight with shape: (64, 3, 7, 7)
bn1.weight with shape: (64,)
bn1.bias with shape: (64,)
bn1.running_mean with shape: (64,)
bn1.running_var with shape: (64,)
bn1.num_batches_tracked with shape: ()
layer1.0.conv1.weight with shape: (64, 64, 3, 3)
layer1.0.bn1.weight with shape: (64,)
layer1.0.bn1.bias with shape: (64,)
layer1.0.bn1.running_mean with shape: (64,)
layer1.0.bn1.running_var with shape: (64,)
layer1.0.bn1.num_batches_tracked with shape: ()
layer1.0.conv2.weight with shape: (64, 64, 3, 3)
layer1.0.bn2.weight with shape: (64,)
layer1.0.bn2.bias with shape: (64,)
layer1.0.bn2.running_mean with shape: (64,)
layer1.0.bn2.running_var with shape: (64,)
layer1.0.bn2.num_batches_tracked with shape: ()
layer1.1.conv1.weight with shape: (64, 64, 3, 3)
layer1.1.bn1.weight with shape: (64,)
layer1.1.bn1.bias with shape: (64,)
layer1.1.bn1.running_mean with shape: (64,)
layer1.1.bn1.running_var with shape: (64,)
layer1.1.bn1.num_batches_tracked with shape: ()
layer1.1.conv2.weight with shape: (64, 64, 3, 3)
layer1.1.bn2.weight with shape: (64,)
layer1.1.bn2.bias with shape: (64,)
layer1.1.bn2.running_mean with shape: (64,)
layer1.1.bn2.running_var with shape: (64,)
layer1.1.bn2.num_batches_tracked with shape: ()
layer1.2.conv1.weight with shape: (64, 64, 3, 3)
layer1.2.bn1.weight with shape: (64,)
layer1.2.bn1.bias with shape: (64,)
layer1.2.bn1.running_mean with shape: (64,)
layer1.2.bn1.running_var with shape: (64,)
layer1.2.bn1.num_batches_tracked with shape: ()
layer1.2.conv2.weight with shape: (64, 64, 3, 3)
layer1.2.bn2.weight with shape: (64,)
layer1.2.bn2.bias with shape: (64,)
layer1.2.bn2.running_mean with shape: (64,)
layer1.2.bn2.running_var with shape: (64,)
layer1.2.bn2.num_batches_tracked with shape: ()
layer2.0.conv1.weight with shape: (128, 64, 3, 3)
layer2.0.bn1.weight with shape: (128,)
layer2.0.bn1.bias with shape: (128,)
layer2.0.bn1.running_mean with shape: (128,)
layer2.0.bn1.running_var with shape: (128,)
layer2.0.bn1.num_batches_tracked with shape: ()
layer2.0.conv2.weight with shape: (128, 128, 3, 3)
layer2.0.bn2.weight with shape: (128,)
layer2.0.bn2.bias with shape: (128,)
layer2.0.bn2.running_mean with shape: (128,)
layer2.0.bn2.running_var with shape: (128,)
layer2.0.bn2.num_batches_tracked with shape: ()
layer2.0.downsample.0.weight with shape: (128, 64)
layer2.0.downsample.1.weight with shape: (128,)
layer2.0.downsample.1.bias with shape: (128,)
layer2.0.downsample.1.running_mean with shape: (128,)
layer2.0.downsample.1.running_var with shape: (128,)
layer2.0.downsample.1.num_batches_tracked with shape: ()
layer2.1.conv1.weight with shape: (128, 128, 3, 3)
layer2.1.bn1.weight with shape: (128,)
layer2.1.bn1.bias with shape: (128,)
layer2.1.bn1.running_mean with shape: (128,)
layer2.1.bn1.running_var with shape: (128,)
layer2.1.bn1.num_batches_tracked with shape: ()
layer2.1.conv2.weight with shape: (128, 128, 3, 3)
layer2.1.bn2.weight with shape: (128,)
layer2.1.bn2.bias with shape: (128,)
layer2.1.bn2.running_mean with shape: (128,)
layer2.1.bn2.running_var with shape: (128,)
layer2.1.bn2.num_batches_tracked with shape: ()
layer2.2.conv1.weight with shape: (128, 128, 3, 3)
layer2.2.bn1.weight with shape: (128,)
layer2.2.bn1.bias with shape: (128,)
layer2.2.bn1.running_mean with shape: (128,)
layer2.2.bn1.running_var with shape: (128,)
layer2.2.bn1.num_batches_tracked with shape: ()
layer2.2.conv2.weight with shape: (128, 128, 3, 3)
layer2.2.bn2.weight with shape: (128,)
layer2.2.bn2.bias with shape: (128,)
layer2.2.bn2.running_mean with shape: (128,)
layer2.2.bn2.running_var with shape: (128,)
layer2.2.bn2.num_batches_tracked with shape: ()
layer2.3.conv1.weight with shape: (128, 128, 3, 3)
layer2.3.bn1.weight with shape: (128,)
layer2.3.bn1.bias with shape: (128,)
layer2.3.bn1.running_mean with shape: (128,)
layer2.3.bn1.running_var with shape: (128,)
layer2.3.bn1.num_batches_tracked with shape: ()
layer2.3.conv2.weight with shape: (128, 128, 3, 3)
layer2.3.bn2.weight with shape: (128,)
layer2.3.bn2.bias with shape: (128,)
layer2.3.bn2.running_mean with shape: (128,)
layer2.3.bn2.running_var with shape: (128,)
layer2.3.bn2.num_batches_tracked with shape: ()
layer3.0.conv1.weight with shape: (256, 128, 3, 3)
layer3.0.bn1.weight with shape: (256,)
layer3.0.bn1.bias with shape: (256,)
layer3.0.bn1.running_mean with shape: (256,)
layer3.0.bn1.running_var with shape: (256,)
layer3.0.bn1.num_batches_tracked with shape: ()
layer3.0.conv2.weight with shape: (256, 256, 3, 3)
layer3.0.bn2.weight with shape: (256,)
layer3.0.bn2.bias with shape: (256,)
layer3.0.bn2.running_mean with shape: (256,)
layer3.0.bn2.running_var with shape: (256,)
layer3.0.bn2.num_batches_tracked with shape: ()
layer3.0.downsample.0.weight with shape: (256, 128)
layer3.0.downsample.1.weight with shape: (256,)
layer3.0.downsample.1.bias with shape: (256,)
layer3.0.downsample.1.running_mean with shape: (256,)
layer3.0.downsample.1.running_var with shape: (256,)
layer3.0.downsample.1.num_batches_tracked with shape: ()
layer3.1.conv1.weight with shape: (256, 256, 3, 3)
layer3.1.bn1.weight with shape: (256,)
layer3.1.bn1.bias with shape: (256,)
layer3.1.bn1.running_mean with shape: (256,)
layer3.1.bn1.running_var with shape: (256,)
layer3.1.bn1.num_batches_tracked with shape: ()
layer3.1.conv2.weight with shape: (256, 256, 3, 3)
layer3.1.bn2.weight with shape: (256,)
layer3.1.bn2.bias with shape: (256,)
layer3.1.bn2.running_mean with shape: (256,)
layer3.1.bn2.running_var with shape: (256,)
layer3.1.bn2.num_batches_tracked with shape: ()
layer3.2.conv1.weight with shape: (256, 256, 3, 3)
layer3.2.bn1.weight with shape: (256,)
layer3.2.bn1.bias with shape: (256,)
layer3.2.bn1.running_mean with shape: (256,)
layer3.2.bn1.running_var with shape: (256,)
layer3.2.bn1.num_batches_tracked with shape: ()
layer3.2.conv2.weight with shape: (256, 256, 3, 3)
layer3.2.bn2.weight with shape: (256,)
layer3.2.bn2.bias with shape: (256,)
layer3.2.bn2.running_mean with shape: (256,)
layer3.2.bn2.running_var with shape: (256,)
layer3.2.bn2.num_batches_tracked with shape: ()
layer3.3.conv1.weight with shape: (256, 256, 3, 3)
layer3.3.bn1.weight with shape: (256,)
layer3.3.bn1.bias with shape: (256,)
layer3.3.bn1.running_mean with shape: (256,)
layer3.3.bn1.running_var with shape: (256,)
layer3.3.bn1.num_batches_tracked with shape: ()
layer3.3.conv2.weight with shape: (256, 256, 3, 3)
layer3.3.bn2.weight with shape: (256,)
layer3.3.bn2.bias with shape: (256,)
layer3.3.bn2.running_mean with shape: (256,)
layer3.3.bn2.running_var with shape: (256,)
layer3.3.bn2.num_batches_tracked with shape: ()
layer3.4.conv1.weight with shape: (256, 256, 3, 3)
layer3.4.bn1.weight with shape: (256,)
layer3.4.bn1.bias with shape: (256,)
layer3.4.bn1.running_mean with shape: (256,)
layer3.4.bn1.running_var with shape: (256,)
layer3.4.bn1.num_batches_tracked with shape: ()
layer3.4.conv2.weight with shape: (256, 256, 3, 3)
layer3.4.bn2.weight with shape: (256,)
layer3.4.bn2.bias with shape: (256,)
layer3.4.bn2.running_mean with shape: (256,)
layer3.4.bn2.running_var with shape: (256,)
layer3.4.bn2.num_batches_tracked with shape: ()
layer3.5.conv1.weight with shape: (256, 256, 3, 3)
layer3.5.bn1.weight with shape: (256,)
layer3.5.bn1.bias with shape: (256,)
layer3.5.bn1.running_mean with shape: (256,)
layer3.5.bn1.running_var with shape: (256,)
layer3.5.bn1.num_batches_tracked with shape: ()
layer3.5.conv2.weight with shape: (256, 256, 3, 3)
layer3.5.bn2.weight with shape: (256,)
layer3.5.bn2.bias with shape: (256,)
layer3.5.bn2.running_mean with shape: (256,)
layer3.5.bn2.running_var with shape: (256,)
layer3.5.bn2.num_batches_tracked with shape: ()
layer4.0.conv1.weight with shape: (512, 256, 3, 3)
layer4.0.bn1.weight with shape: (512,)
layer4.0.bn1.bias with shape: (512,)
layer4.0.bn1.running_mean with shape: (512,)
layer4.0.bn1.running_var with shape: (512,)
layer4.0.bn1.num_batches_tracked with shape: ()
layer4.0.conv2.weight with shape: (512, 512, 3, 3)
layer4.0.bn2.weight with shape: (512,)
layer4.0.bn2.bias with shape: (512,)
layer4.0.bn2.running_mean with shape: (512,)
layer4.0.bn2.running_var with shape: (512,)
layer4.0.bn2.num_batches_tracked with shape: ()
layer4.0.downsample.0.weight with shape: (512, 256)
layer4.0.downsample.1.weight with shape: (512,)
layer4.0.downsample.1.bias with shape: (512,)
layer4.0.downsample.1.running_mean with shape: (512,)
layer4.0.downsample.1.running_var with shape: (512,)
layer4.0.downsample.1.num_batches_tracked with shape: ()
layer4.1.conv1.weight with shape: (512, 512, 3, 3)
layer4.1.bn1.weight with shape: (512,)
layer4.1.bn1.bias with shape: (512,)
layer4.1.bn1.running_mean with shape: (512,)
layer4.1.bn1.running_var with shape: (512,)
layer4.1.bn1.num_batches_tracked with shape: ()
layer4.1.conv2.weight with shape: (512, 512, 3, 3)
layer4.1.bn2.weight with shape: (512,)
layer4.1.bn2.bias with shape: (512,)
layer4.1.bn2.running_mean with shape: (512,)
layer4.1.bn2.running_var with shape: (512,)
layer4.1.bn2.num_batches_tracked with shape: ()
layer4.2.conv1.weight with shape: (512, 512, 3, 3)
layer4.2.bn1.weight with shape: (512,)
layer4.2.bn1.bias with shape: (512,)
layer4.2.bn1.running_mean with shape: (512,)
layer4.2.bn1.running_var with shape: (512,)
layer4.2.bn1.num_batches_tracked with shape: ()
layer4.2.conv2.weight with shape: (512, 512, 3, 3)
layer4.2.bn2.weight with shape: (512,)
layer4.2.bn2.bias with shape: (512,)
layer4.2.bn2.running_mean with shape: (512,)
layer4.2.bn2.running_var with shape: (512,)
layer4.2.bn2.num_batches_tracked with shape: ()
fc.weight with shape: (1000, 512)
fc.bias with shape: (1000,)
Done. Output file: ./resnet-ggml-model-f32.bin
parameters = 21814732

```

### Build & Run
```
$cd engine/build
$ make
$ ./bin/resnet34 ../examples/resnet/resnet-ggml-model-f32.bin ../examples/resnet34/dog.bin
```


### BatchNorm

此处，以模型的bn1为例说明。
#### bn1定义

```
self.bn1 = nn.BatchNorm2d(64)
```

构造参数64是指batch数量，对于图像就是输入特征图的通道数目.
##### bn1的输入和输出
bn1的输入和输出都是： 112 x 112 x   64 x   1，具体如下：
```
model.input  shape:  224 x 224 x    3 x   1
model.conv1  shape:  112 x 112 x   64 x   1
model.bn1  shape:  112 x 112 x   64 x   1
model.relu  shape:  112 x 112 x   64 x   1
model.maxpool  shape:   56 x  56 x   64 x   1
```

##### bn1的参数
bn1有5个参数，推理时实际使用的是4个参数。
```
bn1.weight with shape: (64,)             //如果affine=True，这个参数表示可学习的缩放因子，形状与num_features相同
bn1.bias with shape: (64,)               //如果affine=True，这个参数表示可学习的偏移量，形状与num_features相同。
bn1.running_mean with shape: (64,)       // 运行均值，形状与num_features相同，用于跟踪输入数据的均值。
bn1.running_var with shape: (64,)        // 运行方差，形状与num_features相同，用于跟踪输入数据的方差。
bn1.num_batches_tracked with shape: ()   // 它是一个整数，用于跟踪层自初始化以来处理的批次数量。这个属性主要用于调试和分析模型，以确保批量归一化层能够正常地跟踪统计数据。num_batches_tracked通常只在调试或分析模型性能时使用，对于模型的训练和推理过程没有直接影响。
```
#### pytorch forward
pytorch推理使用简单，pytorch内部已经自动处理，代码如下：
```
out = self.bn1(out)
```

#### ml 推理

##### 1code：
```
static ggml_tensor * apply_bn2d(ggml_context * ctx, ggml_tensor * input, const bn_layer & layer)
{
    struct ggml_tensor * sub_result = ggml_sub(ctx, input, ggml_batch_repeat(ctx, layer.mean, input));

    struct ggml_tensor * sqrt_result = ggml_sqrt(ctx,layer.var);

    struct ggml_tensor * div_result = ggml_div(ctx,sub_result,ggml_batch_repeat(ctx, sqrt_result, sub_result));
    
    struct ggml_tensor * result = ggml_mul(ctx,div_result,ggml_batch_repeat(ctx,layer.weight, div_result));

    result = ggml_add(ctx, result, ggml_batch_repeat(ctx, layer.bias, result));

    return result;
}
```
##### 2推理说明：
对于bn1，执行函数apply_bn2d，

输入input为112 x 112 x 64 x 1  
layer.mean 为 64 x 1 x 1 x 1  
进行减去均值操作具体执行:  
input的第一个特征图（形状为112x112）内的所有数都减去同一个均值layer.mean[0]  
依次类推，input的第64个特征图（形状为112x112）内的所有数都减去同一个均值layer.mean[63]  

利用算子ggml_sub（对应数减法操作）前，需要对参数layer.mean进行数据形状修改
对layer.mean 进行batch repeat 变为112 x 112 x   64 x   1。 
batch repeat所做的工作为，将 64 x   1 x    1 x   1的第一个浮点数，变为112x112的形状。 
比如layer.mean[0]=0.8,最后变为112x112：  

0.8 0.8 0.8 ....  
0.8 0.8 0.8 ....  
....  
0.8 0.8 0.8 ....  

其他对应数的ggml_div，ggml_mul，ggml_add操作类似，都需要对参数进行batch repeat.








