https://pytorch.org/hub/pytorch_vision_vgg/

### vgg.py
```
python vgg.py 
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace=True)
    (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (11): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (12): ReLU(inplace=True)
    (13): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (14): ReLU(inplace=True)
    (15): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (16): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (17): ReLU(inplace=True)
    (18): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (19): ReLU(inplace=True)
    (20): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
Samoyed 0.6673735976219177
Pomeranian 0.16195233166217804
Eskimo dog 0.017759378999471664
collie 0.017686229199171066
keeshond 0.01706553064286709

```

### convert pytorch to ggml
```
python convert-vgg-to-ggml.py vgg.state_dict 
eatures.0.weight with shape: (64, 3, 3, 3)
features.0.bias with shape: (64,)
features.3.weight with shape: (128, 64, 3, 3)
features.3.bias with shape: (128,)
features.6.weight with shape: (256, 128, 3, 3)
features.6.bias with shape: (256,)
features.8.weight with shape: (256, 256, 3, 3)
features.8.bias with shape: (256,)
features.11.weight with shape: (512, 256, 3, 3)
features.11.bias with shape: (512,)
features.13.weight with shape: (512, 512, 3, 3)
features.13.bias with shape: (512,)
features.16.weight with shape: (512, 512, 3, 3)
features.16.bias with shape: (512,)
features.18.weight with shape: (512, 512, 3, 3)
features.18.bias with shape: (512,)
classifier.0.weight with shape: (4096, 25088)
classifier.0.bias with shape: (4096,)
classifier.3.weight with shape: (4096, 4096)
classifier.3.bias with shape: (4096,)
classifier.6.weight with shape: (1000, 4096)
classifier.6.bias with shape: (1000,)
Done. Output file: ./vgg-ggml-model-f32.bin
parameters = 132863336
```

