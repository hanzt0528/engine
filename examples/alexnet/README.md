
https://pytorch.org/hub/pytorch_vision_alexnet/

##Export AlexNet weight to state_dict
```
python export_state_dict.py
```
## Convert state_dict to ggml
```
python convert-alexnet-to-ggml.py alexnet.state_dict
features.0.weight with shape:  (64, 3, 11, 11)
features.0.bias with shape:  (64,)
features.3.weight with shape:  (192, 64, 5, 5)
features.3.bias with shape:  (192,)
features.6.weight with shape:  (384, 192, 3, 3)
features.6.bias with shape:  (384,)
features.8.weight with shape:  (256, 384, 3, 3)
features.8.bias with shape:  (256,)
features.10.weight with shape:  (256, 256, 3, 3)
features.10.bias with shape:  (256,)
classifier.1.weight with shape:  (4096, 9216)
classifier.1.bias with shape:  (4096,)
classifier.4.weight with shape:  (4096, 4096)
classifier.4.bias with shape:  (4096,)
classifier.6.weight with shape:  (1000, 4096)
classifier.6.bias with shape:  (1000,)
Done. Output file: ./alexnet-ggml-model-f32.bin
parameters = 61100840
```

```
./bin/alexnet ../examples/alexnet/alexnet-ggml-model-f32.bin ../examples/lenet/test/MNIST/raw/

```
```
AlexNet(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace=True)
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=9216, out_features=4096, bias=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU(inplace=True)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
```