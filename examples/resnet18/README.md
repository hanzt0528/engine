https://pytorch.org/hub/pytorch_vision_resnet/

### ResNet18 网络结构
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
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
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
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=1000, bias=True)
)

```
### export resnet18 of pytorch  to state_dict
```
 python export_pytorch_model.py 
```

### Convert resnet to ggml
```
python convert-resnet-to-ggml.py resnet18.state_dict
conv1.weight with shape: (64, 3, 7, 7) //权重维度: (out_channels, in_channels, kernel_height, kernel_width)
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
fc.weight with shape: (1000, 512)
fc.bias with shape: (1000,)
Done. Output file: ./resnet-ggml-model-f32.bin
parameters = 11699132

```


```
conv1 = tensor([[[[ 4.3629e+00,  5.6763e+00,  5.8927e+00,  ...,  6.1479e+00,
            6.0579e+00,  5.0241e+00],
          [ 4.3918e-01,  3.8344e-01,  2.3377e-01,  ...,  2.3830e-01,
            2.8579e-01,  3.5735e-01],
          [-5.8754e-02, -2.6915e-01, -3.4214e-01,  ..., -3.8320e-01,
           -3.5673e-01, -2.8614e-01],
          ...,
          [-2.2223e-02,  3.6131e-01, -5.0757e-01,  ..., -2.5818e+00,
           -1.0725e+00,  2.9184e-01],
          [ 1.2983e-01, -2.1259e-01, -3.9361e-01,  ...,  4.0117e-01,
            1.7911e+00, -5.5442e-01],
          [ 2.1933e+00,  2.3124e+00,  2.5271e+00,  ...,  2.6435e+00,
            2.7349e+00,  1.2444e+00]],

         [[ 5.6134e+00,  8.7625e+00,  9.2880e+00,  ...,  9.6701e+00,
            9.5319e+00,  7.5968e+00],
          [-9.0728e-01,  1.1494e+00,  1.3145e+00,  ...,  1.4804e+00,
            1.6091e+00,  3.4004e+00],
          [-1.0218e+00,  9.5702e-01,  1.1493e+00,  ...,  1.1611e+00,4
            1.2797e+00,  3.0486e+00],
          ...,
          [ 1.8546e-01,  1.2346e+00,  2.6786e-01,  ...,  1.1090e-01,
           -8.4476e-01, -2.3423e-01],
          [-1.6221e-01,  1.8537e+00,  1.5329e+00,  ...,  1.8056e+00,
            1.4630e+00,  1.1777e+00],
          [-1.0927e+00,  5.6517e-01,  1.5499e+00,  ...,  1.0297e+00,
            1.6683e+00,  2.8001e+00]],

         [[-6.1478e-06, -1.0587e-05, -1.2469e-05,  ..., -1.2891e-05,
           -1.2881e-05, -9.9851e-06],
          [-7.2367e-06, -1.2809e-05, -1.4924e-05,  ..., -1.5282e-05,
           -1.5377e-05, -1.1873e-05],
          [-5.6018e-06, -1.0566e-05, -1.2140e-05,  ..., -1.2356e-05,
           -1.2450e-05, -9.8512e-06],
          ...,
          [-4.4104e-06, -7.9040e-06, -8.2976e-06,  ..., -8.7380e-06,
           -7.4266e-06, -5.0804e-06],
          [-4.5125e-06, -8.0609e-06, -9.1182e-06,  ..., -8.2206e-06,
           -7.7993e-06, -5.7559e-06],
          [-7.3222e-07, -2.3917e-06, -3.1863e-06,  ..., -3.2196e-06,
           -3.2337e-06, -2.6346e-06]],

         ...,

         [[-2.8804e+00, -3.8044e+00, -3.6452e+00,  ..., -3.7423e+00,
           -3.7104e+00, -3.4506e+00],
          [-5.2958e+00, -6.6897e+00, -6.1827e+00,  ..., -6.3588e+00,
           -6.2698e+00, -5.1878e+00],
          [-5.4938e+00, -6.8206e+00, -5.8854e+00,  ..., -6.0112e+00,
           -5.9405e+00, -5.2222e+00],
          ...,
          [-3.1103e+00, -3.8914e+00, -3.0326e+00,  ..., -2.7317e+00,
           -2.3092e+00, -1.9152e+00],
          [-3.5750e+00, -4.0908e+00, -2.9071e+00,  ..., -2.6332e+00,
           -2.0916e+00, -2.2760e+00],
          [-2.5659e+00, -3.1983e+00, -2.3182e+00,  ..., -2.1234e+00,
           -1.7520e+00, -1.3056e+00]],

         [[ 2.4467e-01,  1.0502e+00,  1.1272e+00,  ...,  1.2102e+00,
            1.2434e+00,  1.6062e+00],
          [ 2.0627e-01,  2.9865e+00,  3.3270e+00,  ...,  3.5114e+00,
            3.5368e+00,  4.2130e+00],
          [ 1.9237e-01,  2.8687e+00,  3.2536e+00,  ...,  3.3693e+00,
            3.4129e+00,  4.1513e+00],
          ...,
          [ 3.4049e-01,  1.8914e+00,  1.4089e+00,  ...,  1.8867e+00,
            1.6037e+00,  1.7953e+00],
          [ 1.5773e-01,  1.9855e+00,  2.1663e+00,  ...,  2.0279e+00,
            1.4644e+00,  1.1593e+00],
          [ 3.0782e-01,  2.3598e+00,  2.7491e+00,  ...,  2.0368e+00,
            2.1450e+00,  2.0121e+00]],

         [[-4.0774e+00,  4.6843e-01,  3.8531e-01,  ...,  8.9900e-01,
            1.0431e+00,  2.3468e+00],
          [-1.5162e+00,  1.0662e+00,  8.6266e-01,  ...,  6.4375e-01,
            6.2438e-01,  9.3583e-01],
          [-1.6212e+00,  9.1913e-01,  5.6466e-01,  ...,  4.6679e-01,
            4.7336e-01,  9.2084e-01],
          ...,
          [-2.3794e-01, -2.4836e-02, -8.2706e-01,  ...,  7.6449e-01,
            4.4538e-01,  2.3527e-02],
          [-1.8042e+00, -4.1037e-02,  2.2220e+00,  ..., -8.4766e-01,
            5.7394e-01, -1.2293e+00],
          [ 8.5052e-01,  3.7601e+00,  2.0755e+00,  ...,  1.6258e+00,
            1.6986e+00,  2.6708e+00]]]])
bn1.mean = tensor([ 2.7681e-03, -2.5769e-02,  2.1254e-07, -8.4605e-02,  2.1121e-08,
         4.9691e-04, -2.2408e-02, -1.1582e-07, -4.8239e-03,  2.7507e-07,
         3.9582e-02,  3.1994e-02, -3.7490e-02, -1.3716e-06,  6.6002e-03,
         4.3782e-03,  6.4797e-02,  1.1176e-01,  3.6002e-02, -7.5075e-02,
        -3.8240e-02,  8.4358e-02, -5.2287e-02, -1.1799e-02,  1.3019e-03,
         3.2172e-02, -1.7784e-02, -9.1009e-02,  1.1319e-01, -4.1632e-02,
         8.7302e-03,  2.9693e-02, -7.0502e-02, -3.4847e-03,  1.0977e-01,
        -1.7341e-03, -5.9423e-08,  2.9330e-02, -7.8553e-09,  6.7320e-03,
        -3.7100e-03,  1.6028e-02, -2.7883e-02,  2.6593e-02,  2.8475e-02,
        -1.2735e-01,  4.4617e-02,  2.6329e-02,  2.1454e-08, -1.7045e-02,
        -3.5617e-03, -4.5841e-02,  6.3876e-02,  1.5220e-02, -3.8511e-02,
        -1.6428e-02, -1.6569e-02,  5.6057e-02, -8.0306e-02, -2.6646e-03,
        -4.1718e-02,  1.2611e-01, -4.9237e-02, -1.3261e-02])
out - mean = tensor([[[[ 4.3601e+00,  5.6735e+00,  5.8900e+00,  ...,  6.1452e+00,
            6.0551e+00,  5.0213e+00],
          [ 4.3641e-01,  3.8067e-01,  2.3100e-01,  ...,  2.3553e-01,
            2.8303e-01,  3.5458e-01],
          [-6.1522e-02, -2.7192e-01, -3.4491e-01,  ..., -3.8597e-01,
           -3.5950e-01, -2.8891e-01],
          ...,
          [-2.4991e-02,  3.5855e-01, -5.1033e-01,  ..., -2.5845e+00,
           -1.0753e+00,  2.8907e-01],
          [ 1.2706e-01, -2.1536e-01, -3.9638e-01,  ...,  3.9840e-01,
            1.7883e+00, -5.5719e-01],
          [ 2.1905e+00,  2.3097e+00,  2.5244e+00,  ...,  2.6407e+00,
            2.7321e+00,  1.2416e+00]],

         [[ 5.6392e+00,  8.7883e+00,  9.3138e+00,  ...,  9.6958e+00,
            9.5576e+00,  7.6226e+00],
          [-8.8151e-01,  1.1751e+00,  1.3402e+00,  ...,  1.5062e+00,
            1.6349e+00,  3.4262e+00],
          [-9.9602e-01,  9.8278e-01,  1.1750e+00,  ...,  1.1869e+00,
            1.3054e+00,  3.0744e+00],
          ...,
          [ 2.1123e-01,  1.2603e+00,  2.9363e-01,  ...,  1.3667e-01,
           -8.1899e-01, -2.0846e-01],
          [-1.3644e-01,  1.8795e+00,  1.5586e+00,  ...,  1.8313e+00,
            1.4888e+00,  1.2035e+00],
          [-1.0669e+00,  5.9093e-01,  1.5757e+00,  ...,  1.0555e+00,
            1.6940e+00,  2.8259e+00]],

         [[-6.3603e-06, -1.0799e-05, -1.2682e-05,  ..., -1.3103e-05,
           -1.3094e-05, -1.0198e-05],
          [-7.4493e-06, -1.3022e-05, -1.5137e-05,  ..., -1.5495e-05,
           -1.5590e-05, -1.2085e-05],
          [-5.8144e-06, -1.0779e-05, -1.2353e-05,  ..., -1.2568e-05,
           -1.2662e-05, -1.0064e-05],
          ...,
          [-4.6230e-06, -8.1165e-06, -8.5101e-06,  ..., -8.9506e-06,
           -7.6392e-06, -5.2929e-06],
          [-4.7251e-06, -8.2734e-06, -9.3308e-06,  ..., -8.4332e-06,
           -8.0118e-06, -5.9684e-06],
          [-9.4477e-07, -2.6042e-06, -3.3989e-06,  ..., -3.4321e-06,
           -3.4462e-06, -2.8472e-06]],

         ...,

         [[-3.0065e+00, -3.9305e+00, -3.7713e+00,  ..., -3.8685e+00,
           -3.8365e+00, -3.5767e+00],
          [-5.4219e+00, -6.8158e+00, -6.3088e+00,  ..., -6.4850e+00,
           -6.3959e+00, -5.3139e+00],
          [-5.6200e+00, -6.9467e+00, -6.0115e+00,  ..., -6.1373e+00,
           -6.0666e+00, -5.3483e+00],
          ...,
          [-3.2364e+00, -4.0175e+00, -3.1587e+00,  ..., -2.8579e+00,
           -2.4353e+00, -2.0413e+00],
          [-3.7012e+00, -4.2169e+00, -3.0332e+00,  ..., -2.7593e+00,
           -2.2177e+00, -2.4021e+00],
          [-2.6920e+00, -3.3245e+00, -2.4444e+00,  ..., -2.2495e+00,
           -1.8781e+00, -1.4317e+00]],

         [[ 2.9391e-01,  1.0995e+00,  1.1764e+00,  ...,  1.2594e+00,
            1.2926e+00,  1.6554e+00],
          [ 2.5551e-01,  3.0357e+00,  3.3762e+00,  ...,  3.5607e+00,
            3.5860e+00,  4.2622e+00],
          [ 2.4161e-01,  2.9180e+00,  3.3029e+00,  ...,  3.4185e+00,
            3.4622e+00,  4.2006e+00],
          ...,
          [ 3.8972e-01,  1.9406e+00,  1.4582e+00,  ...,  1.9359e+00,
            1.6529e+00,  1.8445e+00],
          [ 2.0696e-01,  2.0347e+00,  2.2156e+00,  ...,  2.0772e+00,
            1.5136e+00,  1.2085e+00],
          [ 3.5706e-01,  2.4090e+00,  2.7984e+00,  ...,  2.0861e+00,
            2.1942e+00,  2.0614e+00]],

         [[-4.0642e+00,  4.8169e-01,  3.9857e-01,  ...,  9.1226e-01,
            1.0563e+00,  2.3601e+00],
          [-1.5029e+00,  1.0794e+00,  8.7592e-01,  ...,  6.5701e-01,
            6.3764e-01,  9.4909e-01],
          [-1.6079e+00,  9.3239e-01,  5.7793e-01,  ...,  4.8005e-01,
            4.8662e-01,  9.3410e-01],
          ...,
          [-2.2467e-01, -1.1575e-02, -8.1380e-01,  ...,  7.7775e-01,
            4.5864e-01,  3.6788e-02],
          [-1.7909e+00, -2.7776e-02,  2.2353e+00,  ..., -8.3440e-01,
            5.8720e-01, -1.2160e+00],
          [ 8.6378e-01,  3.7733e+00,  2.0888e+00,  ...,  1.6391e+00,
            1.7119e+00,  2.6841e+00]]]])
```


### image preprocess
Convert jpt to bin file as input.
```
$cd engine/examples/resnet18
$ python convert-jpg-to-bin.py
```

### Build & Run
```
$cd engine/build
$ cmake ..
$ make
$ ./bin/resnet18 ../examples/resnet18/resnet-ggml-model-f32.bin ../examples/resnet18/dog.bin

model.input  shape:  224 x 224 x    3 x   1
model.conv1  shape:  112 x 112 x   64 x   1
model.bn1.mean  shape:   64 x   1 x    1 x   1
model.bn1  shape:  112 x 112 x   64 x   1
model.relu  shape:  112 x 112 x   64 x   1
model.maxpool  shape:   56 x  56 x   64 x   1
Layer 0 block 0 bn2 output shape:   56 x  56 x   64 x   1
Layer 0 block 1 bn2 output shape:   56 x  56 x   64 x   1
Layer 1 block 0 bn2 output shape:   28 x  28 x  128 x   1
Layer 1 block 1 bn2 output shape:   28 x  28 x  128 x   1
Layer 2 block 0 bn2 output shape:   14 x  14 x  256 x   1
Layer 2 block 1 bn2 output shape:   14 x  14 x  256 x   1
Layer 3 block 0 bn2 output shape:    7 x   7 x  512 x   1
Layer 3 block 1 bn2 output shape:    7 x   7 x  512 x   1
model.avgpool  shape:    1 x   1 x  512 x   1
model.fc  shape:  1000 x   1 x    1 x   1
model.softmax  shape:  1000 x   1 x    1 x   1
top[0]=0.884944
top[1]=0.0456407
top[2]=0.0441554
top[3]=0.00562407
top[4]=0.00464509
top[4]=0.00464509 index = 257 label = Great Pyrenees
top[0]=0.884944 index = 258 label = Samoyed
top[3]=0.00562407 index = 259 label = Pomeranian
top[2]=0.0441554 index = 270 label = white wolf
top[1]=0.0456407 index = 279 label = Arctic fox
main: predicted digit is 258
duration = 4887

```

### Conv2d
此处以conv1为例说明。
#### conv1定义

```
self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
```
#### conv1权值形状 64x3x7x7
权值维度说明: (out_channels, in_channels, kernel_height, kernel_width)

conv1输出为64个通道；输入为3个通道，对应图像的RGB 3个通道；卷积核大小为3x3


#### pytorch forward

```
out = self.conv1(x)
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








