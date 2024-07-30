# LeNet5-MNIST-PyTorch

#### This is the simplest implementation of the paper "Gradient-based learning applied to document recognition" in PyTorch.
#### Have a try with artificial intelligence!
#### Feel free to ask anything!
![image](https://user-images.githubusercontent.com/25716030/162345646-b13c9af0-bdb5-4ce7-9a62-c0834cba9e5f.png)
## Requirments
Python3  
PyTorch >= 0.4.0  
torchvision >= 0.1.8
## Usage
```
$git clone https://github.com/ChawDoe/LeNet-5-MNIST-PyTorch.git  
$cd LeNet5-MNIST-PyTorch  
$python3 train.py  
```
model will now run on GPU if available

## Hint
This repo includes the mnist dataset.
## Accuracy
Average precision on test set: 99%

## Model Convert
```
python convert-lenet-to-ggml.py models/mnist_0.981.pkl
Processing variable: conv1.weight with shape:  (6, 5, 5)
Processing variable: conv1.bias with shape:  (6,)
Processing variable: conv2.weight with shape:  (16, 6, 5, 5)
Processing variable: conv2.bias with shape:  (16,)
Processing variable: fc1.weight with shape:  (120, 256)
Processing variable: fc1.bias with shape:  (120,)
Processing variable: fc2.weight with shape:  (84, 120)
Processing variable: fc2.bias with shape:  (84,)
Processing variable: fc3.weight with shape:  (10, 84)
Processing variable: fc3.bias with shape:  (10,)

```
## test.py

```
python test.py 
conv1 result shape = torch.Size([1, 6, 24, 24])
conv2 result shape = torch.Size([1, 16, 8, 8])
fc1 input shape = torch.Size([1, 256])
fc1  result = torch.Size([1, 120])
fc2  result = torch.Size([1, 84])
fc3 result = torch.Size([1, 10])
accuracy: 0.000
Model finished training

```
## lenet run
```
/engine/build# ./bin/lenet ../examples/lenet/lenet-ggml-model-f32.bin ../examples/lenet/test/MNIST/raw/t10k-images-idx3-ubyte 

read conv1 n_dims = 3
n_dims[0]= 5
n_dims[1]= 5
n_dims[2]= 6
conv1 data size = 600
read conv1 bias dims = 1
read conv1 bias dim value = 6
read conv2 weight n_dims = 4
n_dims[0]= 5
n_dims[1]= 5
n_dims[2]= 6
n_dims[3]= 16
conv2 weight data size = 9600
read conv2 bias dims = 1
read conv2 bias dim value = 16
read fc1 weight n_dims = 2
n_dims[0]= 256
n_dims[0]= 120
fc1 weight data size = 122880
read fc1 bias dims = 1
read fc1 bias dim value = 120
read fc2 weight n_dims = 2
n_dims[1]= 120
n_dims[1]= 84
fc2 weight data size = 40320
read fc2 bias dims = 1
read fc2 bias dim value = 84
read fc3 weight n_dims = 2
n_dims[2]= 84
n_dims[2]= 10
fc3 weight data size = 3360
read fc3 bias dims = 1
read fc3 bias dim value = 10
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ _ _ _ _ * * * * _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ _ _ _ * * * * * _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ _ _ * * * * * * _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ * _ _ _ _ _ * * * * * * _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ * _ _ _ _ * * * * * _ _ _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ _ * * * * * _ _ _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ * * * * * _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ * * * * * * _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ * * * * * * _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ * * * * * * _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ * * * * * _ _ _ _ * * * * * _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ * * * * * _ _ _ _ * * * * * * _ _ _ _ _ _ 
_ _ _ _ _ _ _ * * * * * _ _ * * * * * * * * * _ _ _ _ _ 
_ _ _ _ _ _ _ * * * * * _ _ * * * * * * * * * _ _ _ _ _ 
_ _ _ _ _ _ _ * * * * * _ _ * * * * * * * * * _ _ _ _ _ 
_ _ _ _ _ _ _ * * * * * _ * * * * * * * * * * _ _ _ _ _ 
_ _ _ _ _ _ _ * * * * * * * * * * * * * * * * _ _ _ _ _ 
_ _ _ _ _ _ _ _ * * * * * * * * * * * * * * * _ _ _ _ _ 
_ _ _ _ _ _ _ _ * * * * * * * * * * * * * * _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ _ * * * * * * * * * * * * _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

Layer  0 output shape:   24 x  24 x    6 x   1
Layer  1 output shape:    8 x   8 x   16 x   1
Layer  2 output shape:  120 x   1 x    1 x   1
Layer  3 output shape:   84 x   1 x    1 x   1
Layer  4 output shape:   10 x   1 x    1 x   1
Layer  5 output shape:   10 x   1 x    1 x   1
ggml_graph_dump_dot: dot -Tpng lenet.dot -o lenet.dot.png && open lenet.dot.png
model_eval: exported compute graph to 'lenet.ggml'
conv1 bias data:
conv2d_result data:
add_result data:
relu_result data:
probs_data:
0
0
0
0
0
4.76837e-07
1
0
0
0
main: predicted digit is 6

```
