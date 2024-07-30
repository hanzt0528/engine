
import sys
import struct
import numpy as np


import torch
import torch.nn as nn

import torchvision.transforms as transforms
from model import Model
if len(sys.argv) != 2:
    print("Usage: convert-lenet-to-ggml.py models/mnist_0.981.pkl\n")
    sys.exit(1)
    
state_dict_file = sys.argv[1]

fname_out="./lenet-ggml-model-f32.bin"
model = torch.load(state_dict_file,map_location=torch.device('cpu'))

list_vars=model.state_dict()
print(list_vars)
 
fout = open(fname_out,"wb")
fout.write(struct.pack("i",0x67676d6c))

for name in list_vars.keys():
    data = list_vars[name].squeeze().numpy()
    print("Processing variable: " + name + " with shape: ", data.shape) 
    n_dims = len(data.shape)
    fout.write(struct.pack("i",n_dims))
    
    data=data.astype(np.float32)
    
    for i in range(n_dims):
        fout.write(struct.pack("i",data.shape[n_dims-1-i]))
        
    data.tofile(fout)
fout.close()

print("Done. Output file: " + fname_out)
