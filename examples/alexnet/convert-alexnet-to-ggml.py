# Convert MNIS h5 transformer model to ggml format
#
# Load the (state_dict) saved model using PyTorch
# Iterate over all variables and write them to a binary file.
#
# For each variable, write the following:
#   - Number of dimensions (int)
#   - Name length (int)
#   - Dimensions (int[n_dims])
#   - Name (char[name_length])
#   - Data (float[n_dims])
#
# At the start of the ggml file we write the model parameters

import sys
import struct
import json
import numpy as np
import re


import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

if len(sys.argv) != 2:
    print("Usage: convert-alexnet-to-ggml.py model\n")
    sys.exit(1)

state_dict_file = sys.argv[1]
#fname_out = "models/mnist/ggml-model-f32.bin"
fname_out = "./alexnet-ggml-model-f32.bin"

state_dict = torch.load(state_dict_file, map_location=torch.device('cpu'))
#print (model)

list_vars = state_dict
print (list_vars.keys())

fout = open(fname_out, "wb")

fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex

parameters = 0
for name in list_vars.keys():
    data = list_vars[name].squeeze().numpy()
    print(name + " with shape: ", data.shape) 
    n_dims = len(data.shape);
   
    fout.write(struct.pack("i", n_dims))
    
    data = data.astype(np.float32)
    temp = 1
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        temp=temp*data.shape[n_dims-1-i]
        
    parameters =parameters+temp
    # data
    data.tofile(fout)

fout.close()

print("Done. Output file: " + fname_out)
print(f"parameters = {parameters}")
print("")
