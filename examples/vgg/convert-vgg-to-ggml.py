import sys
import struct
import numpy as np

import torch
import torch.nn as nn

if len(sys.argv) != 2:
    print("Usage: convert-vgg-to-ggml.py model\n")
    sys.exit(1)
    

state_dict_file = sys.argv[1]

fname_out = "./vgg-ggml-model-f32.bin"

state_dict = torch.load(state_dict_file,map_location=torch.device('cpu'))

list_vars = state_dict

#print(list_vars.keys())    

fout = open(fname_out,'wb')

fout.write(struct.pack("i",0x67676d6c)) # magic: ggml in hex

parameters = 0


for name in list_vars.keys():
    data = list_vars[name].squeeze().numpy()
    print(name + " with shape:",data.shape)
    n_dims = len(data.shape)
    fout.write(struct.pack("i",n_dims))
    temp = 1
    for i in range(n_dims):
        fout.write(struct.pack("i",data.shape[n_dims-1-i]))
        temp=temp*data.shape[n_dims-1-i]
        
    parameters =parameters+temp
    data = data.astype(np.float32)
    data.tofile(fout)
    
fout.close()

print("Done. Output file: " + fname_out)
print(f"parameters = {parameters}")
print("")
    
    
