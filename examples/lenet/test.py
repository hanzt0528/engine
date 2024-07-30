from model import Model
import numpy as np
import os
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import struct
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 1
    train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor())
    test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = torch.load('models/mnist_0.981.pkl')

    #model = Model().to(device)
    
    print(model.conv2.weight)
    exit(0)
    
    prev_acc = 0
    model.eval()
    all_correct_num = 0
    all_sample_num = 0
    count = 0
    for idx, (test_x, test_label) in enumerate(test_loader):
        
        print(f"test_x:{test_x.shape}")
        fout = open("7.bin","wb")
        data = test_x.squeeze().numpy()
        data.tofile(fout)
        fout.close()
        
        print(test_x)
        for i in range(28):
            for j in range(28):

                if test_x[0][0][i][j]>0:
                    print("*",end="")
                else:
                    print("-",end="")
            print("")

        print("test_lable")
        print(test_label)
        test_x = test_x.to(device)
        test_label = test_label.to(device)
        predict_y = model(test_x.float()).detach()
        predict_y =torch.argmax(predict_y, dim=-1)
        current_correct_num = predict_y == test_label
        all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
        all_sample_num += current_correct_num.shape[0]
     
        break
        
    
        count=count+1
    acc = all_correct_num / all_sample_num
    print('accuracy: {:.3f}'.format(acc), flush=True)
        
    prev_acc = acc
    print("Model finished training")
