import argparse
import os
import numpy as np
import math
import sys
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import random
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from utils import *
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=5e-3, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--z_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--N", type=int, default=256, help="number of devices")
parser.add_argument("--pro", type=int, default=1, help="number of activity")
parser.add_argument("--M", type=int, default=8, help="size of antenna")
parser.add_argument("--L", type=int, default=128, help="size of sequense")
parser.add_argument("--SNR", type=float, default='30', help="var of noise")
parser.add_argument("--name", type=str, default='standard_GAN_1815', help="filename")
parser.add_argument("--device_id", type=int, default='3', help="device_id")
opt = parser.parse_args()

torch.cuda.set_device(opt.device_id)
#generator
path='./result/{}_2_N:{}_M:{}_L:{}_pro:{}_SNR:{}'.format(opt.name,opt.N,opt.M,opt.L,opt.pro,opt.SNR)
generator=torch.load(path+'/generator.pkl').cuda()
#generator.eval()
#z and optim
z=Variable(torch.randn(opt.batch_size,opt.z_dim))
z=z.cuda()
z.requires_grad=True
optimizer=optim.Adam([z],lr=opt.lr) 
# pilot matrix
P_real=torch.randn(opt.N,opt.L).cuda()
P_imag=torch.randn(opt.N,opt.L).cuda()
P=torch.cat((torch.cat((P_real,-P_imag),dim=1),torch.cat((P_imag,P_real),dim=1)),dim=0)
#test data
X,Y,_=gen_data_tensor(opt.L,opt.N,opt.M,opt.batch_size,P,opt.pro,opt.SNR)
Y=torch.matmul(X.transpose(1,2),P)

NMSE_Y_list=[]
NMSE_X_list=[]
for i in range(opt.n_epochs):
    optimizer.zero_grad()
    X_hat=generator(z)
    Y_hat=torch.matmul(X_hat,P)
    loss=torch.norm(Y_hat-Y)**2/opt.batch_size 
    #loss=torch.norm(X_hat.transpose(1,2)-X)**2/opt.batch_size 
    NMSE_Y=20*math.log(torch.norm(Y_hat-Y)/torch.norm(Y),10)
    NMSE_X=20*math.log(torch.norm(X_hat.transpose(1,2)-X)/torch.norm(X),10)
    
    loss.backward()
    optimizer.step()
    if i%500==0:
        NMSE_X_list.append(NMSE_X)
        NMSE_Y_list.append(NMSE_Y)
        print("[Epoch{:d}/{:d}] [NMSE_X:{:.5f}] [NMSE_Y:{:.5f}]".format(i+1,opt.n_epochs,NMSE_X,NMSE_Y))
        
os.makedirs("./result/{0}_N:{1}_M:{2}_L:{3}_pro:{4}_SNR:{5}".format(opt.name,opt.N,opt.M,opt.L,opt.pro,opt.SNR), exist_ok=True) 
file_name="./result/{0}_N:{1}_M:{2}_L:{3}_pro:{4}_SNR:{5}".format(opt.name,opt.N,opt.M,opt.L,opt.pro,opt.SNR)
torch.save(NMSE_X_list,'{}/NMSE_X_list.pt'.format(file_name))      
torch.save(NMSE_Y_list,'{}/NMSE_Y_list.pt'.format(file_name))

    