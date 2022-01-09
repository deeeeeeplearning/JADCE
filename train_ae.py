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
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--z_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--N", type=int, default=256, help="number of devices")
parser.add_argument("--pro", type=int, default=0.1, help="number of activity")
parser.add_argument("--M", type=int, default=8, help="size of antenna")
parser.add_argument("--L", type=int, default=128, help="size of sequense")
parser.add_argument("--SNR", type=float, default='30', help="var of noise")
parser.add_argument("--ngf", type=int, default=2, help="parameter of generator")
parser.add_argument("--lambda1", type=float, default=2, help="parameter of lambda")
parser.add_argument("--name", type=str, default='Auto_encoder', help="filename")
parser.add_argument("--device_id", type=int, default='1', help="device_id")
opt = parser.parse_args()
print(opt)
torch.cuda.set_device(opt.device_id)
X_shape = (opt.batch_size,1, opt.M, 2*opt.N)
Y_shape = (opt.batch_size,1, opt.M, 2*opt.L)
cuda = True if torch.cuda.is_available() else False
opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
opt.name=opt.name+'_'+str(opt.manualSeed)
opt.name=opt.name+'_'+str(opt.lambda1)
print(opt.name)
# Initialize generator and discriminator
encodeX2Y = Encoder_X2Y(opt.N,opt.L)#输入转置的X bs M，2N 输出bs M，2L
encoder_Y2z = Encoder_Y2z(Y_shape,opt.z_dim)#输入bs M，2L 输出 bs z_dim
decoder=Decoder(X_shape,opt.z_dim)#输入 bs z_dim 输出 bs,M，2N
if cuda:
    encodeX2Y.cuda()
    encoder_Y2z.cuda()
    decoder.cuda()
# Optimizers
optimizer_1 = torch.optim.Adam(encodeX2Y.parameters(), lr=opt.lr)
optimizer_2 = torch.optim.Adam(encoder_Y2z.parameters(), lr=opt.lr)
optimizer_3 = torch.optim.Adam(decoder.parameters(), lr=opt.lr)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# ----------
#  Training
# ----------
batches_done = 0
NMSE_list=[]
Softmax=nn.Softmax(dim=2).cuda()
for epoch in range(opt.n_epochs):
    for i in range(36):     
        X,active=gen_data_tensor_X(opt.L,opt.N,opt.M,opt.batch_size,opt.pro,opt.SNR)
        
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        optimizer_3.zero_grad()
        # Configure input
        Y=encodeX2Y(X.transpose(1,2))
        sigmma=torch.std(Y)*10**(-opt.SNR/20)
        noise=torch.randn(Y.shape).cuda()*sigmma
        Y_nosie=Y+noise
        z_hat,_,_=encoder_Y2z(Y_nosie)
        X_hat=decoder(z_hat)
        X_hat=X_hat.transpose(1,2)

        loss=torch.norm(X_hat-X)**2/opt.batch_size
        NMSE=20*math.log(torch.norm(X_hat-X)/torch.norm(X),10)
        loss.backward()
        optimizer_1.step()
        optimizer_2.step()
        optimizer_3.step()
        NMSE_list.append(NMSE)
        if i%100==0:
            print("[Epoch{:d}/{:d}] [NMSE:{:.5f}] [MSE:{:.5f}] ".format(epoch,opt.n_epochs,NMSE,loss))      
os.makedirs("./result/{0}_N:{1}_M:{2}_L:{3}_pro:{4}_SNR:{5}".format(opt.name,opt.N,opt.M,opt.L,opt.pro,opt.SNR), exist_ok=True) 
file_name="./result/{0}_N:{1}_M:{2}_L:{3}_pro:{4}_SNR:{5}".format(opt.name,opt.N,opt.M,opt.L,opt.pro,opt.SNR)
torch.save(NMSE_list,'{}/NMSE_list.pt'.format(file_name))
torch.save(encodeX2Y, '{}/encodeX2Y.pkl'.format(file_name))
torch.save(encoder_Y2z, '{}/encoder_Y2z.pkl'.format(file_name)) 
torch.save(decoder, '{}/decoder.pkl'.format(file_name))        

