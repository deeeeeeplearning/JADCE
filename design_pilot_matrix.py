import numpy as np
import torch
import torch.nn as nn
import argparse
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--N", type=int, default=256, help="number of devices")
parser.add_argument("--pro", type=float, default=0.1, help="number of activity")
parser.add_argument("--M", type=int, default=8, help="size of antenna")
parser.add_argument("--L", type=int, default=128, help="size of sequense")
parser.add_argument("--name", type=str, default='trained_P', help="filename")
parser.add_argument("--device_id", type=int, default='0', help="device_id")
parser.add_argument("--SNR", type=float, default='0', help="var of noise")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False
torch.cuda.set_device(opt.device_id)
encoder_A=Encoder_A(opt.L,opt.N,opt.SNR).cuda()

optimizer=torch.optim.Adam(encoder_A.parameters(), lr=opt.lr,)

for i in range(opt.n_epochs):
    dataset=torch.randn(opt.batch_size,2,opt.N,opt.M).cuda()*0.5**0.5
    temp=torch.rand(opt.batch_size,opt.N).cuda()<opt.pro
    temp=temp+0
    bernoulli=torch.zeros(opt.batch_size,2,opt.N,opt.N).cuda()
    for k in range(opt.batch_size):
        bernoulli[k][0]=temp[k].diag()
    bernoulli[:,1,:,:]=bernoulli[:,0,:,:]  
    dataset=torch.matmul(bernoulli,dataset)#生成稀疏正态分布数据        
    X=torch.complex(dataset[:,0,:,:], dataset[:,1,:,:])#全员复数！
    A,Z=encoder_A(X)
    Ag=torch.linalg.pinv(A)
    norm=torch.norm(torch.matmul(Ag,Z))
    if i%100 ==0:
        std1=torch.std(torch.matmul(Ag,Z))
        std2=torch.std(X)
        SNR=(std2/std1)**2
        print('i:{},norm:{},SNR:{}'.format(i,norm,SNR))
    optimizer.zero_grad()
    norm.backward()
    for name, parms in encoder_A.named_parameters():
        if torch.any(torch.isnan(parms.grad)):
            print('Grad data has NaN!')
            optimizer.zero_grad()
            continue

    optimizer.step()
torch.save(A,'./P_trained.pt')