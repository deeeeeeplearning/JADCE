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
parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--z_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--N", type=int, default=256, help="number of devices")
parser.add_argument("--pro", type=int, default=0.1, help="number of activity")
parser.add_argument("--M", type=int, default=8, help="size of antenna")
parser.add_argument("--L", type=int, default=128, help="size of sequense")
parser.add_argument("--SNR", type=float, default='30', help="var of noise")
parser.add_argument("--ngf", type=int, default=2, help="parameter of generator")
parser.add_argument("--lambda1", type=float, default=2, help="parameter of lambda")
parser.add_argument("--name", type=str, default='standard_GAN_without_norm', help="filename")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--device_id", type=int, default='3', help="device_id")
opt = parser.parse_args()
print(opt)
torch.cuda.set_device(opt.device_id)
X_shape = (opt.batch_size, opt.M, 2*opt.N)
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
generator=Generator(X_shape,opt.z_dim)
discriminator = Discriminator(X_shape).cuda()

if cuda:
    generator.cuda()
    discriminator.cuda()
# Optimizers
optimizer_1 = torch.optim.Adam(generator.parameters(), lr=opt.lr)
optimizer_2 = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

g_loss_list=[]
d_loss_list=[]
for epoch in range(opt.n_epochs):
    for i in range(36):     
        X,active=gen_data_tensor_X(opt.L,opt.N,opt.M,opt.batch_size,opt.pro,opt.SNR)
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        # Configure input
        z=torch.randn(opt.batch_size,opt.z_dim).cuda()
        X_hat=generator(z)
        X_hat=X_hat.transpose(1,2)
        real_loss=-torch.mean(discriminator(X))
        fake_loss=torch.mean(discriminator((X_hat).detach()))
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_2.step()

        if i%1==0:
            g_loss=-torch.mean(discriminator(X_hat))
            g_loss.backward()
            g_loss_list.append(g_loss)
            d_loss_list.append(d_loss)
            optimizer_1.step()
            if i%18==0:
                for p in discriminator.parameters():
                    p.data.clamp_(-opt.clip_value, opt.clip_value)
                print("[Epoch{:d}/{:d}]  [g_loss: {:.2f}] [d_real_loss:{:.5f}] [d_fake_loss:{:.5f}] [d_loss:{:.5f}]".format(epoch,opt.n_epochs,g_loss,real_loss,fake_loss,d_loss))
            
        
    
os.makedirs("./result/{0}_N:{1}_M:{2}_L:{3}_pro:{4}_SNR:{5}".format(opt.name,opt.N,opt.M,opt.L,opt.pro,opt.SNR), exist_ok=True) 
file_name="./result/{0}_N:{1}_M:{2}_L:{3}_pro:{4}_SNR:{5}".format(opt.name,opt.N,opt.M,opt.L,opt.pro,opt.SNR)
torch.save(g_loss_list,'{}/g_loss_list.pt'.format(file_name))
torch.save(d_loss_list,'{}/d_loss_list.pt'.format(file_name))
torch.save(generator, '{}/generator.pkl'.format(file_name))
torch.save(discriminator, '{}/discriminator.pkl'.format(file_name))       

