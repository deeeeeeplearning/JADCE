import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


################ basic components  #######################
class DoubleConv(nn.Module):
    """(convolution => => ReLU) * 2"""
    #Size does not change
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with con1d and then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1,stride=2),
            DoubleConv(in_channels, out_channels)
            
        )

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels , in_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2,cat=True):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
    
    
######################## network ####################################
class Encoder_A(nn.Module):
    # design A
    def __init__(self,L,N,SNR=10):
        super(Encoder_A,self).__init__()
        A_real = torch.randn((L,N),requires_grad=True)
        A_imag = torch.randn((L,N),requires_grad=True)
        self.A_real = nn.Parameter(A_real)
        self.A_imag = nn.Parameter(A_imag)
        self.L=L
        self.N=N
        self.SNR=SNR
        
    def forward(self,X):
        A=torch.complex(self.A_real,self.A_imag)#复数
        Y=torch.matmul(A,X)
        #sigmma=torch.std(Y)*10**(-self.SNR/20)
        Z=torch.complex(torch.randn(Y.shape),torch.randn(Y.shape)).cuda()*0.1#*sigmma
        return A,Z

#Unet block
class Unet_block(nn.Module):#Y B*C*M*2L
    
    def __init__(self,X_shape):
        super(Unet_block,self).__init__()
        B,C,N=X_shape#复数已经变为实数操作 M*2N
        self.inc = DoubleConv(C, 64)
        self.down1 = Down(64, 128)#尺寸减半
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 1 #if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.down5 = Down(1024, 2048 // factor)
        self.up0 = Up(2048, 1024 // factor,)
        self.up1 = Up(1024, 512 // factor,)
        self.up2 = Up(512, 256 // factor, )
        self.up3 = Up(256, 128 // factor,)
        self.up4 = Up(128, 64, )
        self.outc = OutConv(64, C)#1*1 卷积



    def forward(self, x):
        x=x.transpose(1,2)#变为M*N
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        #x6 = self.down5(x5)
        #x = self.up0(x6, x5)
        x = self.up1(x5,x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        X = self.outc(x)
        return X.transpose(1,2)#变回N*M

#  conv1d Discriminator
class Discriminator(nn.Module):
    def __init__(self,X_shape,wgan=True):
        super(Discriminator,self).__init__()
        B,C,N=X_shape#M当做信道
        self.inc = DoubleConv(C, 64)
        self.down1 = Down(64, 128)#尺寸减半N
        self.down2 = Down(128, 256)#N/2
        self.down3 = Down(256, 512)#N/4
        self.outc = OutConv(512, 128)
        if wgan:
            print('wgan')
            self.adv_layer = nn.Sequential(nn.Linear(16*N, 1))##, nn.Sigmoid())#取消变为wgan
        else:
            print('gan')
            self.adv_layer = nn.Sequential(nn.Linear(16*N, 1), nn.Sigmoid())                               
    def forward(self,X):
        X=X.transpose(1,2)
        X=self.inc(X)
        X=self.down1(X)
        X=self.down2(X)
        X=self.down3(X)
        X=self.outc(X)
        X=X.view(X.shape[0],-1)
        X=self.adv_layer(X)
        return X
  
    
# linear  Discriminator
class Discriminator_linear(nn.Module):
    def __init__(self,X_shape):
        super(Discriminator_linear, self).__init__()
        B,C,N=X_shape
        self.model = nn.Sequential(
            nn.Linear(int(C*N), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, X):
        X = X.view(X.shape[0], -1)
        validity = self.model(X)
        return validity
    

    
######################

