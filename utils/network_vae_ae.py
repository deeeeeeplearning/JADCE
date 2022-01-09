import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Encoder_X2Y(nn.Module):
	def __init__(self,N,L):
		super(Encoder_X2Y,self).__init__()
		self.model=nn.Sequential(
			nn.Linear(2*N,2*L,bias=False))
	def forward(self,X):
		Y=self.model(X)
		return Y
#经过矩阵乘法 所以用线性层

class Encoder_Y2z(nn.Module):
	def __init__(self,Y_shape,z_dim):
		super(Encoder_Y2z,self).__init__()
		B,C,H,W=Y_shape
		hidden_dims = [64,128]
		modules=[]
		in_channels=1
		for i in hidden_dims:
			modules.append(
				nn.Sequential(nn.Conv2d(in_channels ,out_channels=i, kernel_size= 3, stride= 2, padding  = 1),
				nn.BatchNorm2d(i),
				nn.LeakyReLU()))
			in_channels=i
		modules.append(nn.Sequential(nn.Conv2d(128 ,out_channels=256, kernel_size= 3, stride= 1, padding  = 1),
				nn.BatchNorm2d(256),
				nn.LeakyReLU()))
		self.model_conv = nn.Sequential(*modules)

		self.model_2_mu=nn.Sequential(
			nn.Linear(16*H*W,z_dim))

		self.model_2_sigma=nn.Sequential(
			nn.Linear(16*H*W,z_dim))

	def forward(self,Y):
		Y=torch.unsqueeze(Y,1)#增加通道数据
		result=self.model_conv(Y)       
		result = torch.flatten(result, start_dim=1)
		mu=self.model_2_mu(result)
		log_var =self.model_2_sigma(result)
		std = torch.exp(log_var/2)
		eps = torch.randn_like(std)
		z=mu + eps * std
		return z,mu,log_var
		#z=z.view(z.shape[0],z.shape[1],1,1)
		



class Decoder(nn.Module):
	def __init__(self,X_shape,z_dim):
		super(Decoder,self).__init__()
		self.X_shape=X_shape
		B,C,H,W=X_shape
		self.decoder_input = nn.Linear(z_dim,16*H*W)
		modules=[]
		hidden_dims = [256,128]
		for i in range(len(hidden_dims)-1):
			modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i],
										hidden_dims[i + 1],
										kernel_size=3,
										stride = 2,
										padding=1,
										output_padding=1),
					nn.LeakyReLU()))
		modules.append(nn.Sequential(nn.ConvTranspose2d(128,
										64,
										kernel_size=3,
										stride = 2,
										padding=1,
										output_padding=1),
					nn.LeakyReLU(),
					nn.Conv2d(64,1,kernel_size= 3, padding= 1)))
		self.decoder=nn.Sequential(*modules)

	def forward(self,z):
		B,C,H,W=self.X_shape      
		result = self.decoder_input(z)
		result = result.view(-1, 256, int(H/4), int(W/4))
		result =self.decoder(result)
		result=result.squeeze()
		return result





class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
    
class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """
    def __init__(self, N=-1, C=2, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)