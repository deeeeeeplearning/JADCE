import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Generator(nn.Module):
	def __init__(self,X_shape,z_dim):
		super(Generator,self).__init__()
		self.X_shape=X_shape
		B,H,W=X_shape
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
					#nn.BatchNorm2d(hidden_dims[i + 1]),
					nn.LeakyReLU()))
		modules.append(nn.Sequential(nn.ConvTranspose2d(128,
										64,
										kernel_size=3,
										stride = 2,
										padding=1,
										output_padding=1),
					#nn.BatchNorm2d(64),
					nn.LeakyReLU(),
					nn.Conv2d(64,1,kernel_size= 3, padding= 1),))
		self.decoder=nn.Sequential(*modules)

	def forward(self,z):
		B,H,W=self.X_shape      
		result = self.decoder_input(z)
		result = result.view(-1, 256, int(H/4), int(W/4))
		result =self.decoder(result)
		result=result.squeeze()
		return result


