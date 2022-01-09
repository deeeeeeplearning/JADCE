import os
import numpy as np
import math
import sys
import time
from datetime import timedelta
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import random
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import *
from options import *
from torch.autograd import Variable
print(torch.__version__)


if __name__ == '__main__':

    opt = Options().parse()  
    print(opt)
    #torch.cuda.set_device(opt.device_id)
    X_shape = (opt.batch_size,opt.M, 2*opt.N)
    Y_shape = (opt.batch_size, 2*opt.L,opt.M)
    cuda = True if torch.cuda.is_available() else False
    opt.manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    opt.name=opt.name+'_'+str(opt.manualSeed)
    print(opt.name)
    #define model
    Unet_block_list=[]
    for i in range(opt.loop):
        tmp=nn.DataParallel(Unet_block(X_shape))
        tmp=tmp.cuda()
        Unet_block_list.append( tmp)
    discriminator = Discriminator(X_shape,opt.wgan).cuda()
    optimizer=[]
    for i in range(opt.loop):
        optimizer.append(torch.optim.Adam(Unet_block_list[i].parameters(), lr=opt.lr,))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr,)#weight_decay=0.001)

    # scheduler_2= lr_scheduler.ReduceLROnPlateau(optimizer_2,'min',patience=5,factor=0.5,threshold=0.01)
    # scheduler_3= lr_scheduler.ReduceLROnPlateau(optimizer_3,'min', patience=5,factor=0.5,threshold=0.01)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    batches_done = 0
    NMSE_list=[]
    NMSE_list_test=[]
    Softmax=nn.Softmax(dim=1).cuda()
    N,M,L=opt.N,opt.M,opt.L
    if opt.trained_P:
        print('pretrained_P')
        P=torch.load('./pt_file/Trained_Pilot_matrix/P_trained.pt').data.cpu()
        P_torch=torch.cat((torch.cat((P.real,-P.imag),dim=1),torch.cat((P.imag,P.real),dim=1)),dim=0)
    else:    
        sigmma = np.sqrt(0.5)
        P = np.vectorize(complex)(np.random.normal(size=(L, N), scale=sigmma).astype(np.float32), 
                  np.random.normal(size=(L, N), scale=sigmma).astype(np.float32))
        P = P / np.linalg.norm(P, axis=1, keepdims=True)
        P_real = np.vstack((np.hstack((P.real, -P.imag)), np.hstack((P.imag, P.real))))
        P_torch=torch.from_numpy(P_real).type(torch.FloatTensor)
    P_torch=P_torch.cuda()
    Pg=torch.pinverse(P_torch)
    Pg=Pg.cuda()
    PgP=torch.matmul(Pg,P_torch)
    I_PgP=torch.eye(*PgP.shape).cuda()-PgP
    numpy_data=False
    # loss function 
    adversarial_loss = torch.nn.BCELoss()
    valid = Variable(Tensor(64, 1).fill_(1.0), requires_grad=False)
    fake = Variable(Tensor(64, 1).fill_(0.0), requires_grad=False)
    start=time.time()
    # ----------
    #  Train and Test
    # ----------
    for i in range(opt.loop):
        optimizer=[]
        for k in range(i+1):
            optimizer.append(torch.optim.Adam(Unet_block_list[k].parameters(), lr=opt.lr,))
        if 1:
            for epoch in range(opt.n_epochs):
                #decay lr
                if epoch==opt.n_epochs/3:
                    print('decay1')
                    for j in range(i+1):
                        optimizer[j]=torch.optim.Adam(Unet_block_list[j].parameters(), lr=opt.lr*0.2,)
                if epoch==opt.n_epochs/3*2:
                    print('decay2')
                    for j in range(i+1):
                        optimizer[j]=torch.optim.Adam(Unet_block_list[j].parameters(), lr=opt.lr*0.02,)          
                #generative data
                if numpy_data:
                    Y,X=gen_data_np(L,N,M,opt.batch_size,P,opt.pro,opt.SNR)
                    Y=torch.from_numpy(Y).type(torch.FloatTensor).cuda()
                    X=torch.from_numpy(X).type(torch.FloatTensor).cuda()
                else:
                    Y,X,_=gen_data_tensor(L,N,M,opt.batch_size,P_torch,opt.pro,opt.SNR)
                X_hat=torch.matmul(Pg,Y)
                #train generator
                for j in range(i+1):
                    X_hat=X_hat+torch.matmul(I_PgP,Unet_block_list[j](X_hat))
                    optimizer[j].zero_grad()
                MSE=torch.norm(X_hat-X)**2/opt.batch_size  
                Y_hat=torch.matmul(P_torch,X_hat)
                NMSE=20*math.log(torch.norm(X_hat-X)/torch.norm(X),10)
                if opt.wgan:
                    g_loss=-opt.weight1*torch.mean(discriminator(X_hat))+MSE
                else:
                    g_loss = opt.weight1*adversarial_loss(discriminator(X_hat), valid)+MSE
                g_loss.backward()
                if epoch<opt.n_epochs/3:
                    optimizer[i].step()
                else:
                    for j in range(i+1):
                        optimizer[j].step()                    
                # train discirmator
                if epoch%3==0:
                    optimizer_D.zero_grad()
                    if opt.wgan:
                        real_loss=-torch.mean(discriminator(X))
                        fake_loss=torch.mean(discriminator((X_hat).detach()))
                    else:
                        real_loss = adversarial_loss(discriminator(X), valid)
                        fake_loss = adversarial_loss(discriminator((X_hat).detach()), fake)
        
                    d_loss = (real_loss + fake_loss) / 2
                    d_loss.backward()
                    optimizer_D.step()
                    if opt.wgan:
                        for p in discriminator.parameters():
                            p.data.clamp_(-opt.clip_value, opt.clip_value)
                #print log
                if epoch%100==0:
                    Y_test,X_test,_=gen_data_tensor(L,N,M,opt.batch_size,P_torch,opt.pro,opt.SNR)
                    X_hat_test=torch.matmul(Pg,Y_test)
                    for j in range(i+1):
                        X_hat_test=X_hat_test+torch.matmul(I_PgP,Unet_block_list[j](X_hat_test))
                    NMSE_test=20*math.log(torch.norm(X_hat_test-X_test)/torch.norm(X_test),10)
                    NMSE_list_test.append(NMSE_test)
                    print("[layer{:d}/{:d}][Epoch{:d}/{:d}] [NMSE:{:.5f}] [MSE:{:.5f}] [g_loss:{:.5f}] [d_loss:{:.5f}]][real_loss:{:.5f}] [fake_loss:{:.5f}] ".format(i+1,opt.loop,epoch,opt.n_epochs,NMSE,MSE,g_loss,d_loss,real_loss.data,fake_loss.data))
    end=time.time()            
    elapsed = end - start
    print("elapsed time of training = " + str(timedelta(seconds=elapsed)))
    #save NMSE    
    os.makedirs("./result/{0}_N:{1}_M:{2}_L:{3}_pro:{4}_SNR:{5}".format(opt.name,opt.N,opt.M,opt.L,opt.pro,opt.SNR), exist_ok=True)  
    torch.save(NMSE_list_test,'./result/{0}_N:{1}_M:{2}_L:{3}_pro:{4}_SNR:{5}/NMSE_list_test.pt'.format(opt.name,opt.N,opt.M,opt.L,opt.pro,opt.SNR))
    #save model 
    for i in range(opt.loop):
        torch.save(Unet_block_list[i],"./result/{0}_N:{1}_M:{2}_L:{3}_pro:{4}_SNR:{5}/Unet_block_{6}.pkl".format(opt.name,opt.N,opt.M,opt.L,opt.pro,opt.SNR,i+1)) 
    torch.save(discriminator,"./result/{0}_N:{1}_M:{2}_L:{3}_pro:{4}_SNR:{5}/discriminator.pkl".format(opt.name,opt.N,opt.M,opt.L,opt.pro,opt.SNR))
       
    

