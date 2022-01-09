import torch
import numpy as np
#得到tensor 或者array形式数据
def gen_data_tensor_X(L,N,M,batch_size,pro,SNR):
    # P torch 形式
    dataset=torch.randn(batch_size,2,N,M).cuda()*0.5**0.5
    temp=torch.rand(batch_size,N).cuda()<pro
    temp=temp+0
    bernoulli=torch.zeros(batch_size,2,N,N).cuda()
    for k in range(batch_size):
        bernoulli[k][0]=temp[k].diag()
    bernoulli[:,1,:,:]=bernoulli[:,0,:,:]  
    dataset=torch.matmul(bernoulli,dataset)#生成稀疏正态分布数据        
    X=torch.cat((dataset[:,0,:,:], dataset[:,1,:,:]), 1)
    return X,temp
def gen_data_tensor(L,N,M,batch_size,P,pro,SNR):
    # P torch 形式
    dataset=torch.randn(batch_size,2,N,M).cuda()*0.5**0.5
    temp=torch.rand(batch_size,N).cuda()<pro
    temp=temp+0
    bernoulli=torch.zeros(batch_size,2,N,N).cuda()
    for k in range(batch_size):
        bernoulli[k][0]=temp[k].diag()
    bernoulli[:,1,:,:]=bernoulli[:,0,:,:]  
    dataset=torch.matmul(bernoulli,dataset)#生成稀疏正态分布数据        
    X=torch.cat((dataset[:,0,:,:], dataset[:,1,:,:]), 1)
    Y=torch.matmul(P,X)
    sigmma=torch.std(Y)*10**(-SNR/20)
    noise=torch.randn(Y.shape).cuda()*sigmma
    Y=Y+noise
    return Y,X,temp


def gen_data_np(L,N,M,batch_size,P,pro,SNR):
    # P complex形式 numpy
    # 返回 real 形式  numpy
    bernoulli = np.random.uniform (size=(N, batch_size)) <= pro
    bernoulli = bernoulli.astype (np.float32)
    for i in range(batch_size):
        A_i = np.diag(bernoulli[:,i])
        sigmma = np.sqrt(0.5)
        H_i = np.vectorize(complex)(np.random.normal(size=(N, M), scale=sigmma).astype(np.float32), 
                                    np.random.normal(size=(N, M), scale=sigmma).astype(np.float32))
        X_i = np.matmul(A_i, H_i)
        Y_i = np.matmul(P, X_i)
        #noise
        power = np.linalg.norm(Y_i)
        std_ = (np.square(power) / L / M) * np.power(10.0, -SNR/10.0)#方差
        std_ = np.maximum (std_, 10e-50)#方差
        sigmma = np.sqrt(std_ / 2)#标准差
        noise = np.vectorize(complex)(np.random.normal(size=Y_i.shape, scale=sigmma).astype(np.float32), 
                               np.random.normal(size=Y_i.shape, scale=sigmma).astype(np.float32))
        Y_i = Y_i + noise
        # change in real    
        X_i = np.vstack((X_i.real, X_i.imag))
        Y_i = np.vstack((Y_i.real, Y_i.imag))
        X_i = np.expand_dims(X_i, 0)
        Y_i = np.expand_dims(Y_i, 0)
        # stroe X, Y
        if(i == 0):
            X = X_i
            Y = Y_i
        else:
            X = np.concatenate((X, X_i))     
            Y = np.concatenate((Y, Y_i))
                
    return Y, X