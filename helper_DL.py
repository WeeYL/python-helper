#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')
from helper import *
import pandas as pd 
from helper import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset


# In[2]:


def dl_back_prop(optimizer,loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# In[2]:


def dl_create_torch_df(li,colname,rowsToCols=True):
    '''
    li takes 2-D matrix of torch elements, 
    colname is a list of col names
    '''
    
    data=[]
    for n in li: 
        res=n.detach()              # detach from list 
        res=[n.item() for n in res] # convert tensor to value
        data.append(res)

    '''rows to columns'''
    df=pd.DataFrame(data)
    if rowsToCols == True:
        df=df.transpose()
    
    for k,v in zip(df.columns,colname):
        df.rename(columns = {k : v}, inplace = True)
    
    return df


# In[3]:


def dl_create_label_encoder(target, returnUniqueLabels=False):
    '''takes in a list of label names strings and labelencode it, 
    labels = dl_create_label_encoder(iris['species'])
    labels, names = dl_create_label_encoder(iris['species'], True)
    '''
    from sklearn.preprocessing import LabelEncoder
    
    labels = target
    labelencoder = LabelEncoder()
    labels=torch.tensor(labelencoder.fit_transform(labels))
    labels=labels.long()
    unique_labels = labelencoder.inverse_transform(range(len(set(target))))
    
    if returnUniqueLabels:
        return labels,unique_labels
    return labels


# In[4]:


'''
ANNModel cross entropy
'''
def dl_create_ANN_crossentropy_model(nInput,nOutput,nHidden,lr=0.01):
    '''
    Two hidden Layer for Cross Entropy
    eg = ANN,lossfun,optimizer = dl_create_ANN_crossentropy_model(nInput=4,nOutput=3,nHidden=64,lr=0.01)
    '''
    ANN = nn.Sequential(
      nn.Linear(nInput,nHidden),      
      nn.ReLU(),                
      nn.Linear(nHidden,nHidden),
      nn.ReLU(),                
      nn.Linear(nHidden,nOutput),      
        )
    # loss function
    lossfun = nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.SGD(ANN.parameters(),lr=lr)
    return ANN,lossfun,optimizer



def dl_create_train_model(model,lossfun,optimizer, numepochs, data, labels):
    '''Train model and returns losses and accuracy for each epoch
    eg: losses, accuracy = dl_train_model(model,lossfun,optimizer, numepochs, data, labels
    '''
    # initialize losses
    losses = torch.zeros(numepochs)
    ongoingAcc = []
    # loop over epochs
    for epochi in range(numepochs):
        # forward pass
        yHat = model(data)
        # compute loss
        loss = lossfun(yHat,labels)
        losses[epochi] = loss
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # compute accuracy
        yHat=torch.argmax(yHat,axis=1)
        accuracy=torch.mean((yHat == labels).float()).item() * 100
        ongoingAcc.append(np.round(accuracy,3))

        
    return losses, ongoingAcc


# In[2]:


def dl_create_train_test_batches(data, labels, batch_size=12, train_size=0.8):
    '''
    Takes in data and label (-1,1) tensor matrix
    eg, 
    train_batches, test_batches, batch_info = dl_create_train_test_batches(data, labels, batch_size, train_size=0.8)
    
    eg to get tensor from train_baches: for X,y in train_batches
    '''
    '''setup split data'''
    train_data,test_data, train_labels,test_labels = train_test_split(data, labels, train_size=train_size)

    '''create a tuple (data, label)''' 
    train_dataT = TensorDataset(train_data,train_labels)
    test_dataT  = TensorDataset(test_data,test_labels)

    '''
    package into batches
    training set is in batches for computation efficiency
    test set is in single batch, to test each data 
    '''
    train_batches = DataLoader(train_dataT,batch_size,shuffle=True,drop_last=True) 
    test_batches  = DataLoader(test_dataT,batch_size=test_dataT.tensors[0].shape[0])
    X_batches_shape = (len(train_batches), batch_size)
    for x,y in train_batches:
        X_train_batch_shape = x.shape
        X_test_batch_shape = y.shape 
        break
    info = {'X_batch_shape':X_batches_shape,
            'X_train_batch_shape':X_train_batch_shape, # each batch shape
            'X_test_batch_shape':X_test_batch_shape,
            'train_data':train_data,
            'test_data':test_data, 
            'train_labels':train_labels,
            'test_labels':test_labels}
    return train_batches, test_batches, info


# In[ ]:


def dl_test_accuracy_from_trained_model(trained_model,test_batches):
    '''
    PERFORM ACCURACY TEST FROM TRAINED MODEL
    eg, test_acc = dl_test_accuracy_from_trained_model(trained_model,test_batches)
    '''
    X,y = next(iter(test_batches))
    with torch.no_grad(): # deactivates autograd

        # PREDICTION
        yHat=trained_model(X)
        # RESULT
        matches = torch.argmax(yHat,axis=1) == y  
    return 100*torch.mean(matches.float())

