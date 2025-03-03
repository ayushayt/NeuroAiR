import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import math
from scipy import signal
import scipy.io
from scipy import signal
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.utils import shuffle
from copy import deepcopy
import time

import argparse
import os
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import datetime
import sys
import scipy.io

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary
import torch.autograd as autograd
from torchvision.models import vgg19

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
# from common_spatial_pattern import csp

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn

device = 'cuda:0'

# Convolution module
# use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (31, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            #nn.Linear(2440, 256),
            nn.Linear(3760, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 26)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=26, **kwargs):
        super().__init__(

            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )
        
subjectaccarr = []
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
for subj_index in range(1,11):

    print('Subject No. ' + str(subj_index))
    lenarr = []
    eeg = pd.read_csv('Subject' + str(subj_index) + '/Data/EEG_Combined.txt',sep='\t')
    eeg = eeg.to_numpy()
    event_file = pd.read_csv('Subject' + str(subj_index) + '/Data/Events_Combined.txt',sep='\t')
    
    
    timestamp = event_file['latency'].to_numpy()
    event = event_file['type'].to_numpy()
    
    event_arr = []
    timestamp_arr = []
    
    for i in range(0,len(event)):
        if event[i]!='boundary':
            event_arr.append(event[i])
            timestamp_arr.append(timestamp[i])
    
    for i in range(0,len(event_arr)):
        
        if event_arr[i]!='PP':
        
            start = timestamp_arr[i]
            stop = timestamp_arr[i+1]
            eeg_sig = eeg[int(start-500):int(stop),:]
            trial = int(i//52)
            #print(trial)
            np.save('Subject' + str(subj_index) + '/Segmented/EEG_'+event_arr[i]+'_'+str(trial)+'.npy',eeg_sig)
            #print(len(eeg_sig))
            lenarr.append(len(eeg_sig))
    
    #print(np.mean(lenarr))
    #print(np.max(lenarr))
    #print(np.min(lenarr))
    
    start = [0]
    end  = [1500]
    
    start_arr = []
    end_arr = []
    acc_arr = []
    
    for i1 in range(1):
        for j1 in range(1):
            
            st = start[i1]
            en = end[j1]
            diff = en-st
            
            X_train = np.zeros((2600,1,31,diff))
            Y_train = np.zeros((2600,))
            
            trainctr = 0
                      
            for fi in os.listdir('Subject' + str(subj_index) + '/Segmented/'):
                eeg_sig = np.load(os.path.join('Subject' + str(subj_index) + '/Segmented/',fi))
                eeg_sig_var = np.zeros((1500,31))
                
                if len(eeg_sig)<=1500:
                    eeg_sig_var[0:len(eeg_sig),:] = eeg_sig
                else:
                    eeg_sig_var = eeg_sig[0:1500,:]
                
                eeg_sig_cut = eeg_sig_var[st:en,:]
                
                _,char,trial = fi.split('_')
                trial,_ = trial.split('.')
                
                for i in range(0,31):
                    X_train[trainctr,0,i,:] = np.transpose(eeg_sig_cut[:,i])
                Y_train[trainctr] = ord(char)-65
                trainctr = trainctr+1
            
            for i in range(0,len(X_train)):
                X_train[i,0,:,:] = (X_train[i,0,:,:]-np.mean(X_train[i,0,:,:]))/np.std(X_train[i,0,:,:])
            
                
            kf = KFold(n_splits=10,shuffle=True,random_state=1)
            kf.get_n_splits(X_train)
            acc = 0
            
            for train_index,test_index in kf.split(X_train):
                
                X_train_final, X_test_final = X_train[train_index],X_train[test_index] 
                Y_train_final, Y_test_final = Y_train[train_index],Y_train[test_index] 
                
                X_train_split, X_val, Y_train_split, Y_val = train_test_split(X_train_final, Y_train_final, test_size=0.2, random_state=42)
                
                model = Conformer().to(device)
                #print(summary(model, (1, 31, 1500)))
                
                criterion = torch.nn.CrossEntropyLoss()
    
                optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
                
                best_val_acc = -1000
                best_val_model = None
                batch_size = 128
                val_acc_arr = -1000*np.ones((10,))
                es_patience=10+1
                num_epochs = 1000
                
                #print('Started training')
                
                for epoch in range(0,num_epochs):  
                    model.train(True)
                    running_loss = 0.0
                    running_acc = 0
                    running_val_loss = 0.0
                    for i in range(0,len(X_train_split)//batch_size):
                        inputs = torch.tensor(X_train_split[i*batch_size:(i+1)*batch_size,:,:,:], dtype=torch.float).to(device)
                        labels = torch.tensor(Y_train_split[i*batch_size:(i+1)*batch_size,], dtype=torch.long).to(device)
                
                        optimizer.zero_grad()
                        temp,outputs = model(inputs)
                        #print(temp.shape)
                        #print(outputs.shape)
                        #print(labels.shape)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                
                        running_loss += loss.item() * inputs.size(0)
                        out = torch.argmax(outputs.detach(),dim=1)
                        assert out.shape==labels.shape
                        running_acc += (labels==out).sum().item()
                
                    correct = 0
                    model.train(False)
                    with torch.no_grad():
                        for i in range(0,len(X_val)//batch_size):
                            inputs = torch.tensor(X_val[i*batch_size:(i+1)*batch_size,:,:,:], dtype=torch.float).to(device)
                            labels = torch.tensor(Y_val[i*batch_size:(i+1)*batch_size,], dtype=torch.long).to(device)
                            temp,out = model(inputs)
                            #print(temp.shape)
                            #print(out.shape)
                            #print(labels.shape)
                            out = torch.argmax(out,dim=1)
                            acc = (out==labels).sum().item()
                            correct += acc
    
                    #print(f"Train loss {epoch+1}: {running_loss/len(X_train_split)},Train Acc:{running_acc*100/len(X_train_split)},Validation Acc:{correct*100/len(X_val)}%")
                    
                    if correct>best_val_acc:
                        best_val_acc = correct
                        best_val_model = deepcopy(model.state_dict())
                        
                    val_acc_arr = np.concatenate((val_acc_arr,np.asarray(np.reshape(correct*100/len(X_val),(1,)))))
                    if val_acc_arr[-es_patience] == np.max(val_acc_arr[-es_patience:]):
                        break
                    
                #print('Finished Training')  
                
                predictions = np.zeros((len(X_test_final),26))
                model.load_state_dict(best_val_model)
                model.train(False)
                
                for i in range(0,len(X_test_final)):
                    temp,test_preds = model(torch.tensor(np.reshape(X_test_final[i,:,:,:],(1,1,31,1500)), dtype=torch.float).to(device))
                    prob = torch.nn.functional.softmax(test_preds, dim=1)
                    prob = prob.cpu().detach().numpy()
                    predictions[i] = prob
                    
                Y_pred = np.argmax(np.asarray(predictions),axis=1)
                #print(accuracy_score(Y_pred,Y_test_final)*100)
                acc_arr.append(accuracy_score(Y_pred,Y_test_final)*100)
    
    
    print('Accuracy Subject '+ str(subj_index) + ': ' + str(np.mean(acc_arr)))           
                        