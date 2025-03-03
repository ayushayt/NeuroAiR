import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import math
from scipy import signal
import scipy.io
from scipy import signal
import mne
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import accuracy_score,confusion_matrix
import tensorflow
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.layers import Dense, Activation, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Flatten, Dropout, BatchNormalization, Input,UpSampling1D
from tensorflow.keras.layers import concatenate, Lambda, Conv2D, MaxPooling2D, GlobalAveragePooling2D,LSTM,Activation
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from EEGModels import EEGNet, ShallowConvNet, DeepConvNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"]="0"

for subj in range(1,11):

    subj_folder = 'Subject' + str(subj)

    ncomparr = []
    avgaccarr = []
    
    for ncomp in range(1,32):
    
        #print('Number of components: ' + str(ncomp))
    
        eeg = pd.read_csv(subj_folder+'/Data/ICA_Combined.txt',sep='\t')
        eeg = eeg.to_numpy()
        event_file = pd.read_csv(subj_folder+'/Data/Events_Combined.txt',sep='\t')
        
        
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
                np.save(subj_folder+'/Segmented/EEG_'+event_arr[i]+'_'+str(trial)+'.npy',eeg_sig)
        
        #print(np.mean(lenarr))
        #print(np.max(lenarr))
        #print(np.min(lenarr))
                
        st = 0
        en = 1500
        diff = en-st
        
        X_train = np.zeros((2600,ncomp,diff,1))
        Y_train = np.zeros((2600,))
        
        trainctr = 0
                  
        for fi in os.listdir(subj_folder+'/Segmented/'):
            eeg_sig = np.load(os.path.join(subj_folder+'/Segmented/',fi))
            eeg_sig_var = np.zeros((1500,31))
            
            if len(eeg_sig)<=1500:
                eeg_sig_var[0:len(eeg_sig),:] = eeg_sig
            else:
                eeg_sig_var = eeg_sig[0:1500,:]
            
            eeg_sig_cut = eeg_sig_var[st:en,0:ncomp]
            
            _,char,trial = fi.split('_')
            trial,_ = trial.split('.')
            
            for i in range(0,ncomp):
                X_train[trainctr,i,:,0] = np.transpose(eeg_sig_cut[:,i])
            Y_train[trainctr] = ord(char)-65
            trainctr = trainctr+1
        
        for i in range(0,len(X_train)):
            X_train[i,:,:,0] = (X_train[i,:,:,0]-np.mean(X_train[i,:,:,0]))/np.std(X_train[i,:,:,0])
        
            
        kf = KFold(n_splits=10,shuffle=True,random_state=1)
        kf.get_n_splits(X_train)
        acc = 0
        
        for train_index,test_index in kf.split(X_train):
            
            X_train_final, X_test_final = X_train[train_index],X_train[test_index] 
            Y_train_final, Y_test_final = Y_train[train_index],Y_train[test_index] 
            
            
            model = EEGNet(nb_classes=26,Chans=ncomp,Samples=diff)
            model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
            es = EarlyStopping(monitor='val_accuracy', verbose=0, patience=20,restore_best_weights=True)
            model.fit(X_train_final, y=to_categorical(Y_train_final),validation_split=0.2,epochs=500, batch_size=128,verbose=0,callbacks=[es])
            pred = model.predict(X_test_final)
            Y_pred = np.argmax(pred,axis=1)
            acc = acc + accuracy_score(Y_pred,Y_test_final)*100
            #print(accuracy_score(Y_pred,Y_test_final)*100)
        #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        #print(st,en,acc/10)
        #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        ncomparr.append(ncomp)
        avgaccarr.append(acc/10)
                
    df1 = pd.DataFrame()
    
    df1['Number of Components'] = np.asarray(ncomparr)
    df1['Average Accuracy'] = np.asarray(avgaccarr)
        
    
    df1.to_csv(subj_folder+'/ICA_Num_components.csv',index=False)     
            

