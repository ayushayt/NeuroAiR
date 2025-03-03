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
from tensorflow.keras.layers import concatenate, Lambda, Conv2D, MaxPooling2D, GlobalAveragePooling2D,LSTM,Activation,Reshape,Conv2DTranspose
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
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
import time
from minnet_loss import mean_squared_error, triplet_loss, SparseCategoricalCrossentropy
from minnet_utils import TimeHistory, compute_class_weight
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, f1_score
import glob





os.environ["CUDA_VISIBLE_DEVICES"]="0"

subjectaccarr = []

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
            
            X_train = np.zeros((2600,1,diff,31))
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
                    X_train[trainctr,0,:,i] = (eeg_sig_cut[:,i])
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
                
                X_train_net, X_val_net, Y_train_net, Y_val_net = train_test_split(X_train_final, Y_train_final, test_size=0.2, random_state=42)
                
                
                tensorflow.keras.backend.clear_session()
                
                input_shape=(1,1500,31)
                encoder_input  = Input((1,1500,31))
                en_conv        = Conv2D(31, (1, 64), activation='elu', padding="same", 
                                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(encoder_input)
                en_conv        = BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1)(en_conv)
                en_conv        = AveragePooling2D(pool_size=(1,1500//100))(en_conv)  
                en_conv        = Conv2D(10, (1, 32), activation='elu', padding="same", 
                                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(en_conv)
                en_conv        = BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1)(en_conv)
                en_conv        = AveragePooling2D(pool_size=(1,4))(en_conv)
                en_conv        = Flatten()(en_conv)
                encoder_output = Dense(64, kernel_constraint=max_norm(0.5))(en_conv)
                encoder        = Model(inputs=encoder_input, outputs=encoder_output, name='encoder')
                
                'decoder'
                decoder_input  = Input(shape=(64,), name='decoder_input')
                de_conv        = Dense(1*(1500//(1500//100)//(4))*10, activation='elu', 
                                       kernel_constraint=max_norm(0.5))(decoder_input)
                de_conv        = Reshape((1, (1500//(1500//100)//(4)), 10))(de_conv)
                de_conv        = Conv2DTranspose(filters=10, kernel_size=(1, 64), 
                                                 activation='elu', padding='same', strides=(1,4), 
                                                 kernel_constraint=max_norm(2., axis=(0, 1, 2)))(de_conv)
                decoder_output = Conv2DTranspose(filters=31, kernel_size=(1, 32), 
                                                 activation='elu', padding='same', strides=(1,1500//100), 
                                                 kernel_constraint=max_norm(2., axis=(0, 1, 2)))(de_conv)
                decoder        = Model(inputs=decoder_input, outputs=decoder_output, name='decoder')
    
                latent         = encoder(encoder_input)
                train_xr       = decoder(latent)
                z              = Dense(26, activation='softmax', kernel_constraint=max_norm(0.5), 
                                       name='classifier')(latent)
                
                model =  Model(inputs=encoder_input, outputs=[train_xr, latent, z],name='MIN2Net')
                
                checkpointer  = ModelCheckpoint(monitor='val_loss', save_best_only=True, save_weight_only=True,filepath='out_weights.h5')
                
                reduce_lr     = ReduceLROnPlateau(monitor='val_loss', patience=20,factor=0.5, mode='min', verbose=0,min_lr=1e-3)
                
                es            = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20)
                
                #print(model.summary())
                
                model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss=[mean_squared_error, triplet_loss(margin=1.0), 'sparse_categorical_crossentropy'], metrics='accuracy', loss_weights=[1., 1., 1.])
                
                
                
                model.fit(x=X_train_net, y=[X_train_net,Y_train_net,Y_train_net],batch_size=128,epochs=500, validation_data=(X_val_net, [X_val_net,Y_val_net,Y_val_net]),callbacks=[checkpointer,reduce_lr,es],verbose=0)
                
                
                loss, decoder_loss, trip_loss, classifier_loss, decoder_acc, trip_acc, classifier_acc  = model.evaluate(x=X_test_final,y=[X_test_final,Y_test_final,Y_test_final],batch_size=128,verbose=0)
                            
                acc = acc + classifier_acc*100
                #print(classifier_acc*100)
            subjectaccarr.append(acc/10)
            print('Accuracy Subject '+ str(subj_index) + ': ' + str(acc/10)) 

