# import libraries
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization, GRU, Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
import pickle
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers

pd.options.display.float_format = '{:.6f}'.format
from sklearn.preprocessing import MinMaxScaler
import random


def data_preprocessing(df):
    cols = df.columns
    for i in cols:
        df[i] = df[i].astype('category')

    df['time'] = pd.to_numeric(df['time'])
    df['time_abs'] = df.time - min(df.time)

    # create features for time difference and time diff for each id
    df['time_dif'] = df['time_abs'].diff()
    df['time_dif'] = df['time_dif'].fillna(df['time_dif'].mean())
    df['ID_time_diff'] = df.groupby('id')['time_abs'].diff()
    df['ID_time_diff'] = df['ID_time_diff'].fillna(df.groupby('id')['ID_time_diff'].transform('mean'))

    # function to convert hex to int
    def hex_to_int(hex):
        val = int(hex, 16)
        return val

    # fillna payloads with 00
    dd_cols = ['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7']
    for i in dd_cols:
        df[i] = df[i].fillna('00')

    # convert hex signal to int signal
    df['id_int'] = df['id'].apply(hex_to_int).astype('int')
    df['d1_int'] = df['d0'].apply(hex_to_int).astype('int')
    df['d2_int'] = df['d1'].apply(hex_to_int).astype('int')
    df['d3_int'] = df['d2'].apply(hex_to_int).astype('int')
    df['d4_int'] = df['d3'].apply(hex_to_int).astype('int')
    df['d5_int'] = df['d4'].apply(hex_to_int).astype('int')
    df['d6_int'] = df['d5'].apply(hex_to_int).astype('int')
    df['d7_int'] = df['d6'].apply(hex_to_int).astype('int')
    df['d8_int'] = df['d7'].apply(hex_to_int).astype('int')

    return df

#%%
RPM = pd.read_csv('/Volumes/Personal/Phd/Data/HCR_Lab_Dataset/CAN_Intrusion/RPM_dataset.csv', engine='python',header=None)
# fix for payload = 2 bytes
RPM.loc[RPM[2] == 2, 11] = RPM[5]
RPM.loc[RPM[5] == 'R', 5] = '00'
RPM.loc[RPM[5] == 'T', 5] = '00'

RPM= RPM.rename(columns={0:'time',1:'id',2:'dlc',3:'d0',4:'d1',5:'d2',6:'d3',7:'d4',8:'d5',9:'d6',10:'d7',11:'flag'})
#RPM['label']= np.where(RPM.flag=='R',0,1)
RPM = RPM.fillna('00')
RPM.loc[RPM['flag'] == 'R', 'label'] = 0
RPM.loc[RPM['flag'] == 'T', 'label'] = 1
RPM['label'] = RPM['label'].fillna(0)
RPM['label'] = RPM['label'].astype(int)
RPM = RPM[['id','dlc','d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'time','label','flag']]
RPM['time'] = pd.to_numeric(RPM['time'])

RPM = RPM[0:100000]
RPM = data_preprocessing(RPM)
RPM.reset_index(drop=True, inplace=True)
RPM.name = 'rpm'

#%%
# import and pre-process attack data
DOS = pd.read_csv('/Volumes/Personal/Phd/Data/HCR_Lab_Dataset/CAN_Intrusion/DoS_dataset.csv', engine='python',header=None)
# fix for payload = 2 bytes
DOS.loc[DOS[2] == 2, 11] = DOS[5]
DOS.loc[DOS[5] == 'R', 5] = '00'
DOS.loc[DOS[5] == 'T', 5] = '00'

DOS = DOS.rename(columns={0:'time',1:'id',2:'dlc',3:'d0',4:'d1',5:'d2',6:'d3',7:'d4',8:'d5',9:'d6',10:'d7',11:'flag'})
#DOS['label']= np.where(DOS.flag=='R',0,1)
DOS = DOS.fillna('00')
DOS.loc[DOS['flag'] == 'R', 'label'] = 0
DOS.loc[DOS['flag'] == 'T', 'label'] = 1
DOS['label'] = DOS['label'].fillna(0)
DOS['label'] = DOS['label'].astype(int)
DOS = DOS[['id','dlc','d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'time','label','flag']]
DOS['time'] = pd.to_numeric(DOS['time'])

DOS = DOS[0:100000]
DOS = data_preprocessing(DOS)
DOS.reset_index(drop=True, inplace=True)
DOS.name = 'dos'

#%%
# import and pre-process attack data
fuzzy = pd.read_csv('/Volumes/Personal/Phd/Data/HCR_Lab_Dataset/CAN_Intrusion/Fuzzy_dataset.csv', engine='python',header=None)
# fix for payload = 2 bytes
fuzzy.loc[fuzzy[2] == 2, 11] = fuzzy[5]
fuzzy.loc[fuzzy[5] == 'R', 5] = '00'
fuzzy.loc[fuzzy[5] == 'T', 5] = '00'

fuzzy = fuzzy.rename(columns={0:'time',1:'id',2:'dlc',3:'d0',4:'d1',5:'d2',6:'d3',7:'d4',8:'d5',9:'d6',10:'d7',11:'flag'})
#fuzzy['label']= np.where(fuzzy.flag=='R',0,1)
fuzzy = fuzzy.fillna('00')
fuzzy.loc[fuzzy['flag'] == 'R', 'label'] = 0
fuzzy.loc[fuzzy['flag'] == 'T', 'label'] = 1
fuzzy['label'] = fuzzy['label'].fillna(0)
fuzzy['label'] = fuzzy['label'].astype(int)
fuzzy = fuzzy[['id','dlc','d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'time','label','flag']]
fuzzy['time'] = pd.to_numeric(fuzzy['time'])
fuzzy['d5'].replace({'R':'00'}, inplace=True)

fuzzy = fuzzy[0:100000]
fuzzy = data_preprocessing(fuzzy)
fuzzy.reset_index(drop=True, inplace=True)
fuzzy.name = 'fuzzy'

#%%
# import and pre-process attack data
gear = pd.read_csv('/Volumes/Personal/Phd/Data/HCR_Lab_Dataset/CAN_Intrusion/gear_dataset.csv', engine='python',header=None)
# fix for payload = 2 bytes
gear.loc[gear[2] == 2, 11] = gear[5]
gear.loc[gear[5] == 'R', 5] = '00'
gear.loc[gear[5] == 'T', 5] = '00'

gear = gear.rename(columns={0:'time',1:'id',2:'dlc',3:'d0',4:'d1',5:'d2',6:'d3',7:'d4',8:'d5',9:'d6',10:'d7',11:'flag'})
#gear['label']= np.where(gear.flag=='R',0,1)
gear = gear.fillna('00')
gear.loc[gear['flag'] == 'R', 'label'] = 0
gear.loc[gear['flag'] == 'T', 'label'] = 1
gear['label'] = gear['label'].fillna(0)
gear['label'] = gear['label'].astype(int)
gear = gear[['id','dlc','d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'time','label','flag']]
gear['time'] = pd.to_numeric(gear['time'])


gear = gear[0:100000]
gear = data_preprocessing(gear)
gear.reset_index(drop=True, inplace=True)
gear.name = 'gear'
