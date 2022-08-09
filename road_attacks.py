# attacks for road dataset                         *
# pre-processed for id and payloads  (int)         *
#***************************************************

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


# function to preprocess data
def data_preprocessing(df):
    df['time'], df['can'], df['ID'] = df['CAN_frame'].str.split('\s+', 2).str
    df['id'], df['payload'] = df['ID'].str.split('#').str
    df = df[['id', 'payload', 'time']]
    df['time'] = df['time'].str.replace(r"\(", "")
    df['time'] = df['time'].str.replace(r"\)", "")
    df['label'] = 0

    # change datatypes
    cols = df.columns
    for i in cols:
        df[i] = df[i].astype('category')

    df['time'] = pd.to_numeric(df['time'])
    df['time_abs'] = df.time - min(df.time)

    # create features for time difference and time diff for each id
    df['time_dif'] = df['time_abs'].diff()
    # df['time_dif'] = df['time_dif'].fillna(df['time_dif'].mean())
    # df[['ID_time_diff']] = df.groupby('id')['time_abs'].diff()
    # df['ID_time_diff'] = df['ID_time_diff'].fillna(df.groupby('id')['ID_time_diff'].transform('mean'))

    df['d1'] = df['payload'].str[:2].astype('category')
    df['d2'] = df['payload'].str[2:4].astype('category')
    df['d3'] = df['payload'].str[4:6].astype('category')
    df['d4'] = df['payload'].str[6:8].astype('category')
    df['d5'] = df['payload'].str[8:10].astype('category')
    df['d6'] = df['payload'].str[10:12].astype('category')
    df['d7'] = df['payload'].str[12:14].astype('category')
    df['d8'] = df['payload'].str[14:16].astype('category')

    # function to convert hex to int
    def hex_to_int(hex):
        val = int(hex, 16)
        return val

    def hex_to_bin(hex):
        val = bin(int(hex, 16))[2:]
        return val

    # convert hex signal to int signal
    df['id_int'] = df['id'].apply(hex_to_int).astype('int')
    df['d1_int'] = df['d1'].apply(hex_to_int).astype('int')
    df['d2_int'] = df['d2'].apply(hex_to_int).astype('int')
    df['d3_int'] = df['d3'].apply(hex_to_int).astype('int')
    df['d4_int'] = df['d4'].apply(hex_to_int).astype('int')
    df['d5_int'] = df['d5'].apply(hex_to_int).astype('int')
    df['d6_int'] = df['d6'].apply(hex_to_int).astype('int')
    df['d7_int'] = df['d7'].apply(hex_to_int).astype('int')
    df['d8_int'] = df['d8'].apply(hex_to_int).astype('int')

    # convert id to binary
    # df['id_bin'] = df['id'].apply(hex_to_bin)
    # for i in range(12):
    #     df['id_'+ str(i)] = df['id_bin'].str[i].astype('category')

    sorted_df = df.sort_values(by=['time'])

    return sorted_df

#%%
# testing dataset pre-processing
print('fuzzing attack...')
fuzzing = '/Users/sampathrajapaksha/Document-local/PhD/Data/road/attacks/fuzzing_attack_1.log'
fuzzing = pd.read_csv(fuzzing, engine='python',header=None)
fuzzing.columns =['CAN_frame']
fuzzing = data_preprocessing(fuzzing)

ID_fuzzing = fuzzing
A_start = 1000000004.622975
A_end = 1000000007.958234
ID_fuzzing['label'] = np.where((ID_fuzzing.time>=A_start)
                            &(ID_fuzzing.time<=A_end)
                            &(ID_fuzzing.payload =='FFFFFFFFFFFFFFFF'),1,0)

ID_fuzzing.reset_index(drop=True, inplace=True)
ID_fuzzing.name = 'Fuzzing attack'

#df = ID_fuzzing
# X_test = scaler.fit_transform(ID_fuzzing[cols])
#
# X_test, Y_test = to_sequences(X_test, ID_fuzzing['label'].values, seq_size)


# **********************************************************************************************************************
# testing dataset pre-processing
print('max_speedometer...')
max_speedometer = '/Users/sampathrajapaksha/Document-local/PhD/Data/road/attacks/max_speedometer_attack_1.log'
max_speedometer = pd.read_csv(max_speedometer, engine='python',header=None)
max_speedometer.columns =['CAN_frame']
max_speedometer = data_preprocessing(max_speedometer)

ID_speedometer = max_speedometer
A_start = 1110000042.009204
A_end = 1110000066.449010
# A_start = 42.009204
# A_end = 66.449010
ID_speedometer['label'] = np.where((ID_speedometer.time>=A_start)
                            &(ID_speedometer.time<=A_end)
                            &(ID_speedometer.id=='0D0')
                            &(ID_speedometer.d6=='FF'),1,0)
ID_speedometer.reset_index(drop=True, inplace=True)

# cols = ['id','payload','time','time_abs','ID_time_diff','label']
# ID_speedometer = ID_speedometer[cols]
ID_speedometer.name = 'Speedometer attack'
#df = ID_speedometer
# X_test = scaler.fit_transform(ID_speedometer[cols])
#
# X_test, Y_test = to_sequences(X_test, ID_speedometer['label'].values, seq_size)

# **********************************************************************************************************************
print('max_speedometer_mas...')
max_speedometer_mas = '/Users/sampathrajapaksha/Document-local/PhD/Data/road/attacks/max_speedometer_attack_1_masquerade.log'
max_speedometer_mas = pd.read_csv(max_speedometer_mas, engine='python',header=None)
max_speedometer_mas.columns =['CAN_frame']
max_speedometer_mas = data_preprocessing(max_speedometer_mas)

ID_max_speedometer_mas = max_speedometer_mas
A_start = 1140000042.009204
A_end = 1140000066.449010
ID_max_speedometer_mas['label'] = np.where((ID_max_speedometer_mas.time>=A_start)
                                   &(ID_max_speedometer_mas.time<=A_end)
                                   &(ID_max_speedometer_mas.id=='0D0')
                                   &(ID_max_speedometer_mas.d6=='FF'),1,0)
ID_max_speedometer_mas.reset_index(drop=True, inplace=True)
ID_max_speedometer_mas.name = 'Max speedometer mas attack'
# X_test = scaler.fit_transform(ID_max_speedometer_mas[cols])
#
# X_test, Y_test = to_sequences(X_test, ID_max_speedometer_mas['label'].values, seq_size)

# **********************************************************************************************************************
print('corr_sig...')
corr_sig = '/Users/sampathrajapaksha/Document-local/PhD/Data/road/attacks/correlated_signal_attack_1.log'
corr_sig = pd.read_csv(corr_sig, engine='python',header=None)
corr_sig.columns =['CAN_frame']
corr_sig = data_preprocessing(corr_sig)

ID_corr_sig = corr_sig
A_start = 1030000009.191851
A_end = 1030000030.050109
# A_start = 9.191851
# A_end = 30.050109
ID_corr_sig['label'] = np.where((ID_corr_sig.time>=A_start)
                       &(ID_corr_sig.time<=A_end)
                       &(ID_corr_sig.id=='6E0')
                       &(ID_corr_sig.payload=='595945450000FFFF'),1,0)
ID_corr_sig.reset_index(drop=True, inplace=True)
ID_corr_sig.name = 'corr_sig attack'
#df = ID_corr_sig
# X_test = scaler.fit_transform(ID_corr_sig[cols])
#
# X_test, Y_test = to_sequences(X_test, ID_corr_sig['label'].values, seq_size)

# **********************************************************************************************************************
print('corr_sig_mas...')
corr_sig_mas = '/Users/sampathrajapaksha/Document-local/PhD/Data/road/attacks/correlated_signal_attack_1_masquerade.log'
corr_sig_mas = pd.read_csv(corr_sig_mas, engine='python',header=None)
corr_sig_mas.columns =['CAN_frame']
corr_sig_mas = data_preprocessing(corr_sig_mas)


ID_corr_sig_mas = corr_sig_mas
A_start = 1060000009.191851
A_end = 1060000030.050109
ID_corr_sig_mas['label'] = np.where((ID_corr_sig_mas.time>=A_start)
                                &(ID_corr_sig_mas.time<=A_end)
                                &(ID_corr_sig_mas.id=='6E0')
                                &(ID_corr_sig_mas.payload=='595945450000FFFF'),1,0)
ID_corr_sig_mas.reset_index(drop=True, inplace=True)
ID_corr_sig_mas.name = 'corr_sig_mas attack'
# X_test = scaler.fit_transform(ID_corr_sig_mas[cols])
#
# X_test, Y_test = to_sequences(X_test, ID_corr_sig_mas['label'].values, seq_size)

# **********************************************************************************************************************
print('reverse_light_on...')
reverse_light_on = '/Users/sampathrajapaksha/Document-local/PhD/Data/road/attacks/reverse_light_on_attack_1.log'
reverse_light_on = pd.read_csv(reverse_light_on, engine='python',header=None)
reverse_light_on.columns =['CAN_frame']
reverse_light_on = data_preprocessing(reverse_light_on)

ID_reverse_light_on = reverse_light_on
A_start = 1230000018.929177
A_end = 1230000038.836015
# A_start = 18.929177
# A_end = 38.836015
ID_reverse_light_on['label'] = np.where((ID_reverse_light_on.time>=A_start)
                       &(ID_reverse_light_on.time<=A_end)
                       &(ID_reverse_light_on.id=='0D0')
                       &(ID_reverse_light_on.d3=='0C'),1,0)
ID_reverse_light_on.reset_index(drop=True, inplace=True)
ID_reverse_light_on.name = 'reverse_light_on attack'
# X_test = scaler.fit_transform(ID_reverse_light_on[cols])
#
# X_test, Y_test = to_sequences(X_test, ID_reverse_light_on['label'].values, seq_size)

# **********************************************************************************************************************
print('reverse_light_on_mas...')
reverse_light_on_mas = '/Users/sampathrajapaksha/Document-local/PhD/Data/road/attacks/reverse_light_on_attack_1_masquerade.log'
reverse_light_on_mas = pd.read_csv(reverse_light_on_mas, engine='python',header=None)
reverse_light_on_mas.columns =['CAN_frame']
reverse_light_on_mas = data_preprocessing(reverse_light_on_mas)

ID_reverse_light_on_mas = reverse_light_on_mas
A_start = 1260000018.929177
A_end = 1260000038.836015
ID_reverse_light_on_mas['label'] = np.where((ID_reverse_light_on_mas.time>=A_start)
                                        &(ID_reverse_light_on_mas.time<=A_end)
                                        &(ID_reverse_light_on_mas.id=='0D0')
                                        &(ID_reverse_light_on_mas.d3=='0C'),1,0)
ID_reverse_light_on_mas.reset_index(drop=True, inplace=True)
ID_reverse_light_on_mas.name = 'reverse_light_on_mas attack'

# **********************************************************************************************************************
print('reverse_light_off...')
reverse_light_off = '/Users/sampathrajapaksha/Document-local/PhD/Data/road/attacks/reverse_light_off_attack_1.log'
reverse_light_off = pd.read_csv(reverse_light_off, engine='python',header=None)
reverse_light_off.columns =['CAN_frame']
reverse_light_off = data_preprocessing(reverse_light_off)

ID_reverse_light_off = reverse_light_off
A_start = 1170000016.627923
A_end = 1170000023.347311
# A_start = 16.627923
# A_end = 23.347311
ID_reverse_light_off['label'] = np.where((ID_reverse_light_off.time>=A_start)
                                        &(ID_reverse_light_off.time<=A_end)
                                        &(ID_reverse_light_off.id=='0D0')
                                        &(ID_reverse_light_off.d3=='04'),1,0)
ID_reverse_light_off.reset_index(drop=True, inplace=True)
ID_reverse_light_off.name = 'reverse_light_off attack'

# **********************************************************************************************************************
print('reverse_light_off_mas...')
reverse_light_off_mas = '/Users/sampathrajapaksha/Document-local/PhD/Data/road/attacks/reverse_light_off_attack_1_masquerade.log'
reverse_light_off_mas = pd.read_csv(reverse_light_off_mas, engine='python',header=None)
reverse_light_off_mas.columns =['CAN_frame']
reverse_light_off_mas = data_preprocessing(reverse_light_off_mas)

ID_reverse_light_off_mas = reverse_light_off_mas
A_start = 1200000016.627923
A_end = 1200000023.347311
ID_reverse_light_off_mas['label'] = np.where((ID_reverse_light_off_mas.time>=A_start)
                                         &(ID_reverse_light_off_mas.time<=A_end)
                                         &(ID_reverse_light_off_mas.id=='0D0')
                                         &(ID_reverse_light_off_mas.d3=='04'),1,0)
ID_reverse_light_off_mas.reset_index(drop=True, inplace=True)
ID_reverse_light_off_mas.name = 'reverse_light_off_mas'

# **********************************************************************************************************************
print('max_engine...')
max_engine = '/Users/sampathrajapaksha/Document-local/PhD/Data/road/attacks/max_engine_coolant_temp_attack.log'
max_engine = pd.read_csv(max_engine, engine='python',header=None)
max_engine.columns =['CAN_frame']
max_engine = data_preprocessing(max_engine)

ID_max_engine = max_engine
A_start = 1090000019.979080
A_end = 1090000024.170183
# A_start = 19.979080
# A_end = 24.170183
ID_max_engine['label'] = np.where((ID_max_engine.time>=A_start)
                       &(ID_max_engine.time<=A_end)
                       &(ID_max_engine.id=='4E7')
                       &(ID_max_engine.d6=='FF'),1,0)
ID_max_engine.reset_index(drop=True, inplace=True)
ID_max_engine.name = 'max_engine coolant temp attack'

# **********************************************************************************************************************
print('max_engine_mas...')
max_engine_mas = '/Users/sampathrajapaksha/Document-local/PhD/Data/road/attacks/max_engine_coolant_temp_attack_masquerade.log'
max_engine_mas = pd.read_csv(max_engine_mas, engine='python',header=None)
max_engine_mas.columns =['CAN_frame']
max_engine_mas = data_preprocessing(max_engine_mas)

ID_max_engine_mas = max_engine_mas
A_start = 1100000019.979080
A_end = 1100000024.170183
ID_max_engine_mas['label'] = np.where((ID_max_engine_mas.time>=A_start)
                                  &(ID_max_engine_mas.time<=A_end)
                                  &(ID_max_engine_mas.id=='4E7')
                                  &(ID_max_engine_mas.d6=='FF'),1,0)
ID_max_engine_mas.reset_index(drop=True, inplace=True)
ID_max_engine_mas.name = 'max_engine coolant temp mas attack'