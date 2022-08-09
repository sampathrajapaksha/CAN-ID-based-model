# payload analysis for payload model

# import libraries
# import tensorflow as tf
# from tensorflow.keras import backend as K
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization, GRU, Layer
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.optimizers import Adam
# from keras.models import load_model
import pickle
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')
# from keras.callbacks import EarlyStopping
# from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
# from keras.models import Model
# from keras import regularizers

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
    df['time_dif'] = df['time_dif'].fillna(df['time_dif'].mean())
    df[['ID_time_diff']] = df.groupby('id')['time_abs'].diff()
    df['ID_time_diff'] = df['ID_time_diff'].fillna(df.groupby('id')['ID_time_diff'].transform('mean'))

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

    # convert hex signal to int signal
    df['id_int'] = df['id'].apply(hex_to_int)
    df['d1_int'] = df['d1'].apply(hex_to_int)
    df['d2_int'] = df['d2'].apply(hex_to_int)
    df['d3_int'] = df['d3'].apply(hex_to_int)
    df['d4_int'] = df['d4'].apply(hex_to_int)
    df['d5_int'] = df['d5'].apply(hex_to_int)
    df['d6_int'] = df['d6'].apply(hex_to_int)
    df['d7_int'] = df['d7'].apply(hex_to_int)
    df['d8_int'] = df['d8'].apply(hex_to_int)

    sorted_df = df.sort_values(by=['time'])

    # filter columns
    sorted_df = sorted_df[['id','time_abs','d1_int','d2_int','d3_int','d4_int','d5_int','d6_int','d7_int','d8_int']]

    # def hex_to_bin(hex):
    #   val = bin(int(hex, 16))[2:]
    #   return val

    return sorted_df

# %%
# import benign datasets
basic_long = '/Volumes/Personal/Phd/Data/road/ambient/ambient_dyno_drive_basic_long.log'
basic_short = '/Volumes/Personal/Phd/Data/road/ambient/ambient_dyno_drive_basic_short.log'
benign_anomaly = '/Volumes/Personal/Phd/Data/road/ambient/ambient_dyno_drive_benign_anomaly.log'
extended_long = '/Volumes/Personal/Phd/Data/road/ambient/ambient_dyno_drive_extended_long.log'
extended_short = '/Volumes/Personal/Phd/Data/road/ambient/ambient_dyno_drive_extended_short.log'
radio_infotainment = '/Volumes/Personal/Phd/Data/road/ambient/ambient_dyno_drive_radio_infotainment.log'
drive_winter = '/Volumes/Personal/Phd/Data/road/ambient/ambient_dyno_drive_winter.log'
exercise_all_bits = '/Volumes/Personal/Phd/Data/road/ambient/ambient_dyno_exercise_all_bits.log'
idle_radio_infotainment = '/Volumes/Personal/Phd/Data/road/ambient/ambient_dyno_idle_radio_infotainment.log'
reverse = '/Volumes/Personal/Phd/Data/road/ambient/ambient_dyno_reverse.log'
highway_street_driving = '/Volumes/Personal/Phd/Data/road/ambient/ambient_highway_street_driving_diagnostics.log'
highway_street_driving_long = '/Volumes/Personal/Phd/Data/road/ambient/ambient_highway_street_driving_long.log'

# read csv as pandas
print('read .log files...')
df_basic_long = pd.read_csv(basic_long, engine='python',header=None)
df_basic_long.columns =['CAN_frame']
df_basic_short = pd.read_csv(basic_short, engine='python',header=None)
df_basic_short.columns =['CAN_frame']
df_extended_long = pd.read_csv(extended_long, engine='python',header=None)
df_extended_long.columns =['CAN_frame']
df_extended_short = pd.read_csv(extended_short, engine='python',header=None)
df_extended_short.columns =['CAN_frame']
df_radio_infotainment = pd.read_csv(radio_infotainment, engine='python',header=None)
df_radio_infotainment.columns =['CAN_frame']
df_drive_winter = pd.read_csv(drive_winter, engine='python',header=None)
df_drive_winter.columns =['CAN_frame']
df_exercise_all_bits = pd.read_csv(exercise_all_bits, engine='python',header=None)
df_exercise_all_bits.columns =['CAN_frame']
df_idle_radio_infotainment = pd.read_csv(idle_radio_infotainment, engine='python',header=None)
df_idle_radio_infotainment.columns =['CAN_frame']
df_reverse = pd.read_csv(reverse, engine='python',header=None)
df_reverse.columns =['CAN_frame']
df_highway_street_driving = pd.read_csv(highway_street_driving, engine='python',header=None)
df_highway_street_driving.columns =['CAN_frame']
df_highway_street_driving_long = pd.read_csv(highway_street_driving_long, engine='python',header=None)
df_highway_street_driving_long.columns =['CAN_frame']
df_benign_anomaly = pd.read_csv(benign_anomaly, engine='python',header=None)
df_benign_anomaly.columns =['CAN_frame']
print('file reading completed')

print('hello')

#%%
# data preprocessing
df_basic_long = data_preprocessing(df_basic_long)
df_basic_short = data_preprocessing(df_basic_short)
df_extended_long = data_preprocessing(df_extended_long)
df_extended_short = data_preprocessing(df_extended_short)
df_radio_infotainment = data_preprocessing(df_radio_infotainment)
df_drive_winter = data_preprocessing(df_drive_winter)
df_exercise_all_bits = data_preprocessing(df_exercise_all_bits)
df_idle_radio_infotainment = data_preprocessing(df_idle_radio_infotainment)
df_reverse = data_preprocessing(df_reverse)
df_highway_street_driving = data_preprocessing(df_highway_street_driving)
df_highway_street_driving_long = data_preprocessing(df_highway_street_driving_long)
df_benign_anomaly = data_preprocessing(df_benign_anomaly)

#%%
# unique payload value identification
dataframe = df_basic_long
id_groups = dataframe.groupby('id')
id_groups_n = id_groups.agg({'d1_int':'nunique', 'd2_int':'nunique','d3_int':'nunique', 'd4_int':'nunique', 'd5_int':'nunique', 'd6_int':'nunique', 'd7_int':'nunique', 'd8_int':'nunique' })
id_groups_n = id_groups_n.reset_index()

#%%
import plotly.express as px
import plotly.io as pio
id = df_basic_long[df_basic_long.id == '0C0']

y = 'd1_int'
pio.renderers.default = "browser"
fig = px.line(id, x =id.time_abs, y = y )
fig.add_vrect(x0=525, x1=575, fillcolor="red", opacity=0.25, line_width=0)
fig.show()

#%%
# analysis with all data
BenTrainSet = pd.read_pickle('/Volumes/Personal/Phd/Data/road/road_benign.pkl')
# convert payload values to int
payload_columns = ['d1_int', 'd2_int', 'd3_int', 'd4_int', 'd5_int', 'd6_int', 'd7_int', 'd8_int']
for i in payload_columns:
    BenTrainSet[i] = BenTrainSet[i].astype('int')
print('dataset imported')

#%%
# unique payload value identification
dataframe = BenTrainSet
id_groups = dataframe.groupby('id')
id_groups_int = id_groups.agg({'d1_int':'nunique', 'd2_int':'nunique','d3_int':'nunique', 'd4_int':'nunique', 'd5_int':'nunique', 'd6_int':'nunique', 'd7_int':'nunique', 'd8_int':'nunique' })
id_groups_int = id_groups_int.reset_index()

#%%
# identify constants, sensor values, counters and catogories
val_df_basic = id_groups_n[['d1_int', 'd2_int', 'd3_int', 'd4_int', 'd5_int', 'd6_int', 'd7_int', 'd8_int']]
values_basic = pd.value_counts(val_df_basic.values.ravel())

#%%
# Create data for GRU model
one_hot_id = pd.get_dummies(df_basic_long.id)
df_basic_long_id = df_basic_long.join(one_hot_id)


