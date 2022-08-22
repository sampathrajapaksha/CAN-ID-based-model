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
import time
import seaborn as sns
from sklearn.svm import OneClassSVM
import random
import nltk
import multiprocess as mp
import pickle
import dill
dill.settings['recurse'] = True

# from road_attacks import ID_speedometer

warnings.filterwarnings('ignore')
from keras.callbacks import EarlyStopping


# data preprocessing
# data preprocessing
def data_preprocessing(df):

    df = df.rename(columns={0: 'time', 1: 'id', 2: 'dlc', 3: 'payload'})
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

    df['label'] = 0

    return df

#%%
# benign period of the attack data take as the benign data
# flooding
flooding = pd.read_csv('/Volumes/Personal/Phd/Data/dataset/Soul/Flooding_dataset_KIA.txt', engine='python',header=None)
flooding = flooding[:]
flooding = data_preprocessing(flooding)
flooding['id'] = flooding['id'].str[1:]
flooding.loc[flooding[11] == 'T', 'label'] = 1
flooding_benign = flooding[:60000] # to training dataset
flooding = flooding[60000:]
flooding.reset_index(drop=True, inplace=True)
flooding.name = 'flooding'

# fuzzy
fuzzy = pd.read_csv('/Volumes/Personal/Phd/Data/dataset/Soul/Fuzzy_dataset_KIA.txt', engine='python',header=None)
fuzzy = fuzzy[:]
fuzzy = data_preprocessing(fuzzy)
fuzzy['id'] = fuzzy['id'].str[1:]
fuzzy.loc[fuzzy[11] == 'T', 'label'] = 1
fuzzy_benign = fuzzy[:40000]
fuzzy = fuzzy[40000:]
fuzzy.reset_index(drop=True, inplace=True)
fuzzy.name = 'fuzzy'

# malfunction
malfunction = pd.read_csv('/Volumes/Personal/Phd/Data/dataset/Soul/Malfunction153_dataset_KIA.txt', engine='python',header=None)
malfunction = malfunction[:]
malfunction = data_preprocessing(malfunction)
malfunction['id'] = malfunction['id'].str[1:]
malfunction.loc[malfunction[11] == 'T', 'label'] = 1
malfunction_benign = malfunction[:40000]
malfunction = malfunction[40000:]
malfunction.reset_index(drop=True, inplace=True)
malfunction.name = 'malfunction'

# benign splits of attacks
benign = [flooding_benign, fuzzy_benign, malfunction_benign]

benign_df = pd.DataFrame().append(benign)

# select a random number to split a sample dataset
indx1 = random.randint(0,len(benign_df)-130000)
indx2 = indx1 + 10000
BenTestSet1 = benign_df[indx1:indx2]
sorted_trip1_rest = benign_df[~benign_df.index.isin(BenTestSet1.index)]
BenTrainSet1 = sorted_trip1_rest.copy()

# append both datasets
test_data = [BenTestSet1]
train_data = [sorted_trip1_rest]

BenTestSet = pd.DataFrame().append(test_data)
BenTestSet = BenTestSet.reset_index(drop=True)
BenTrainSet = pd.DataFrame().append(train_data)
BenTrainSet = BenTrainSet.reset_index(drop=True)

print('BenTestSet size :' + str(len(BenTestSet)))
print('BenTrainSet size :' + str(len(BenTrainSet)))

#%%
# functions
# frequency calculation
def calFreq(my_array, my_columns):
    tempData = pd.DataFrame(my_array, columns= my_columns)
    tempData['alert'] = 'B'
    msgs = len(tempData)
    return msgs


def calFreqAtt(my_array, my_columns):
    tempData = pd.DataFrame(my_array, columns= my_columns)
    tempData['alert'] = 'B'
    msgs = len(tempData)

    w_label_1 = len(tempData[tempData['label']==1])
    w_label_all = len(tempData)

    if w_label_1/w_label_all > 0.1: # 0.001 for low frequent attacks, 0.01 for others
        w_label = 1
    else:
        w_label = 0

    return msgs, w_label

# n-gram ratio calculation (3 gram and 1 gram)
def calRatio_3(my_array, my_columns):
    tempData = pd.DataFrame(my_array, columns= my_columns)
    tempData['alert'] = 'B'
    msgs = len(tempData)

    df_list = list()
    #print(len(tempData[tempData['alert']=='B']))
    # x-grams to lists
    gram_3 = ngrams[3]['xgram'].tolist()
    gram_2 = ngrams[2]['xgram'].tolist()

    # # try set (detection latency)
    # gram_3 = ngrams[3]['xgram'].apply(set)
    # gram_2 = ngrams[2]['xgram'].apply(set)

    for i in range(2,len(tempData)):
        inputText = pd.Series(tempData.id[i-2:i])
        inputText = inputText.str.cat(sep=' ')
        # create a list with all possible next id based on n-grams
        pred_list =[]
        # first use longest n-gram (check input text is within 4 gram). if fails try 3,2 and 1-gram respectively
        if inputText in gram_3:
            df = ngrams[3][ngrams[3].xgram==inputText]
            pred_list.extend(df['predict'].tolist())
        else:
            # consider most frequent item as benign
            one_gram = ngrams[1].predict.head(1).to_list()
            pred_list.extend(one_gram)

        if tempData.iloc[i,1] not in pred_list:
            tempData.at[i,'alert'] = 'A'

    w_label_1 = len(tempData[tempData['label']==1])
    w_label_all = len(tempData)

    if w_label_1/w_label_all > 0.1:
        w_label = 1
    else:
        w_label = 0

    return len(tempData[tempData['alert']=='A'])/len(tempData[tempData['alert']=='B']), w_label

# n-gram ratio calculation (2 gram and 1 gram)
def calRatio_2(my_array, my_columns):
    tempData = pd.DataFrame(my_array, columns= my_columns)
    tempData['alert'] = 'B'
    df_list = list()
    # gram_3 = ngrams[3]['xgram'].tolist()
    gram_2 = ngrams[2]['xgram'].tolist()
    for i in range(1,len(tempData)):
        inputText = pd.Series(tempData.id[i-1:i])
        inputText = inputText.str.cat(sep=' ')
        # create a list with all possible next id based on n-grams
        pred_list =[]
        # check inputText in 2-gram xgrams
        if inputText in gram_2:
            df = ngrams[2][ngrams[2].xgram==inputText]
            pred_list.extend(df['predict'].tolist())
        #max frequent id in 1-gram
        #         else:
        #             #print('cannot found in n-grams')
        #             one_gram = ngrams[1].predict.head(5).to_list()
        #             pred_list.extend(one_gram)

        if tempData.iloc[i,1] not in pred_list:
            tempData.at[i,'alert'] = 'A'

    w_label_1 = len(tempData[tempData['label']==1])
    w_label_all = len(tempData)

    if w_label_1/w_label_all > 0.1:
        w_label = 1
    else:
        w_label = 0

    return len(tempData[tempData['alert']=='A'])/len(tempData[tempData['alert']=='B']), w_label

#%%
# N-Gram Analysis
# BenTrainSet = BenTrainSet[:50000]
tempId = pd.Series(BenTrainSet['id'])
print("Total number of ids :"+ str(len(tempId)))
tempId = tempId.str.cat(sep=' ')


# Function to generate n-grams from sentences.
data = tempId
def extract_ngrams(data, num):
    #print(num)
    n_grams = nltk.ngrams(nltk.word_tokenize(data), num)
    return [ ' '.join(grams) for grams in n_grams]

ngrams = list()
for i in range(1,5):
    ngram_fd = nltk.FreqDist(extract_ngrams(data,i))
    ngrams_df = pd.DataFrame(ngram_fd.items(), columns=['ngram', 'freq'])
    ngrams_df = ngrams_df.sort_values(by=['freq'], ascending=False)
    ngrams.append(ngrams_df)

ngrams = [None]+ ngrams

# Calculation of maximum likelihood for last word = count(n-gram) / count(x-gram)
# Step 1 - split ngram in xgram + last word

ngrams_pred = list()
for n in range(1,5):
    df = ngrams[n]
    if n==1:
        df['predict']=df['ngram']
        ngrams_pred.append(df)
    else:
        df_xgram = df['ngram'].str.rpartition()
        df_con = pd.concat([df,df_xgram], axis=1)
        df_con = df_con.rename(columns={0:'xgram',2:'predict'})
        df_con = df_con[['xgram','ngram','freq','predict']]
        ngrams_pred.append(df_con)

ngrams = ngrams_pred.copy()
ngrams = [None]+ ngrams

# Step 2 - calculate ML
ngrams_ml = list()
for n in range(1,5):
    df = ngrams[n]
    if n==1:
        df['ml']=df['freq']/df['freq'].sum()
        ngrams_ml.append(df)
    else:
        dfx = ngrams[n-1]
        df = pd.merge(df, dfx[['ngram','freq']],left_on='xgram',right_on='ngram', how='left')
        df = df.rename(columns={'ngram_x':'ngram','freq_x':'freq'})
        df['ml']=df['freq']/df['freq_y']
        df = df.sort_values(by=['freq','ml'],ascending=False)
        df = df[['xgram', 'ngram', 'freq', 'predict','freq_y', 'ml']]
        ngrams_ml.append(df)

ngrams = ngrams_ml.copy()
ngrams = [None]+ ngrams

#%%
# time-based threshold estimation
#genarate values to use as time for make time intervales

trip_val = BenTestSet
trip_val = trip_val.reset_index(drop=True)

wndowSize = 0.1
numWindows = (trip_val.time.max()-trip_val.time.min())/wndowSize
#benWinLimit = round((testSet.time[len(BenTestSet)]-testSet.time.min())/wndowSize)

startValue = trip_val.time.min()
stopValue = startValue+wndowSize

# Parallel ratio calculation

print(mp.cpu_count())
pool = mp.Pool(mp.cpu_count())

# start = time.time()
print('Parallel processing started...')
ratio_list_th = [pool.apply(calFreq, args=(trip_val[(trip_val.time >= startValue + ((i-1)*wndowSize)) & (trip_val.time < startValue + ((i)*wndowSize))].to_numpy(),trip_val.columns)) for i in range(1,int(numWindows-1))]

pool.close()

# end = time.time()

k_list = list(range(1, int(numWindows-1)))
# print((end - start))

print('Max :', max(ratio_list_th))
print('Min :', min(ratio_list_th))

#%%
# n-gram threshold estimation

trip_val = BenTestSet
trip_val = trip_val.reset_index(drop=True)

wndowSize = 0.1
numWindows = (trip_val.time.max()-trip_val.time.min())/wndowSize
#benWinLimit = round((testSet.time[len(BenTestSet)]-testSet.time.min())/wndowSize)

startValue = trip_val.time.min()
stopValue = startValue+wndowSize

# Parallel ratio calculation
print(mp.cpu_count())
pool = mp.Pool(mp.cpu_count())

start = time.time()
print('Parallel processing started...')
est_ratio_list = [pool.apply(calRatio_2, args=(trip_val[(trip_val.time >= startValue + ((i-1)*wndowSize)) & (trip_val.time < startValue + ((i)*wndowSize))].to_numpy(),trip_val.columns)) for i in range(1,int(numWindows-1))]

pool.close()

end = time.time()

k_list = list(range(1, int(numWindows-1)))
print((end - start))

est_freq = []
for i in range(len(est_ratio_list)):
    l = est_ratio_list[i][0]
    est_freq.append(l)

print('N_gram threshold:', max(est_freq))
two_gram_threshold = max(est_freq)

#%%
attacks = [flooding, fuzzy, malfunction]

# attack dataset frequency
for i in attacks:
    # convert df data into a series
    name = i.name
    print('Attack : ', i.name)
    Attack_dataset = i

    wndowSize = 0.1
    numWindows = (Attack_dataset.time.max()-Attack_dataset.time.min())/wndowSize

    startValue = Attack_dataset.time.min()
    stopValue = startValue+wndowSize

    # Parallel ratio calculation
    print(mp.cpu_count())
    pool = mp.Pool(mp.cpu_count())

    # start = time()
    print('Parallel processing started for frequency...')
    attack_freq = [pool.apply(calFreqAtt, args=(Attack_dataset[(Attack_dataset.time >= startValue + ((i-1)*wndowSize))
                                                               & (Attack_dataset.time < startValue + ((i)*wndowSize))].to_numpy(),Attack_dataset.columns)) for i in range(1,int(numWindows-1))]

    # pool.close()

    # end = time()

    k_list_attack = list(range(1, int(numWindows-1)))

    fuzzing_freq = []
    for i in range(len(attack_freq)):
        l = attack_freq[i][0]
        fuzzing_freq.append(l)

    fuzzing_label = []
    for i in range(len(attack_freq)):
        l = attack_freq[i][1]
        fuzzing_label.append(l)

    y_true = fuzzing_label


    # n_gram calculation
    # start = time()
    print('Parallel processing started for n-grams...')
    attack_freq_ngram = [pool.apply(calRatio_2, args=(Attack_dataset[(Attack_dataset.time >= startValue + ((i-1)*wndowSize)) & (Attack_dataset.time < startValue + ((i)*wndowSize))].to_numpy(),Attack_dataset.columns)) for i in range(1,int(numWindows-1))]

    pool.close()

    # end = time()

    k_list_attack = list(range(1, int(numWindows-1)))
    print('k_list size :', len(k_list_attack))
    # print((end - start))

    fuzzing_ratio = []
    for i in range(len(attack_freq_ngram)):
        l = attack_freq_ngram[i][0]
        fuzzing_ratio.append(l)

    y_pred = []
    for i in range(len(fuzzing_ratio)):
        if fuzzing_ratio[i]> 0.05:
            y_pred.append(1)
        else:
            y_pred.append(0)

    y_true = fuzzing_label

    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    TN = cm[0][0]
    FN = cm[1][0]
    TP = cm[1][1]
    FP = cm[0][1]
    print('FP rate', FP*100/(FP+TN))
    print('FN rate', FN*100/(FN+TP))

    plt.rcParams["figure.figsize"] = [10,5]
    plt.plot(k_list_attack, fuzzing_ratio)
    plt.title(name)
    plt.xlabel('window number')
    plt.ylabel('message frequence')
    plt.axhline(y=0.05, ls ='--', color = 'red')
    plt.show()

#%%
# window = 0.3
plt.rcParams["figure.figsize"] = [10,5]
plt.plot(k_list_attack, fuzzing_ratio)
plt.title(flooding.name)
plt.xlabel('window number')
plt.ylabel('message frequence')
plt.axhline(y=0.22, ls ='--', color = 'red')
plt.show()