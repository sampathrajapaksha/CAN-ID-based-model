import nltk
import os
import pandas as pd
import matplotlib.pyplot as plt
import random
pd.set_option('display.max_columns',20)
pd.set_option('display.max_colwidth', 10000)
pd.set_option('display.float_format', '{:.6f}'.format)
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from time import time
#import multiprocessing as mp
import multiprocess as mp
from datetime import datetime
plt.rcParams["figure.figsize"] = [10,5]
import seaborn as sns
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.svm import OneClassSVM
import re
import pickle

# data pre-processing

def data_preprocessing(df):
    df['time'], df['can'], df['ID'] = df['CAN_frame'].str.split('\s+',2).str
    df['id'],df['payload'] = df['ID'].str.split('#').str
    df = df[['id','payload','time']]
    df['time'] = df['time'].str.replace(r"\(","")
    df['time'] = df['time'].str.replace(r"\)","")
    df['label'] = 0

    # change datatypes
    cols = df.columns
    for i in cols:
        df[i] = df[i].astype('category')

    df['time'] = pd.to_numeric(df['time'])
    sorted_df = df.sort_values(by=['time'])

    return sorted_df

# %%
# import benign.pkl dataset
BenTrainSet = pd.read_pickle('/Volumes/Personal/Phd/Data/road/road_benign.pkl')
# convert payload values to int
payload_columns = ['d1_int', 'd2_int', 'd3_int', 'd4_int', 'd5_int', 'd6_int', 'd7_int', 'd8_int']
for i in payload_columns:
    BenTrainSet[i] = BenTrainSet[i].astype('int')
print('dataset imported')

train = BenTrainSet
train.reset_index(drop=True, inplace=True)
print('train size ', train.shape)

extended_short = '/Volumes/Personal/Phd/Data/road/ambient/ambient_dyno_drive_basic_short.log'
df_extended_short = pd.read_csv(extended_short, engine='python', header=None)
df_extended_short.columns = ['CAN_frame']
df_extended_short = data_preprocessing(df_extended_short)
BenTestSet = df_extended_short
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

    if w_label_1/w_label_all > 0.01: # 0.001 for low frequent attacks, 0.01 for others
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

        if tempData.iloc[i,0] not in pred_list:
            tempData.at[i,'alert'] = 'A'

    w_label_1 = len(tempData[tempData['label']==1])
    w_label_all = len(tempData)

    if w_label_1/w_label_all > 0.01:
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

        if tempData.iloc[i,0] not in pred_list:
            tempData.at[i,'alert'] = 'A'

    w_label_1 = len(tempData[tempData['label']==1])
    w_label_all = len(tempData)

    if w_label_1/w_label_all > 0.01:
        w_label = 1
    else:
        w_label = 0

    return len(tempData[tempData['alert']=='A'])/len(tempData[tempData['alert']=='B']), w_label


#%%
# N-Gram Analysis

tempId = pd.Series(train['id'])
print("Total number of ids :"+ str(len(tempId)))
tempId = tempId.str.cat(sep=' ')


# Function to generate n-grams from sentences.
data = tempId
def extract_ngrams(data, num):
    #print(num)
    n_grams = nltk.ngrams(nltk.word_tokenize(data), num)
    return [ ' '.join(grams) for grams in n_grams]

ngrams = list()
for i in range(1,7):
    ngram_fd = nltk.FreqDist(extract_ngrams(data,i))
    ngrams_df = pd.DataFrame(ngram_fd.items(), columns=['ngram', 'freq'])
    ngrams_df = ngrams_df.sort_values(by=['freq'], ascending=False)
    ngrams.append(ngrams_df)

ngrams = [None]+ ngrams

# Calculation of maximum likelihood for last word = count(n-gram) / count(x-gram)
# Step 1 - split ngram in xgram + last word

ngrams_pred = list()
for n in range(1,7):
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
for n in range(1,7):
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

trip_val = BenTestSet[:50000]
trip_val = trip_val.reset_index(drop=True)

wndowSize = 0.3
numWindows = (trip_val.time.max()-trip_val.time.min())/wndowSize
#benWinLimit = round((testSet.time[len(BenTestSet)]-testSet.time.min())/wndowSize)

startValue = trip_val.time.min()
stopValue = startValue+wndowSize

# Parallel ratio calculation

print(mp.cpu_count())
pool = mp.Pool(mp.cpu_count())

start = time()
print('Parallel processing started...')
ratio_list_th = [pool.apply(calFreq, args=(trip_val[(trip_val.time >= startValue + ((i-1)*wndowSize)) & (trip_val.time < startValue + ((i)*wndowSize))].to_numpy(),trip_val.columns)) for i in range(1,int(numWindows-1))]

pool.close()

end = time()

k_list = list(range(1, int(numWindows-1)))
print((end - start)/3600)

print('Max :', max(ratio_list_th))
print('Min :', min(ratio_list_th))

#%%
# n-gram threshold estimation

trip_val = BenTestSet[:50000]
trip_val = trip_val.reset_index(drop=True)

wndowSize = 0.1
numWindows = (trip_val.time.max()-trip_val.time.min())/wndowSize
#benWinLimit = round((testSet.time[len(BenTestSet)]-testSet.time.min())/wndowSize)

startValue = trip_val.time.min()
stopValue = startValue+wndowSize

# Parallel ratio calculation
print(mp.cpu_count())
pool = mp.Pool(mp.cpu_count())

start = time()
print('Parallel processing started...')
est_ratio_list = [pool.apply(calRatio_2, args=(trip_val[(trip_val.time >= startValue + ((i-1)*wndowSize)) & (trip_val.time < startValue + ((i)*wndowSize))].to_numpy(),trip_val.columns)) for i in range(1,int(numWindows-1))]

pool.close()

end = time()

k_list = list(range(1, int(numWindows-1)))
print((end - start))

est_freq = []
for i in range(len(est_ratio_list)):
    l = est_ratio_list[i][0]
    est_freq.append(l)

print('N_gram threshold:', max(est_freq))
two_gram_threshold = max(est_freq)

#%%
# Attacks
attacks = [ID_fuzzing, ID_speedometer, ID_max_speedometer_mas, ID_corr_sig, ID_corr_sig_mas, ID_reverse_light_on,
           ID_reverse_light_on_mas, ID_reverse_light_off, ID_reverse_light_off_mas]
# attacks = [ID_fuzzing]
# attacks = [ID_max_engine_mas]

# attack dataset frequency
for i in attacks:
    # convert df data into a series
    print('Attack : ', i.name)
    Attack_dataset = i

    wndowSize = 0.1
    numWindows = (Attack_dataset.time.max()-Attack_dataset.time.min())/wndowSize
    AttStart = round((A_start-Attack_dataset.time.min())/wndowSize)
    AttEnd = round((A_end-Attack_dataset.time.min())/wndowSize)

    startValue = Attack_dataset.time.min()
    stopValue = startValue+wndowSize

    # Parallel ratio calculation
    print(mp.cpu_count())
    pool = mp.Pool(mp.cpu_count())

    start = time()
    print('Parallel processing started for frequency...')
    attack_freq = [pool.apply(calFreqAtt, args=(Attack_dataset[(Attack_dataset.time >= startValue + ((i-1)*wndowSize))
                                                               & (Attack_dataset.time < startValue + ((i)*wndowSize))].to_numpy(),Attack_dataset.columns)) for i in range(1,int(numWindows-1))]

    # pool.close()

    end = time()

    k_list_attack = list(range(1, int(numWindows-1)))
    # print('k_list size :', len(k_list_attack))
    # print((end - start))
    # print('AttStart :', AttStart)
    # print('AttEnd :', AttEnd)

    fuzzing_freq = []
    for i in range(len(attack_freq)):
        l = attack_freq[i][0]
        fuzzing_freq.append(l)

    fuzzing_label = []
    for i in range(len(attack_freq)):
        l = attack_freq[i][1]
        fuzzing_label.append(l)

    # y_pred = []
    # for i in range(len(fuzzing_freq)):
    #     if fuzzing_freq[i]> 257:
    #         y_pred.append(1)
    #     else:
    #         y_pred.append(0)

    y_true = fuzzing_label

    # print(classification_report(y_true, y_pred))
    # cm = confusion_matrix(y_true, y_pred)
    # print(cm)
    #
    # TN = cm[0][0]
    # FN = cm[1][0]
    # TP = cm[1][1]
    # FP = cm[0][1]
    # print('FP rate', FP*100/(FP+TN))
    # print('FN rate', FN*100/(FN+TP))


    # n_gram calculation
    start = time()
    print('Parallel processing started for n-grams...')
    attack_freq_ngram = [pool.apply(calRatio_3, args=(Attack_dataset[(Attack_dataset.time >= startValue + ((i-1)*wndowSize)) & (Attack_dataset.time < startValue + ((i)*wndowSize))].to_numpy(),Attack_dataset.columns)) for i in range(1,int(numWindows-1))]

    pool.close()

    end = time()

    k_list_attack = list(range(1, int(numWindows-1)))
    print('k_list size :', len(k_list_attack))
    print((end - start))
    # print('AttStart :', AttStart)
    # print('AttEnd :', AttEnd)

    fuzzing_ratio = []
    for i in range(len(attack_freq_ngram)):
        l = attack_freq_ngram[i][0]
        fuzzing_ratio.append(l)

    y_pred = []
    for i in range(len(fuzzing_ratio)):
        if fuzzing_ratio[i]> 0.008:
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

 #%%
# n-gram model for attacks

sns.kdeplot(est_freq, fill=True)
plt.show()