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
indx1 = random.randint(0,len(benign_df)-10000)
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
# time based features
BenTrainSet['ID_time_diff'] = BenTrainSet['ID_time_diff'].fillna(0)
max_time = dict(BenTrainSet.groupby('id')['ID_time_diff'].max())  # max values have outliers
min_time = dict(BenTrainSet.groupby('id')['ID_time_diff'].min())
max_time_df = pd.DataFrame(max_time.items(), columns=['id', 'max_time'])
max_time_df['min_time'] = max_time_df['id'].map(min_time)

# %%
benign_df = BenTrainSet.copy()

# convert df data into a series
tempId = pd.Series(benign_df['id'])
tempId = tempId.str.cat(sep=' ')

tokenizer = Tokenizer(oov_token=True)
tokenizer.fit_on_texts([tempId])

# saving the tokenizer for predict function.
# pickle.dump(tokenizer, open('/Volumes/Personal/Phd/Results/models/LSTM/road_lstm_tok_2.pkl', 'wb'))

sequence_data = tokenizer.texts_to_sequences([tempId])[0]

vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)

# %%
# create sequences
def seq(sequence_data):
    sequences = []
    # for 2 context ids beside center id
    # j to defined the context window, 2 for one context word from one side, 4 for 2 context words
    j = 10
    for i in range(j, len(sequence_data)):
        words = sequence_data[i - j:i + 1]
        sequences.append(words)

    # print("The Length of sequences are: ", len(sequences))
    sequences = np.array(sequences)

    # create X and y
    X = []
    y = []

    # for j context ids beside center id
    k = int(j / 2)
    for i in range(len(sequences)):
        X.append(list((np.delete(sequences[i], k, 0).flatten())))

    for i in sequences:
        y.append(i[k])  # select center id as y

    X = np.array(X)
    y = np.array(y)

    return X, y


X, y = seq(sequence_data)
y = to_categorical(y, num_classes=vocab_size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=1)

# %%
# model training
j = 10
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=j))
model.add((GRU(32, return_sequences=False)))
model.add(Dropout(0.3))
model.add(Dense(vocab_size, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001), metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.3, callbacks=[es]).history

plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%
# prediction
def winRatio(tempData):
    true_label_1 = len(tempData[tempData['label'] == 1])
    true_label_all = len(tempData)

    pred_label_1 = len(tempData[tempData['pred_class'] == 1])
    pred_label_all = len(tempData)

    label_ratio = 0.1 # change for low frequent attacks
    if true_label_1 / true_label_all > label_ratio:
        true_label = 1
    else:
        true_label = 0

    if pred_label_1 / pred_label_all > label_ratio:
        pred_label = 1
    else:
        pred_label = 0

    return true_label, pred_label


# prediction function
def results_evaluation(pred_RPM, eval_df, y, winRatio, id_threshold_dic, threshold_all):
    # best prediction
    pred_prob = []
    for i in range(len(pred_RPM)):
        id_prob = pred_RPM[i][y[i]]  # tokenizer.word_index to get the mapping of id and index
        pred_prob.append(id_prob)

    eval_df['true_id'] = y
    eval_df['id_pred_prob'] = pred_prob
    eval_df['threshold'] = eval_df['id'].map(id_threshold_dic)
    eval_df['threshold'] = eval_df['threshold'].astype('float')
    eval_df['threshold'] = eval_df['threshold'].fillna(1)

    # ID based threshold pred class
    eval_df['pred_class'] = np.where(eval_df['id_pred_prob'] <= eval_df['threshold'], 1, 0)

    # threshold considering all IDs
    # eval_df['pred_class'] = np.where(eval_df['id_pred_prob'] <= threshold_all, 1, 0)

    # time pred class
    # eval_df['min_time'] = eval_df['min_time'].fillna(1)
    # eval_df['pred_class'] = eval_df['time_pred_class']

    # payload pred class
    # eval_df['pred_class'] = eval_df['payload_pred_class']


    windowSize = 0.1
    numWindows = (eval_df.time.max() - eval_df.time.min()) / windowSize
    # benWinLimit = round((testSet.time[len(BenTestSet)]-testSet.time.min())/wndowSize)

    startValue = eval_df.time.min()
    stopValue = startValue + windowSize

    k_list = []
    ratio_list = []
    for k in range(1, int(numWindows - 1)):
        smallerWindow = eval_df[(eval_df.time >= startValue) & (eval_df.time < stopValue)]

        ratio = winRatio(smallerWindow)
        k_list.append(k)
        ratio_list.append(ratio)

        startValue = stopValue
        stopValue = startValue + windowSize

    # ratio return 2 lists, true window at 0 and pred_window at 1
    true_window = []
    for i in range(len(ratio_list)):
        l = ratio_list[i][0]
        true_window.append(l)

    pred_window = []
    for i in range(len(ratio_list)):
        l = ratio_list[i][1]
        pred_window.append(l)

    # end = time.time()
    # tot_time = end - start
    # print('Total prediction time :', tot_time)
    # print('Time for one frame :', tot_time / len(eval_df))
    print(accuracy_score(eval_df['label'], eval_df['pred_class']))

    return true_window, pred_window

# %%
# Threshold estimation for each ID using benign dataset
start = time.time()

df = BenTestSet

# get train dataset to define threshold
# df = train

# convert df data into a series
BenId = pd.Series(df['id'])
BenId = BenId.str.cat(sep=' ')

sequence_data = tokenizer.texts_to_sequences([BenId])[0]
X, y = seq(sequence_data)
eval_df_benign = df[5:-5]  # 3 - no of context words

pred_RPM = model.predict(X)
eval_df_benign['true_id'] = y

# pred_id - to calculate benign prediction accuracy
y_pred = np.argmax(pred_RPM, axis=1)
eval_df_benign['pred_id'] = y_pred

# find the predicted probability for each ID
pred_prob = []
for i in range(len(pred_RPM)):
    id_prob = pred_RPM[i][y[i]]  # tokenizer.word_index to get the mapping of id and index
    pred_prob.append(id_prob)

eval_df_benign['id_pred_prob'] = pred_prob
threshold_all = eval_df_benign['id_pred_prob'].quantile(0.001)

# threshold calculation for each id

threshold_df = eval_df_benign.groupby('id')['id_pred_prob'].min()
id_threshold_dic = dict(threshold_df)
end = time.time()
tot_time = end - start
print('groupby time : ', tot_time)
print(accuracy_score(y, y_pred))

# %%
#
attacks = [flooding, fuzzy, malfunction]
# attacks = [flooding]
# attacks = [DOS]
#
for i in attacks:
    # convert df data into a series
    print('Attack : ', i.name)
    df = i
    BenId = pd.Series(df['id'])
    BenId = BenId.str.cat(sep=' ')

    start = time.time()
    sequence_data = tokenizer.texts_to_sequences([BenId])[0]
    X, y = seq(sequence_data)
    eval_df = df[5:-5]


    # join max_time_df into eval_df to detect point anomalies in time
    eval_df = eval_df.merge(max_time_df, on ='id', how='left')
    eval_df['time_pred_class'] = np.where((eval_df.ID_time_diff>eval_df.max_time)|
                                          (eval_df.ID_time_diff<eval_df.min_time),1,0)


    pred_RPM = model.predict(X)

    true_window, pred_window = results_evaluation(pred_RPM, eval_df, y, winRatio, id_threshold_dic, threshold_all)

    # gru_list = pred_window
    # time_list = pred_window
    # payload_list = pred_window

    # latency calculation
    end = time.time()
    tot_time = end - start
    print('Total time :', tot_time)
    print('Time for one frame :', tot_time / len(eval_df))

    print(classification_report(true_window, pred_window))
    cm = confusion_matrix(true_window, pred_window)
    print(cm)

    TN = cm[0][0]
    FN = cm[1][0]
    TP = cm[1][1]
    FP = cm[0][1]
    print('FP rate', FP*100/(FP+TN))
    print('FN rate', FN*100/(FN+TP))

    print('New attack data loading....')
    print('**************************************************')
    print()

