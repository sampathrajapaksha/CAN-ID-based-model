# CyCon Paper model

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

# from road_attacks import ID_speedometer

warnings.filterwarnings('ignore')
from keras.callbacks import EarlyStopping


# data preprocessing
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

train = BenTrainSet[:1000000]
train = train[['id', 'time_abs', 'time_dif', 'ID_time_diff']]
train.reset_index(drop=True, inplace=True)
print('train size without max speedometer', train.shape)

maxengine_ben = ID_max_engine[['id', 'time_abs', 'time_dif', 'ID_time_diff']][:40000]
maxengine_ben_mas = ID_max_engine_mas[['id', 'time_abs', 'time_dif', 'ID_time_diff']][:40000]

train = pd.DataFrame().append([train, maxengine_ben])
print('train size with max speedometers', train.shape)
#%%
# create payload and time rule based features (max min values)
# define max values for each payload value
# max_d1 = dict(BenTrainSet.groupby('id')['d1_int'].max())
# max_d2 = dict(BenTrainSet.groupby('id')['d2_int'].max())
# #max_d2['2B4'] = 255
# max_d3 = dict(BenTrainSet.groupby('id')['d3_int'].max())
# max_d4 = dict(BenTrainSet.groupby('id')['d4_int'].max())
# max_d5 = dict(BenTrainSet.groupby('id')['d5_int'].max())
# max_d6 = dict(BenTrainSet.groupby('id')['d6_int'].max())
# #max_d6['03A'] = 255
# max_d7 = dict(BenTrainSet.groupby('id')['d7_int'].max())
# max_d8 = dict(BenTrainSet.groupby('id')['d8_int'].max())
#
# # create a df of max values and add all values as columns
# max_df = pd.DataFrame(max_d1.items(), columns=['id', 'max_d1'])
# max_cols = [max_d2, max_d3, max_d4, max_d5, max_d6, max_d7, max_d8]
# for i in range(len(max_cols)):
#     max_df['max_d' + str(i + 2)] = max_df['id'].map(max_cols[i]).astype('int')  # i+2 to start with max_d2

# difference with previous raw by groups
# eval_df_check['d1_diff'] = eval_df_check.groupby('id')['d1_int'].diff()

# time based features
BenTrainSet['ID_time_diff'] = BenTrainSet['ID_time_diff'].fillna(0)
max_time = dict(BenTrainSet.groupby('id')['ID_time_diff'].quantile(0.999)) # max values have outliers
min_time = dict(BenTrainSet.groupby('id')['ID_time_diff'].min())
max_time_df = pd.DataFrame(max_time.items(), columns=['id', 'max_time'])
max_time_df['min_time'] = max_time_df['id'].map(min_time)

# %%
# load models ( 6 context model - best results)
model = load_model('/saved_files/load_lstm_id_model.h5')
history = pickle.load(open("/saved_files/load_lstm_id_history.pkl", "rb"))
tokenizer = pickle.load(open("/saved_files/road_lstm_tok.pkl", "rb"))

# %%
benign_df = train.copy()

# convert df data into a series
tempId = pd.Series(benign_df['id'])
tempId = tempId.str.cat(sep=' ')

tokenizer = Tokenizer(oov_token=True)
tokenizer.fit_on_texts([tempId])
#
# # saving the tokenizer for predict function.
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
    j = 10    # use 10 for dgx model
    for i in range(j, len(sequence_data)):
        words = sequence_data[i - j:i + 1]
        sequences.append(words)

    # print("The Length of sequences are: ", len(sequences))
    sequences = np.array(sequences)

    # create X and y
    X = []
    y = []

    # # for 2 context ids beside center id
    # for i in sequences:
    #     X.append(i[::2])  # select context ids as x (beside center)
    #     y.append(i[1])  # select center id as y

    # for j context ids beside center id
    k = int(j / 2)
    for i in range(len(sequences)):
        X.append(list((np.delete(sequences[i], k, 0).flatten())))

    for i in sequences:
        y.append(i[k])  # select center id as y

    X = np.array(X)
    y = np.array(y)

    return X, y

#%%
X, y = seq(sequence_data)
y = to_categorical(y, num_classes=vocab_size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.005, random_state=1)

# %%
# model training
j = 10
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=j))
# model.add((LSTM(300, return_sequences=True)))
model.add((GRU(32, return_sequences=False)))
model.add(Dropout(0.2))
# model.add((GRU(256, return_sequences=False)))
# model.add(Dense(128, activation="relu"))
model.add(Dense(vocab_size, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001), metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.5, callbacks=[es]).history

model.save('/Volumes/Personal/Phd/Results/models/LSTM/saved_files/load_lstm_id_model_2.h5')
pickle.dump(history, open("/Volumes/Personal/Phd/Results/models/LSTM/saved_files/load_lstm_id_history_2.pkl", "wb"))

# %%
# load saved models from dgx
model = load_model('/Volumes/Personal/Phd/Results/models/LSTM/dgx_models/lstm_word2vec_model.h5')
history = pickle.load(open("/Volumes/Personal/Phd/Results/models/LSTM/dgx_models/lstm_word2vec_history.pkl", "rb"))
tokenizer = pickle.load(open("/Volumes/Personal/Phd/Results/models/LSTM/dgx_models/tokenizer_word2vec.pkl", "rb"))

# %%

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

    label_ratio = 0.001 # change for low frequent attacks
    if true_label_1 / true_label_all > label_ratio:
        true_label = 1
    else:
        true_label = 0

    label_ratio = 0.001
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

    # pred class
    eval_df['pred_class'] = np.where(eval_df['id_pred_prob'] <= eval_df['threshold'], 1, 0)
    # eval_df['pred_class'] = np.where(eval_df['id_pred_prob'] <= 0.0001, 1, 0)
    # threshold considering all IDs
    # eval_df['pred_class'] = np.where(eval_df['id_pred_prob'] <= threshold_all, 1, 0)
    # time pred class
    # eval_df['pred_class'] = eval_df['time_pred_class']
    # payload pred class
    # eval_df['pred_class'] = eval_df['payload_pred_class']

    # # OCSVM prediction
    # pred_ocsvm = model_ocsvm.predict(eval_df[['id_pred_prob']])
    # pred_ocsvm = np.where(pred_ocsvm==1,0,1)
    # eval_df['pred_class'] = pred_ocsvm

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
extended_short = '/Volumes/Personal/Phd/Data/road/ambient/ambient_dyno_drive_extended_short.log'
# extended_short = '/Volumes/Personal/Phd/Data/road/ambient/ambient_dyno_drive_basic_short.log'
df_extended_short = pd.read_csv(extended_short, engine='python', header=None)
df_extended_short.columns = ['CAN_frame']
# use data_preprocessing from road_attacks
df_extended_short = data_preprocessing(df_extended_short)
# df_extended_short = df_extended_short[df_extended_short.id !='FFF']

start = time.time()
cols = ['id', 'payload', 'time', 'time_abs', 'ID_time_diff', 'label']
df = df_extended_short[cols]

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

# find the predicted probability for each ID
pred_prob = []
for i in range(len(pred_RPM)):
    id_prob = pred_RPM[i][y[i]]  # tokenizer.word_index to get the mapping of id and index
    pred_prob.append(id_prob)

eval_df_benign['id_pred_prob'] = pred_prob
# threshold considering all IDs
threshold_all = eval_df_benign['id_pred_prob'].quantile(0.001)

#%%
# threshold calculation for each id
threshold_df = eval_df_benign.groupby('id')['id_pred_prob'].min()  # defalut - min()
id_threshold_dic = dict(threshold_df)
end = time.time()
tot_time = end - start
print('groupby time : ', tot_time)

# %%
# train a OCSVM for time and pred_prob
# df_ocsvm = eval_df_benign[['id_pred_prob']]
# model_ocsvm = OneClassSVM(kernel = 'rbf', gamma = 100, nu = 0.01).fit(df_ocsvm.head(100000))


# %%
# density plot
df_id = eval_df_benign[eval_df_benign.id == '4E7']
#df_id = eval_df_benign
plt.figure(figsize=(10, 6), dpi=80)
# plt.title('density plot 580', fontsize=16)
sns.distplot(df_id['id_pred_prob'], bins=20, kde=True, color='blue');
plt.xlabel('softmax probability', fontsize = 16)
plt.ylabel('frequency', fontsize = 16)
plt.show()
plt.figure(figsize=(10, 6), dpi=80)
sns.distplot(df_id['ID_time_diff'], bins=20, kde=True, color='blue');
plt.xlabel('ID inter arrival time', fontsize = 16)
plt.ylabel('frequency', fontsize = 16)
plt.show()

#%%
eval_6e0 = eval_df[eval_df['id']=='6E0']
eval_6e0['prob-'] = 1- eval_6e0['id_pred_prob']
sns.lineplot(data=eval_6e0, x="time_abs", y="prob-")
plt.show()



# %%
#
attacks = [ID_fuzzing, ID_speedometer, ID_max_speedometer_mas, ID_corr_sig, ID_corr_sig_mas, ID_reverse_light_on,
           ID_reverse_light_on_mas, ID_reverse_light_off, ID_reverse_light_off_mas]

# 6E0
# id_threshold_dic['6E0'] = 0.01
# attacks = [ID_corr_sig, ID_corr_sig_mas]

attacks = [ID_max_engine, ID_max_engine_mas]
#attacks = [ID_max_engine_mas]
# attacks = attacks[40000:]
#
for i in attacks:
    # convert df data into a series
    print('Attack : ', i.name)
    df = i
    df = df[40000:]
    BenId = pd.Series(df['id'])
    BenId = BenId.str.cat(sep=' ')

    start = time.time()
    sequence_data = tokenizer.texts_to_sequences([BenId])[0]
    X, y = seq(sequence_data)
    eval_df = df[5:-5] # [5:-5] for dgx model

    # # join max_df into eval_df to detect point anomalies in payload
    # eval_df = eval_df.merge(max_df, on ='id', how='left')
    # # check for payload point anomalies
    # eval_df['payload_pred_class'] = np.where(((eval_df.d1_int>eval_df.max_d1)|
    #                                         (eval_df.d2_int>eval_df.max_d2)|
    #                                         (eval_df.d3_int>eval_df.max_d3)|
    #                                         (eval_df.d4_int>eval_df.max_d4)|
    #                                         (eval_df.d5_int>eval_df.max_d5)|
    #                                         (eval_df.d6_int>eval_df.max_d6)|
    #                                         (eval_df.d7_int>eval_df.max_d7)|
    #                                         (eval_df.d8_int>eval_df.max_d8)),1,0)


    # join max_time_df into eval_df to detect point anomalies in time
    eval_df = eval_df.merge(max_time_df, on ='id', how='left')
    eval_df['time_pred_class'] = np.where((eval_df.ID_time_diff>eval_df.max_time)|
                                           (eval_df.ID_time_diff<eval_df.min_time),1,0)


    pred_RPM = model.predict(X)


    true_window, pred_window = results_evaluation(pred_RPM, eval_df, y, winRatio, id_threshold_dic, threshold_all)

    gru_list = pred_window
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

# %%
# Ensemble model predictions
# true_window , gru_list, payload_list, time_list

or_list = [a or b or c for a,b,c in zip(gru_list, time_list, payload_list)]

print(classification_report(true_window, or_list))
cm = confusion_matrix(true_window, or_list)
print(cm)

TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print('FP rate', FP*100/(FP+TN))
print('FN rate', FN*100/(FN+TP))

#%%
# id wise payload analysis
g1 = BenTrainSet.groupby('id')['d1_int'].nunique()
g2 = BenTrainSet.groupby('id')['d2_int'].nunique()
g3 = BenTrainSet.groupby('id')['d3_int'].nunique()
g4 = BenTrainSet.groupby('id')['d4_int'].nunique()
g5 = BenTrainSet.groupby('id')['d5_int'].nunique()
g6 = BenTrainSet.groupby('id')['d6_int'].nunique()
g7 = BenTrainSet.groupby('id')['d7_int'].nunique()
g8 = BenTrainSet.groupby('id')['d8_int'].nunique()

#%%
unique_df = pd.DataFrame(g1)
unique_df = unique_df.merge(g2, on ='id', how='left')
unique_df['d2_int'] = unique_df.index.map(dict(g2))
unique_df['d3_int'] = unique_df.index.map(dict(g3))
unique_df['d4_int'] = unique_df.index.map(dict(g4))
unique_df['d5_int'] = unique_df.index.map(dict(g5))
unique_df['d6_int'] = unique_df.index.map(dict(g6))
unique_df['d7_int'] = unique_df.index.map(dict(g7))
unique_df['d8_int'] = unique_df.index.map(dict(g8))