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

from road_attacks import ID_speedometer

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
print('dataset imported')
# BenTrainSet = BenTrainSet[BenTrainSet.id!='FFF']

train = BenTrainSet[:10000000]
train.reset_index(drop=True, inplace=True)
print('train size ', train.shape)

# %%
benign_df = train.copy()

# convert df data into a series
tempId = pd.Series(benign_df['id'])
tempId = tempId.str.cat(sep=' ')

tokenizer = Tokenizer(oov_token=True)
tokenizer.fit_on_texts([tempId])

# saving the tokenizer for predict function.
pickle.dump(tokenizer, open('/Volumes/Personal/Phd/Results/models/LSTM/road_lstm_tok.pkl', 'wb'))

sequence_data = tokenizer.texts_to_sequences([tempId])[0]

vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)


# %%
# create sequences
def seq(sequence_data):
    sequences = []
    # for 2 context ids beside center id
    # j to defined the context window, 2 for one context word from one side, 4 for 2 context words
    j = 6
    for i in range(j, len(sequence_data)):
        words = sequence_data[i - j:i + 1]
        sequences.append(words)
    print(sequences[0:2])
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
    k = int(j/2)
    for i in range(len(sequences)):
        X.append(list((np.delete(sequences[i], k, 0).flatten())))

    for i in sequences:
        y.append(i[k]) # select center id as y

    X = np.array(X)
    y = np.array(y)

    return X, y


X, y = seq(sequence_data)
y = to_categorical(y, num_classes=vocab_size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.005, random_state=1)

# %%
# model training
j = 6
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=j))
#model.add((LSTM(300, return_sequences=True)))
model.add((GRU(256, return_sequences=False)))
model.add(Dropout(0.3))
#model.add((GRU(256, return_sequences=False)))
model.add(Dense(128, activation="relu"))
model.add(Dense(vocab_size, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001), metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
history = model.fit(X_train, y_train, epochs=5, batch_size=512, validation_split=0.3, callbacks=[es]).history

model.save('/Volumes/Personal/Phd/Results/models/LSTM/load_lstm_id_model.h5')
pickle.dump(history, open("/Volumes/Personal/Phd/Results/models/LSTM/load_lstm_id_history.pkl", "wb"))

#%%
# load saved models from dgx
model = load_model('/Volumes/Personal/Phd/Results/models/LSTM/dgx_models/xx.h5')
history = pickle.load(open("/Volumes/Personal/Phd/Results/models/LSTM/dgx_models/xx.pkl", "rb"))
tokenizer = pickle.load(open("/Volumes/Personal/Phd/Results/models/LSTM/dgx_models/xx.pkl", "rb"))

#%%

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

    if true_label_1 / true_label_all > 0.001:
        true_label = 1
    else:
        true_label = 0

    if pred_label_1 / pred_label_all > 0.001:
        pred_label = 1
    else:
        pred_label = 0

    return true_label, pred_label


# prediction function
def results_evaluation(pred_RPM, eval_df, y, winRatio):
    # best prediction
    y_pred = np.argmax(pred_RPM, axis=1)

    # # best N predictions
    # pred_list = []
    # for i in range(len(pred_RPM)):
    #     y_pred_max = (-pred_RPM[i]).argsort()[:90]  # (- to get the descending order)
    #     pred_list.append(y_pred_max)

    pred_list = []
    for i in range(len(pred_RPM)):
      y_pred_max = np.argwhere(pred_RPM[i]> 0.000005)
      pred_list.append(y_pred_max)

    eval_df['predict_id'] = y_pred
    eval_df['true_id'] = y
    eval_df['pred_all'] = pred_list
    eval_df['one_class'] = np.where(eval_df['true_id'] == eval_df['predict_id'], 0, 1)

    pred_label = []
    for i in range(len(y)):
        if y[i] in pred_list[i]:
            pred_class = 0
            pred_label.append(pred_class)
        else:
            pred_class = 1
            pred_label.append(pred_class)

    eval_df['pred_class'] = pred_label

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

    true_window = []
    for i in range(len(ratio_list)):
        l = ratio_list[i][0]
        true_window.append(l)

    pred_window = []
    for i in range(len(ratio_list)):
        l = ratio_list[i][1]
        pred_window.append(l)

    end = time.time()
    tot_time = end - start
    print('Total prediction time :', tot_time)
    print('Time for one frame :', tot_time / len(eval_df))
    print(accuracy_score(y, y_pred))

    return true_window, pred_window


# %%
# Threshold estimation for each ID using benign dataset
extended_short = '/Volumes/Personal/Phd/Data/road/ambient/ambient_dyno_drive_basic_short.log'
df_extended_short = pd.read_csv(extended_short, engine='python', header=None)
df_extended_short.columns = ['CAN_frame']
df_extended_short = data_preprocessing(df_extended_short)

cols = ['id','payload','time','time_abs','ID_time_diff','label']
df = df_extended_short[cols]

# convert df data into a series
BenId = pd.Series(df['id'])
BenId = BenId.str.cat(sep=' ')

sequence_data = tokenizer.texts_to_sequences([BenId])[0]
X, y = seq(sequence_data)
eval_df = df[3:-3] # 3 - no of context words

pred_RPM = model.predict(X)
eval_df['true_id'] = y

# find the predicted probability for each ID
pred_prob = []
for i in range(len(pred_RPM)):
    id_prob = pred_RPM[i][y[i]] # tokenizer.word_index to get the mapping of id and index
    pred_prob.append(id_prob)

eval_df['id_pred_prob'] = pred_prob
#%%
# threshold calculation for each id
start = time.time()
threshold_df = eval_df.groupby('id')['id_pred_prob'].min()
id_threshold_dic = dict(threshold_df)
end = time.time()
tot_time = end-start
print('groupby time : ', tot_time )
#%%
# density plot
df_id = eval_df[eval_df.id=='4E7']
plt.figure(figsize=(10,6), dpi=80)
plt.title('density plot', fontsize=16)
sns.distplot(df_id['id_pred_prob'], bins = 20, kde= True, color = 'blue');
plt.show()

#%%

# attacks = [ID_fuzzing, ID_speedometer, ID_max_speedometer_mas, ID_corr_sig, ID_corr_sig_mas, ID_reverse_light_on,
#            ID_reverse_light_on_mas, ID_reverse_light_off, ID_reverse_light_off_mas, ID_max_engine,
#            ID_max_engine_mas]

# attacks = [ID_max_engine, ID_max_engine_mas]
#
# for i in attacks:
#     # convert df data into a series
#     print('Attack : ', i.name)
#     df = i
#     BenId = pd.Series(df['id'])
#     BenId = BenId.str.cat(sep=' ')
#
#     sequence_data = tokenizer.texts_to_sequences([BenId])[0]
#     X, y = seq(sequence_data)
#     eval_df = df[3:-3]
#
#     start = time.time()
#     pred_RPM = model.predict(X)
#     end = time.time()
#     tot_time = end - start
#     print('Total time :', tot_time)
#     print('Time for one frame :', tot_time / len(ID_speedometer))
#
#
#     true_window, pred_window = results_evaluation(pred_RPM, eval_df, y, winRatio)
#     print(classification_report(true_window, pred_window))
#     cm = confusion_matrix(true_window, pred_window)
#     print(cm)
#     print('New attack data loading....')
#     print('**************************************************')
#     print()

#%%
