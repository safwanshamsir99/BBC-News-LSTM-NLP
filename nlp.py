# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:09:37 2022

@author: safwanshamsir99
"""

import pandas as pd
import os
import numpy as np
import json
import pickle
import datetime

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard

from module_nlp import EDA,ModelCreation,ModelEvaluation

#%% STATIC
URL_PATH = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_FOLDER_PATH = os.path.join(os.getcwd(),'logs',log_dir)
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')
TOKENIZER_PATH = os.path.join(os.getcwd(),'tokenizer_category.json')
OHE_PATH = os.path.join(os.getcwd(),'ohe.pkl')

vocab_size = 10000
oov_token = 'OOV'
max_len = 320

#%% DATA LOADING
df = pd.read_csv(URL_PATH)

#%% CHECKPOINT
df_copy = df.copy() # backup
# df = df_copy.copy() # check point

#%% DATA INSPECTION
df.head(10)
df.tail(10)
df.info()

df['category'].unique() # to get the unique target
df['text'][2222]
df['category'][2222]
df.duplicated().sum() # 99 duplicated data

#%% DATA CLEANING
# single letter that have to be removed
# numbers can be filtered
# need to remove duplicated data

df = df.drop_duplicates() # remove duplicate data

text = df['text'].values # features X
category = df['category'].values # target y

eda = EDA()
eda.remove_single_letter(text=text) # letter such as single s,single t,etc
eda.remove_number(text=text)

#%% FEATURES SELECTION
# nothing to select

#%% PREPROCESSING 
#(convert into lower case,tokenization,padding and truncating,OHE)
# tokenization
tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)

tokenizer.fit_on_texts(text)
word_index = tokenizer.word_index
print(word_index)

train_sequences = tokenizer.texts_to_sequences(text) # to convert into numbers

# TOKENIZER SAVING
token_json = tokenizer.to_json()

with open(TOKENIZER_PATH,'w') as json_file:
    json.dump(token_json,json_file)

# padding and truncating
length_of_review = [len(i) for i in train_sequences] # list comprehension
print(np.median(length_of_review)) # to get the number of max length for padding

padded_review = pad_sequences(train_sequences,maxlen=max_len,
                              padding='post',truncating='post')

# One hot encoding
ohe = OneHotEncoder(sparse=False)
category = ohe.fit_transform(np.expand_dims(category,axis=-1))

# OHE SAVING
with open(OHE_PATH,'wb') as file:
    pickle.dump(ohe,file)

# train_test_split
X_train,X_test,y_train,y_test = train_test_split(padded_review,
                                                 category,
                                                 test_size=0.3,
                                                 random_state=3)

X_train = np.expand_dims(X_train,axis=-1)
X_test = np.expand_dims(X_test,axis=-1)

#%% MODEL DEVELOPMENT
# using LSTM layers, dropout, dense, input
# must achieve more than 70% f1 score
embedding_dim = 200
mc = ModelCreation()
model = mc.simple_lstm_model(max_len=max_len,
                             vocab_size=vocab_size,
                             embedding_dim=embedding_dim,
                             output=5)

plot_model(model,show_shapes=(True))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='acc')

# callbacks
tensorboard_callbacks = TensorBoard(log_dir=LOG_FOLDER_PATH)

hist = model.fit(X_train,y_train,batch_size=128,epochs=10,
                 validation_data=(X_test,y_test),
                 callbacks=(tensorboard_callbacks))

#%% MODEL ACCURACY AND LOSS PLOTTING
hist.history.keys()

me = ModelEvaluation()
me.plot_model_evaluation(hist)

#%% MODEL EVALUATION
y_true = np.argmax(y_test,axis=1)
y_pred = np.argmax(model.predict(X_test),axis=1)

cr = classification_report(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
acc_score = accuracy_score(y_true, y_pred)

print(cr)
print(cm)
print("Accuracy score: " + str(acc_score))

#%% MODEL SAVING
model.save(MODEL_SAVE_PATH)

#%% DISCUSSION/REPORTING
'''
The model achieved 92.8% accuracy score during evaluation process.

Recall and f1-score also reported a high percentage in range of 0.85 to 
0.97 and 0.88 to 0.96 respectively. 

However, the model started to overfit after 2nd epochs based on the 
graph displayed oon the Tensorboard.

To solve this problem, early stopping can be introduced to prevent 
overfitting and increasing the dropout data also can control 
the model from overfitting.
'''



















