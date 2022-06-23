# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:56:42 2022

@author: Acer
"""

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout
from tensorflow.keras.layers import Bidirectional,Embedding
from tensorflow.keras import Input
import re

class EDA():
    def __init__(self):
        pass
    
    def remove_single_letter(self,text):
        for index,rev in enumerate(text):
            # remove single character(s,t,etc)
            # ? dont be greedy
            text[index] = re.sub(r'(?:^| )\w(?:$| )',' ',rev).strip()
            return 
    
    def remove_number(self,text):
        for index,rev in enumerate(text):
            # remove numbers
            # ^ means NOT
            text[index] = re.sub('[^a-zA-Z]',' ',rev).split()
            return 

class ModelCreation():
    def __init__(self):
        pass
    
    def simple_lstm_model(self,max_len,vocab_size,embedding_dim,
                          output,node_num=128,drop_rate=0.2):
        model = Sequential()
        model.add(Input(shape=(max_len))) 
        model.add(Embedding(vocab_size,embedding_dim))
        model.add(Bidirectional(LSTM(embedding_dim,return_sequences=(True)))) # bidirectional
        model.add(Dropout(drop_rate))
        model.add(Bidirectional(LSTM(node_num))) # put 2nd bidirectional layer
        model.add(Dropout(drop_rate))
        model.add(Dense(node_num,activation='relu'))
        model.add(Dropout(drop_rate))
        model.add(Dense(output,activation='softmax')) # output needed
        model.summary()
        
        return model

class ModelEvaluation():
    def __init__(self):
        pass
    
    def plot_model_evaluation(self,hist):
        hist_keys = [i for i in hist.history.keys()]
        plt.figure()
        plt.plot(hist.history[hist_keys[0]])
        plt.plot(hist.history[hist_keys[2]])
        plt.legend(['train_loss','val_loss'])
        plt.title('Loss')
        plt.show()

        plt.figure()
        plt.plot(hist.history[hist_keys[1]])
        plt.plot(hist.history[hist_keys[3]])
        plt.legend(['train_acc','val_acc'])
        plt.title('Accuracy')
        plt.show()
        
