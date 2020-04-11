# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 17:03:23 2020

@author: Ratul
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense,MaxPooling2D,Dropout,Flatten,Conv2D,LSTM,Embedding
from matplotlib import pyplot as plt
from keras.preprocessing.sequence import pad_sequences

from keras.datasets import imdb
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=20000)

X_train=pad_sequences(x_train,maxlen=200)
X_test=pad_sequences(x_test,maxlen=200)

X_train.shape



model=Sequential()
model.add(Embedding(20000,128,input_shape=(200,)))
model.add(LSTM(120,activation='tanh'))
model.add(Dropout(0.4))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=5,batch_size=64,validation_data=[X_test,y_test])



