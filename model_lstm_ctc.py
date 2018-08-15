# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 09:11:33 2018

@author: yy
"""

import numpy as np
from sklearn.model_selection import train_test_split


import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate,Flatten,Dense,Dropout,GRU,LSTM,Add
from keras.regularizers import l2
import keras.backend as K

char_set = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
image_size = (128, 32)

IMAGE_HEIGHT = image_size[1]
IMAGE_WIDTH = image_size[0]


def get_lstm_ctc_model(image_size = image_size,
            seq_len = 8,#字符最大长度
            label_count = 63):#标签数量
    img_height, img_width = image_size[0], image_size[1]

    input_tensor = Input((img_height, img_width, 1))
    x = input_tensor
    for i in range(3):
        x = Conv2D(32*2**i, (3, 3), activation='relu', padding='same')(x)
        # x = Convolution2D(32*2**i, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    conv_shape = x.get_shape()
    # print(conv_shape)
    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)

    x = Dense(32, activation='relu')(x)

    lstm_1 = LSTM(32, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(x)
    lstm_1b = LSTM(32, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(x)
    lstm1_merged = Add()([lstm_1, lstm_1b])  

    lstm_2 = LSTM(32, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
    lstm_2b = LSTM(32, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(
        lstm1_merged)
    
    x = Concatenate()([lstm_2, lstm_2b])  
    x = Dropout(0.25)(x)
    x = Dense(label_count, kernel_initializer='he_normal', activation='softmax')(x)
    base_model = Model(inputs=input_tensor, outputs=x)

    labels = Input(name='the_labels', shape=[seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

    ctc_model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=[loss_out])
    ctc_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')
    ctc_model.summary()
    return conv_shape, base_model, ctc_model

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
  