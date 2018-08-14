# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 09:11:33 2018

@author: yy
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 21:21:46 2018

@author: yy
"""
import numpy as np
from sklearn.model_selection import train_test_split


import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate,Flatten,Dense,Dropout,GRU,LSTM,Add
from keras.regularizers import l2
import keras.backend as K


def get_gru_ctc_model(image_size,
            n_classes,
            seq_len,#字符最大长度
            label_count):#标签数量
    n_classes += 1 # 增加一个背景类
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

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

    lstm_1 = LSTM(32, return_sequences=True, kernel_initializer='he_normal', name='lstm_1')(x)
    lstm_1b = LSTM(32, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm_1b')(x)
    lstm1_merged = Add([lstm_1, lstm_1b])  

    lstm_2 = LSTM(32, return_sequences=True, kernel_initializer='he_normal', name='lstm_2')(lstm1_merged)
    lstm_2b = LSTM(32, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm_2b')(
        lstm1_merged)
    x = Concatenate([lstm_2, lstm_2b])  
    x = Dropout(0.25)(x)
    x = Dense(label_count, kernel_initializer='he_normal', activation='softmax')(x)

    labels = Input(name='the_labels', shape=[seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

    model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=[loss_out])
    
    return model

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    # y_pred = y_pred[:, 2:, :] 测试感觉没影响
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
  