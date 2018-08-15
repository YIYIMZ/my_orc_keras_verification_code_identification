# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 08:43:26 2018

@author: yy
"""
from __future__ import division
from keras.models import Model
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate,Flatten,Dense,Dropout
from keras.regularizers import l2
import keras.backend as K
from dataset_load import label_parse
obj = label_parse()
#image,text,vec = obj.gen_image()

#图像大小
image_size = (128, 32)

IMAGE_HEIGHT = image_size[1]
IMAGE_WIDTH = image_size[0]

MAX_CAPTCHA = obj.max_size
CHAR_SET_LEN = obj.len

def get_cnn_model(image_size,
            n_classes = MAX_CAPTCHA*CHAR_SET_LEN
            ):
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]
    l2_reg = 0.0005 # L2 正则化
    
    
    x = Input(shape=(img_height, img_width, img_channels))
    
    ##网络
    conv1_1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1_1')(x)
    pool1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 1), padding='same', name='pool1')(conv1_1)

    conv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2_1')(pool1)
    pool2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 1), padding='same', name='pool2')(conv2_1)

    conv3_1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_1')(pool2)
    pool3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 1), padding='same', name='pool3')(conv3_1)

    conv4_1 = Conv2D(64, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_1')(pool3)
    flatten = Flatten()(conv4_1)
    fc4 = Dense(n_classes, activation='softmax', name='fc4')(flatten)

    model = Model(inputs=x, outputs=fc4)
    return model