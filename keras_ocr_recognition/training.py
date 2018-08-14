# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 21:39:44 2018

@author: yy
"""

import numpy as np
from sklearn.model_selection import train_test_split

import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate,Flatten,Dense,Dropout,GRU,LSTM,Add
from keras.regularizers import l2
import keras.backend as K

from keras.optimizers import SGD
from keras.callbacks import *
from util import get_image_data_vector, get_image_data_seq,load_dataset_vector, load_dataset_seq
from model_gru_ctc import get_gru_ctc_model
from model_lstm_ctc import get_lstm_ctc_model
from model_cnn import get_cnn_model
# from keras.utils.visualize_util import plot
from visual_callbacks import AccLossPlotter
plotter = AccLossPlotter(graphs=['acc', 'loss'], save_graph=True, save_graph_path=sys.path[0])

#识别字符集
char_ocr='0123456789' #string.digits
#定义识别字符串的最大长度
seq_len=8
#识别结果集合个数 0-9
label_count=len(char_ocr)+1

#定义一些常量
#图片大小，32 x 256
OUTPUT_SHAPE = (32,256)

#训练最大轮次
num_epochs = 10000
#LSTM
num_hidden = 64
num_layers = 1

obj = gen_id_card()
num_classes = obj.len + 1 + 1  # 10位数字 + blank + ctc blank

#初始化学习速率
INITIAL_LEARNING_RATE = 1e-3
DECAY_STEPS = 5000
REPORT_STEPS = 100
LEARNING_RATE_DECAY_FACTOR = 0.9  # The learning rate decay factor
MOMENTUM = 0.9

DIGITS='0123456789'
BATCHES = 10
BATCH_SIZE = 64
TRAIN_SIZE = BATCHES * BATCH_SIZE

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        model.save_weights('model_1018.w')
        test(base_model)

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


    
def train():
    
    batch_size=20
    nb_epoch = 1000
    # checkpointer = ModelCheckpoint(filepath="keras_seq2seq_1018.hdf5", verbose=1, save_best_only=True, )
    history = LossHistory()
    
    dataset=load_dataset_vector()
    model = get_gru_ctc_model()
    #采用SGD+momentum的优化器进行训练，首先生成一个优化器对象  
    sgd = SGD(lr = 0.01, decay = 1e-6, 
              momentum = 0.9, nesterov = True) 
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
                       optimizer=sgd,
                       metrics=['accuracy'])   #完成实际的模型配置工作
    
    model.fit(dataset.train_images,
               dataset.train_labels,
               batch_size = batch_size,
               nb_epoch = nb_epoch,
               validation_data = (dataset.valid_images, dataset.valid_labels),
               shuffle = True)
            
    
    model.summary()

    
    
if __name__ == '__main__':
    train()