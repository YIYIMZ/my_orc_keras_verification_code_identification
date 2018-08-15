# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 22:37:44 2018

@author: yy
"""
import os,sys
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback, EarlyStopping

# from keras.utils.visualize_util import plot
#from visual_callbacks import AccLossPlotter
#plotter = AccLossPlotter(graphs=['acc', 'loss'], save_graph=True, save_graph_path=sys.path[0])

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import SGD, Adam
from keras.models import load_model
from keras import backend as K

from dataset_split import Dataset
from dataset_load_ctc import get_image_data_ctc

from model_gru_ctc import get_gru_ctc_model
from model_lstm_ctc import get_lstm_ctc_model
from model_cnn import get_cnn_model

#识别字符集
char_set = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
#定义识别字符串的最大长度
seq_len=8
#识别结果集合个数 0-9
label_count=len(char_set)+1
image_size = (128, 32)

IMAGE_HEIGHT = image_size[1]
IMAGE_WIDTH = image_size[0]

#CNN网络模型类            
class Training_Predict:
    def __init__(self):
        self.base_model = None 
        self.ctc_model = None
        self.conv_shape = None
        
    #建立模型
    def build_model(self):
        #构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加，专业名称为序贯模型或线性堆叠模型
        self.conv_shape, self.base_model, self.ctc_model = get_lstm_ctc_model(image_size, seq_len, label_count) 
        
    def predict(self):
        file_list = []
        X, Y = get_image_data_ctc('./img_data/ctc_test/', file_list)
        y_pred = self.base_model.predict(X)
        shape = y_pred[:, :, :].shape  # 2:
        out = K.get_value(K.ctc_decode(y_pred[:, :, :], input_length=np.ones(shape[0]) * shape[1])[0][0])[:,:seq_len]  # 2:
        print()
        error_count=0
        for i in range(len(X)):
            print(file_list[i])
            str_src = str(os.path.split(file_list[i])[-1]).split('.')[0].split('_')[-1]
            print(out[i])
            str_out = ''.join([str(  char_set[x]   ) for x in out[i] if x!=-1 ])
            print(str_src, str_out)
            if str_src!=str_out:
                error_count+=1
                print('This is a error image---------------------------:',error_count)


    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []

        def on_epoch_end(self, epoch, logs=None):
            self.ctc_model.save_weights('ctc_model.w')
            self.base_model.save_weights('base_model.w')
            self.test()

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))

       
        #训练模型
    def train(self, batch_size = 32, nb_epoch = 15, data_augmentation = False):        
    
        X,Y=get_image_data_ctc(dir='./img_data/ctc/')
        print('train----------',X.shape,Y.shape)
        conv_shape = self.conv_shape
        
        
        maxin=2000
        result=self.ctc_model.fit([X[:maxin], Y[:maxin], np.array(np.ones(len(X))*int(conv_shape[1]))[:maxin], np.array(np.ones(len(X))*seq_len)[:maxin]], Y[:maxin],
                         batch_size=20,
                         epochs=200,
                         callbacks=[ EarlyStopping(patience=10)], #checkpointer, history,history, plotter,
                         validation_data=([X[maxin:], Y[maxin:], np.array(np.ones(len(X))*int(conv_shape[1]))[maxin:], np.array(np.ones(len(X))*seq_len)[maxin:]], Y[maxin:]),
                         )
        
        
        
        
        
    MODEL_PATH = './lstm.model.h5'
    def save_model(self, file_path = MODEL_PATH):
        self.base_model.save(file_path+'base')
        self.ctc_model.save(file_path+'ctc')
        
    def load_model(self, file_path = MODEL_PATH):
        self.base_model = load_model(file_path)
        
    def evaluate(self, dataset):
        score = self.base_model.evaluate(dataset.test_images, dataset.test_labels, verbose = 1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))
    
    
    
    
if __name__ == '__main__':
    
    #训练模型，这段代码不用，注释掉  
    model = Training_Predict()
    
    model.build_model()
    model.train()
    
    model.save_model(file_path = './model/lstm_ctc_model.h5')
    
#    model.load_model(file_path = './model/lstm_ctc_model.h5base')
    model.predict()
    '''
    #评估模型
    model = Model()
    model.load_model(file_path = './model/lstm_ctc_model.h5')
    model.evaluate(dataset)
    '''
 