# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 21:20:32 2018

@author: yy
"""

import os,sys,string
import sys
import logging
import multiprocessing
import time
import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import keras
import keras.backend as K
from keras.datasets import mnist
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras import backend as K
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



def get_label(filepath):
    # print(str(os.path.split(filepath)[-1]).split('.')[0].split('_')[-1])
    lab=[]
    for num in str(os.path.split(filepath)[-1]).split('.')[0].split('_')[-1]:
        lab.append(int(char_ocr.find(num)))
    if len(lab) < seq_len:
        cur_seq_len = len(lab)
        for i in range(seq_len - cur_seq_len):
            lab.append(label_count) #
    return lab

def get_image_data_seq(dir=r'data\train', file_list=[]):
    dir_path = dir
    for rt, dirs, files in os.walk(dir_path):  # =pathDir
        for filename in files:
            # print (filename)
            if filename.find('.') >= 0:
                (shotname, extension) = os.path.splitext(filename)
                # print shotname,extension
                if extension == '.tif':  # extension == '.png' or
                    file_list.append(os.path.join('%s\\%s' % (rt, filename)))
                    # print (filename)

    print(len(file_list))
    index = 0
    X = []
    Y = []
    for file in file_list:

        index += 1
        # if index>1000:
        #     break
        # print(file)
        img = cv2.imread(file, 0)
        # print(np.shape(img))
        # cv2.namedWindow("the window")
        # cv2.imshow("the window",img)
        img = cv2.resize(img, (150, 50), interpolation=cv2.INTER_CUBIC)
        img = cv2.transpose(img,(50,150))
        img =cv2.flip(img,1)
        # cv2.namedWindow("the window")
        # cv2.imshow("the window",img)
        # cv2.waitKey()
        img = (255 - img) / 256  # 反色处理
        X.append([img])
        Y.append(get_label(file))
        # print(get_label(file))
        # print(np.shape(X))
        # print(np.shape(X))

    # print(np.shape(X))
    X = np.transpose(X, (0, 2, 3, 1))
    X = np.array(X)
    Y = np.array(Y)
    return X,Y

def get_image_data_vector(dir=r'data\train', file_list=[]):
    dir_path = dir
    for rt, dirs, files in os.walk(dir_path):  # =pathDir
        for filename in files:
            # print (filename)
            if filename.find('.') >= 0:
                (shotname, extension) = os.path.splitext(filename)
                # print shotname,extension
                if extension == '.tif':  # extension == '.png' or
                    file_list.append(os.path.join('%s\\%s' % (rt, filename)))
                    # print (filename)

    print(len(file_list))
    index = 0
    X = []
    Y = []
    for file in file_list:

        index += 1
        # if index>1000:
        #     break
        # print(file)
        img = cv2.imread(file, 0)
        # print(np.shape(img))
        # cv2.namedWindow("the window")
        # cv2.imshow("the window",img)
        img = cv2.resize(img, (150, 50), interpolation=cv2.INTER_CUBIC)
        img = cv2.transpose(img,(50,150))
        img =cv2.flip(img,1)
        # cv2.namedWindow("the window")
        # cv2.imshow("the window",img)
        # cv2.waitKey()
        img = (255 - img) / 256  # 反色处理
        X.append([img])
        label = get_label(file)
        Y.append(sparse_tuple_from(label))
        # print(get_label(file))
        # print(np.shape(X))
        # print(np.shape(X))

    # print(np.shape(X))
    X = np.transpose(X, (0, 2, 3, 1))
    X = np.array(X)
    Y = np.array(Y)
    return X,Y

def gen_text(self, is_ran=False):
    text = ''
    vecs = np.zeros((self.max_size * self.len))
    
    #唯一变化，随机设定长度
    if is_ran == True:
        size = random.randint(1, self.max_size)
    else:
        size = self.max_size
        
    for i in range(size):
        c = random.choice(self.char_set)
        vec = self.char2vec(c)
        text = text + c
        vecs[i*self.len:(i+1)*self.len] = np.copy(vec)
    return text,vecs

# 生成一个训练batch
def get_next_batch(batch_size=128):
    obj = gen_id_card()
    #(batch_size,256,32)
    inputs = np.zeros([batch_size, OUTPUT_SHAPE[1],OUTPUT_SHAPE[0]])
    codes = []

    for i in range(batch_size):
        #生成不定长度的字串
        image, text, vec = obj.gen_image(True)
        #np.transpose 矩阵转置 (32*256,) => (32,256) => (256,32)
        inputs[i,:] = np.transpose(image.reshape((OUTPUT_SHAPE[0],OUTPUT_SHAPE[1])))
        #标签转成列表保存在codes
        codes.append(list(text))
    #比如batch_size=2，两条数据分别是"12"和"1"，则targets [['1','2'],['1']]
    targets = [np.asarray(i) for i in codes]
    #targets转成稀疏矩阵
    sparse_targets = sparse_tuple_from(targets)
    #(batch_size,) sequence_length值都是256，最大划分列数
    seq_len = np.ones(inputs.shape[0]) * OUTPUT_SHAPE[1]

    return inputs, sparse_targets, seq_len

#转化一个序列列表为稀疏矩阵    
def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []
    
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), xrange(len(seq))))
        values.extend(seq)
 
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
 
    return indices, values, shape

def decode_sparse_tensor(sparse_tensor):
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    result = []
    for index in decoded_indexes:
        result.append(decode_a_seq(index, sparse_tensor))
    return result
    
def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        str = DIGITS[spars_tensor[1][m]]
        decoded.append(str)
    return decoded




