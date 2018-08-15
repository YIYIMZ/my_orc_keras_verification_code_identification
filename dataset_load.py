# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 16:34:16 2018

@author: yy
"""

import sys
import cv2
import os
import numpy as np
import random
char_set = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
image_size = (128, 32)


#读取训练数据
images = []
labels = []
def read_path_cnn(data_path):    
    for dir_item in os.listdir(data_path):
        #从初始路径开始叠加，合并成可识别的操作路径
        full_path = os.path.abspath(os.path.join(data_path, dir_item))
        
        if os.path.isdir(full_path):    
            #如果是文件夹，继续递归调用
            read_path_cnn(full_path)
        else:   #文件 cv2.imread 能自动识别格式
            if dir_item.endswith('.jpg') or dir_item.endswith('.bmp'):
                #cv2.imread 能自动识别格式
                image = cv2.imread(full_path)                
                grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
                images.append(grey)
                
                #通过文件获取 标签
                label = get_label(dir_item)
                #将标签转为向量 one-hot
                label_parse_obj = label_parse()
                label_vector = label_parse_obj.text2vec(label)
                labels.append(label_vector)                                
#                print (label)    
#                print (label_parse_obj.vec2text(label_vector))    
    return images,labels

def get_label(file_full_name):
    file_name = file_full_name.split('.')
    name = file_name[0].split('_')
    return name[1]

class label_parse(object):
    def __init__(self):
       #self.words = open('AllWords.txt', 'r').read().split(' ')
       self.number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
       self.lowercase_leter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                     'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 
                     'u', 'v', 'w', 'x', 'y', 'z']
       self.capital_leter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                     'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                     'U', 'V', 'W', 'X', 'Y', 'Z']
       self.char_set = self.number + self.lowercase_leter + self.capital_leter
       #self.char_set = self.words + self.number
       self.len = len(self.char_set)
       
       self.max_size = 4
       
    #随机生成字串，长度固定
    #返回text,及对应的向量
    def random_text(self):
        text = ''
        vecs = np.zeros((self.max_size * self.len))
        #size = random.randint(1, self.max_size)
        size = self.max_size
        for i in range(size):
            c = random.choice(self.char_set)
            vec = self.char2vec(c)
            text = text + c
            vecs[i*self.len:(i+1)*self.len] = np.copy(vec)
        return text,vecs

    #单字转向量
    def char2vec(self, c):
        vec = np.zeros((self.len))
        for j in range(self.len):
            if self.char_set[j] == c:
                vec[j] = 1
        return vec
        
    #向量转文本
    def vec2text(self, vecs):
        text = ''
        v_len = len(vecs)
        for i in range(v_len):
            if(vecs[i] == 1):
                text = text + self.char_set[i % self.len]
        return text

    def text2vec(self,text):
        vecs = np.zeros((self.max_size * self.len))
        text_size = len(text)
        if text_size > self.max_size:
            raise ValueError("Text is too long!")
        for i in range(text_size):
            c = text[i]
            vec = self.char2vec(c)
            vecs[i*self.len:(i+1)*self.len] = np.copy(vec)
        return vecs

#从指定路径读取训练数据
def load_dataset_cnn(data_path):
    images,labels = read_path_cnn(data_path)    
    
    images = np.array(images)
    print(images.shape)    
    
    labels = np.array(labels)
    
    return images, labels
