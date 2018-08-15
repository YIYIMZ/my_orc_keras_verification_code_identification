# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 21:20:32 2018

@author: yy
"""

import os,sys,string
import cv2
import numpy as np

#识别字符集
char_set = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
#定义识别字符串的最大长度
seq_len=8
label_count=len(char_set)+1
image_size = (128, 32)

def get_label(filepath):
    lab=[]
    for num in str(os.path.split(filepath)[-1]).split('.')[0].split('_')[-1]:
        lab.append(int(char_set.find(num)))
    if len(lab) < seq_len:
        cur_seq_len = len(lab)
        for i in range(seq_len - cur_seq_len):
            lab.append(label_count) #
    return lab

def get_image_data_ctc(dir='./img_data/ctc/', file_list=[]):
    
    img_height, img_width = image_size[1], image_size[0]
    dir_path = dir
    for rt, dirs, files in os.walk(dir_path): 
        for filename in files:
            # print (filename)
            if filename.find('.') >= 0:
                (shotname, extension) = os.path.splitext(filename)
                # print shotname,extension
                if extension == '.jpg':  # extension == '.png' or
                    file_list.append(os.path.join('%s\\%s' % (rt, filename)))
                    # print (filename)

    print(len(file_list))
    index = 0
    X = []
    Y = []
    for file in file_list:

        index += 1
        img = cv2.imread(file, 0)
        
        img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
        img = cv2.transpose(img,(img_height,img_width))
        
        
        img =cv2.flip(img,1)
        img = (255 - img) / 256  # 反色处理
        X.append([img])
        Y.append(get_label(file))

    X = np.transpose(X, (0, 2, 3, 1))
    X = np.array(X)
    Y = np.array(Y)
    return X,Y

