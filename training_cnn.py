# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 21:39:44 2018

@author: yy
"""
from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import SGD, Adam
from keras.models import load_model
from keras import backend as K

from dataset_split import Dataset
from model_gru_ctc import get_gru_ctc_model
from model_lstm_ctc import get_lstm_ctc_model
from model_cnn import get_cnn_model

char_set = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
image_size = (128, 32)

IMAGE_HEIGHT = image_size[1]
IMAGE_WIDTH = image_size[0]

#CNN网络模型类            
class Training_Predict:
    def __init__(self):
        self.model = None 
        
    #建立模型
    def build_model(self, dataset):
        #构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加，专业名称为序贯模型或线性堆叠模型
        self.model = get_cnn_model(dataset.input_shape) 
        
        #训练模型
    def train(self, dataset, batch_size = 32, nb_epoch = 15, data_augmentation = False):        
        #采用SGD+momentum的优化器进行训练，首先生成一个优化器对象  
#        sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True) 
        adam = Adam()
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=adam,
                           metrics=['accuracy'])   #完成实际的模型配置工作
        
        #不使用数据提升，所谓的提升就是从我们提供的训练数据中利用旋转、翻转、加噪声等方法创造新的
        #训练数据，有意识的提升训练数据规模，增加模型训练量
        if not data_augmentation:            
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size = batch_size,
                           nb_epoch = nb_epoch,
                           validation_data = (dataset.valid_images, dataset.valid_labels),
                           shuffle = True)
        #使用实时数据提升
        else:            
            #定义数据生成器用于数据提升，其返回一个生成器对象datagen，datagen每被调用一
            #次其生成一组数据（顺序生成），节省内存，其实就是python的数据生成器
            datagen = ImageDataGenerator(
                featurewise_center = False,             #是否使输入数据去中心化（均值为0），
                samplewise_center  = False,             #是否使输入数据的每个样本均值为0
                featurewise_std_normalization = False,  #是否数据标准化（输入数据除以数据集的标准差）
                samplewise_std_normalization  = False,  #是否将每个样本数据除以自身的标准差
                zca_whitening = False,                  #是否对输入数据施以ZCA白化
                rotation_range = 20,                    #数据提升时图片随机转动的角度(范围为0～180)
                width_shift_range  = 0.2,               #数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
                height_shift_range = 0.2,               #同上，只不过这里是垂直
                horizontal_flip = True,                 #是否进行随机水平翻转
                vertical_flip = False)                  #是否进行随机垂直翻转

            #计算整个训练样本集的数量以用于特征值归一化、ZCA白化等处理
            datagen.fit(dataset.train_images)                        

            #利用生成器开始训练模型
            self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                                   batch_size = batch_size),
                                     samples_per_epoch = dataset.train_images.shape[0],
                                     nb_epoch = nb_epoch,
                                     validation_data = (dataset.valid_images, dataset.valid_labels))
        
        
    MODEL_PATH = './cnn.model.h5'
    def save_model(self, file_path = MODEL_PATH):
        self.model.save(file_path)
 
    def load_model(self, file_path = MODEL_PATH):
        self.model = load_model(file_path)
        
    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose = 1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))
    
    
    #识别人脸
    def predict(self, image):    
        #依然是根据后端系统确定维度顺序
        if K.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_WIDTH, IMAGE_HEIGHT):
            image = image.reshape((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH))   #与模型训练不同，这次只是针对1张图片进行预测    
        elif K.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_WIDTH, IMAGE_HEIGHT, 3):
            image = image.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))                    
        
        #浮点并归一化
        image = image.astype('float32')
        image /= 255
        
        #给出输入属于各个类别的概率，我们是二值类别，则该函数会给出输入图像属于0和1的概率各为多少
        result = self.model.predict_proba(image)
        print('result:', result)
        
        #给出类别预测：0或者1
        result = self.model.predict_classes(image)        

        #返回类别预测结果
        return result[0]  
    
    
if __name__ == '__main__':
    dataset = Dataset('./img_data/all/')    
    dataset.load()
    #训练模型，这段代码不用，注释掉  
    model = Training_Predict()
    model.build_model(dataset)
    model.train(dataset)
    model.save_model(file_path = './model/cnn_model.h5')
    '''
    #评估模型
    model = Model()
    model.load_model(file_path = './model/cnn_model.h5')
    model.evaluate(dataset)
    '''
 