# my_orc_keras_Identification_verification_code
本项目实现了ocr主流算法lstm+ctc+cnn架构，进行验证码识别，达到不分割字符而识别验证码内容的效果。验证码内容包含了大小字母以及数字。本项目技术能够训练长序列的ocr识别，更换数据集和相关调整，即可用于比如身份证号码、车牌、手机号、邮编等识别任务，也可用于汉字识别。

#环境
为方便初学者使用，环境在 window 下，tensorflow 用 CPU 版
1、 OS Windows 10 x64
2、 Python 3.6.2
3、 Tensorflow 1.8.0 CPU 版
4、 Keras 2.2.0
5、 Opencv 3.4.2

#数据
好多人问数据在哪，我真的想说数据就在源码中。dataset_gen_image.py 这个文件就是数据！

#欢迎大家一起学习讨论
