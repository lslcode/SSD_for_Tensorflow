"""
date: 2017/11/10
author: lslcode [jasonli8848@qq.com]
"""

import numpy as np
import tensorflow as tf

class SSD300:
    def __init__(self, tf_sess, isTraining):
        # tensorflow session
        self.sess = tf_sess
        # 是否训练
        self.isTraining = isTraining
        # 允许的图像大小
        self.img_size = [300, 300]
        # 分类总数量
        self.classes_size = 21
        # 背景分类的值
        self.background_classes_val = self.classes_size - 1
        # 每个特征图单元的default box数量
        self.default_box_size = [4, 6, 6, 6, 4, 4]
        # default box 尺寸长宽比例
        self.box_aspect_ratio = [
            [1.0, 1.25, 2.0, 3.0],
            [1.0, 1.25, 2.0, 3.0, 1.0 / 2.0, 1.0 / 3.0],
            [1.0, 1.25, 2.0, 3.0, 1.0 / 2.0, 1.0 / 3.0],
            [1.0, 1.25, 2.0, 3.0, 1.0 / 2.0, 1.0 / 3.0],
            [1.0, 1.25, 2.0, 3.0],
            [1.0, 1.25, 2.0, 3.0]
        ]
        # 最小default box面积比例
        self.min_box_scale = 0.1
        # 最大default box面积比例
        self.max_box_scale = 0.9
        # 每个特征层的面积比例
        # numpy生成等差数组，效果等同于论文中的s_k=s_min+(s_max-s_min)*(k-1)/(m-1)
        self.default_box_scale = np.linspace(self.min_box_scale, self.max_box_scale, num = np.amax(self.default_box_size))
        print('##   default_box_scale:'+str(self.default_box_scale))
        # 卷积步长
        self.conv_strides_1 = [1, 1, 1, 1]
        self.conv_strides_2 = [1, 2, 2, 1]
        self.conv_strides_3 = [1, 3, 3, 1]
        # 池化窗口
        self.pool_size = [1, 2, 2, 1]
        # 池化步长
        self.pool_strides = [1, 2, 2, 1]
        # Batch Normalization 算法的 decay 参数
        self.conv_bn_decay = 0.999
        # Batch Normalization 算法的 variance_epsilon 参数
        self.conv_bn_epsilon = 0.0001
        # Jaccard相似度判断阀值
        self.jaccard_value = 0.5
        # 冗余小数，用于除法运算预防Nan
        self.extra_decimal = 1e-7

        # 初始化Tensorflow Graph
        self.generate_graph()
        
    def generate_graph(self):

        # 输入数据
        self.input = tf.placeholder(shape=[None, self.img_size[0], self.img_size[1], 3], dtype=tf.float32, name='input_image')

        # vvg16卷积层 1 相关参数权重
        self.conv_weight_1_1 = tf.Variable(tf.truncated_normal([3, 3,  3, 64], 0, 1), dtype=tf.float32, name='weight_1_1')
        self.conv_weight_1_2 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], 0, 1), dtype=tf.float32, name='weight_1_2')
        self.conv_bias_1_1 = tf.Variable(tf.truncated_normal([64], 0, 1), dtype=tf.float32, name='bias_1_1')
        self.conv_bias_1_2 = tf.Variable(tf.truncated_normal([64], 0, 1), dtype=tf.float32, name='bias_1_2')        
    
        # vvg16卷积层 2 相关参数权重
        self.conv_weight_2_1 = tf.Variable(tf.truncated_normal([3, 3,  64, 128], 0, 1), dtype=tf.float32, name='weight_2_1')
        self.conv_weight_2_2 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], 0, 1), dtype=tf.float32, name='weight_2_2')
        self.conv_bias_2_1 = tf.Variable(tf.truncated_normal([128], 0, 1), dtype=tf.float32, name='bias_2_1')
        self.conv_bias_2_2 = tf.Variable(tf.truncated_normal([128], 0, 1), dtype=tf.float32, name='bias_2_2')        
    
        # vvg16卷积层 3 相关参数权重
        self.conv_weight_3_1 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], 0, 1), dtype=tf.float32, name='weight_3_1')
        self.conv_weight_3_2 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], 0, 1), dtype=tf.float32, name='weight_3_2')
        self.conv_weight_3_3 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], 0, 1), dtype=tf.float32, name='weight_3_3')
        self.conv_bias_3_1 = tf.Variable(tf.truncated_normal([256], 0, 1), dtype=tf.float32, name='bias_3_1')
        self.conv_bias_3_2 = tf.Variable(tf.truncated_normal([256], 0, 1), dtype=tf.float32, name='bias_3_2')
        self.conv_bias_3_3 = tf.Variable(tf.truncated_normal([256], 0, 1), dtype=tf.float32, name='bias_3_3')        
    
        # vvg16卷积层 4 相关参数权重
        self.conv_weight_4_1 = tf.Variable(tf.truncated_normal([3, 3, 256, 512], 0, 1), dtype=tf.float32, name='weight_4_1')
        self.conv_weight_4_2 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], 0, 1), dtype=tf.float32, name='weight_4_2')
        self.conv_weight_4_3 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], 0, 1), dtype=tf.float32, name='weight_4_3')
        self.conv_bias_4_1 = tf.Variable(tf.truncated_normal([512], 0, 1), dtype=tf.float32, name='bias_4_1')
        self.conv_bias_4_2 = tf.Variable(tf.truncated_normal([512], 0, 1), dtype=tf.float32, name='bias_4_2')
        self.conv_bias_4_3 = tf.Variable(tf.truncated_normal([512], 0, 1), dtype=tf.float32, name='bias_4_3')        
    
        # vvg16卷积层 5 相关参数权重
        self.conv_weight_5_1 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], 0, 1), dtype=tf.float32, name='weight_5_1')
        self.conv_weight_5_2 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], 0, 1), dtype=tf.float32, name='weight_5_2')
        self.conv_weight_5_3 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], 0, 1), dtype=tf.float32, name='weight_5_3')
        self.conv_bias_5_1 = tf.Variable(tf.truncated_normal([512], 0, 1), dtype=tf.float32, name='bias_5_1')
        self.conv_bias_5_2 = tf.Variable(tf.truncated_normal([512], 0, 1), dtype=tf.float32, name='bias_5_2')
        self.conv_bias_5_3 = tf.Variable(tf.truncated_normal([512], 0, 1), dtype=tf.float32, name='bias_5_3')        
    
        # 卷积层 6 相关参数权重
        self.conv_weight_6_1 = tf.Variable(tf.truncated_normal([3, 3, 512, 1024], 0, 1), dtype=tf.float32, name='weight_6_1')
        self.conv_bias_6_1 = tf.Variable(tf.truncated_normal([1024], 0, 1), dtype=tf.float32, name='bias_6_1')        
    
        # 卷积层 7 相关参数权重
        self.conv_weight_7_1 = tf.Variable(tf.truncated_normal([1, 1, 1024, 1024], 0, 1), dtype=tf.float32, name='weight_7_1')
        self.conv_bias_7_1 = tf.Variable(tf.truncated_normal([1024], 0, 1), dtype=tf.float32, name='bias_7_1')        
    
        # 卷积层 8 相关参数权重
        self.conv_weight_8_1 = tf.Variable(tf.truncated_normal([1, 1, 1024, 256], 0, 1), dtype=tf.float32, name='weight_8_1')
        self.conv_weight_8_2 = tf.Variable(tf.truncated_normal([3, 3,  256, 512], 0, 1), dtype=tf.float32, name='weight_8_2')
        self.conv_bias_8_1 = tf.Variable(tf.truncated_normal([256], 0, 1), dtype=tf.float32, name='bias_8_1')
        self.conv_bias_8_2 = tf.Variable(tf.truncated_normal([512], 0, 1), dtype=tf.float32, name='bias_8_2')        
    
        # 卷积层 9 相关参数权重
        self.conv_weight_9_1 = tf.Variable(tf.truncated_normal([1, 1, 512, 128], 0, 1), dtype=tf.float32, name='weight_9_1')
        self.conv_weight_9_2 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], 0, 1), dtype=tf.float32, name='weight_9_2')
        self.conv_bias_9_1 = tf.Variable(tf.truncated_normal([128], 0, 1), dtype=tf.float32, name='bias_9_1')
        self.conv_bias_9_2 = tf.Variable(tf.truncated_normal([256], 0, 1), dtype=tf.float32, name='bias_9_2')        
    
        # 卷积层 10 相关参数权重
        self.conv_weight_10_1 = tf.Variable(tf.truncated_normal([1, 1, 256, 128], 0, 1), dtype=tf.float32, name='weight_10_1')
        self.conv_weight_10_2 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], 0, 1), dtype=tf.float32, name='weight_10_2')
        self.conv_bias_10_1 = tf.Variable(tf.truncated_normal([128], 0, 1), dtype=tf.float32, name='bias_10_1')
        self.conv_bias_10_2 = tf.Variable(tf.truncated_normal([256], 0, 1), dtype=tf.float32, name='bias_10_2')
    
        # 卷积层 11 相关参数权重
        self.conv_weight_11_1 = tf.Variable(tf.truncated_normal([1, 1, 256, 128], 0, 1), dtype=tf.float32, name='weight_11_1')
        self.conv_weight_11_2 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], 0, 1), dtype=tf.float32, name='weight_11_2')
        self.conv_bias_11_1 = tf.Variable(tf.truncated_normal([128], 0, 1), dtype=tf.float32, name='bias_11_1')
        self.conv_bias_11_2 = tf.Variable(tf.truncated_normal([256], 0, 1), dtype=tf.float32, name='bias_11_2')

        # ssd卷积特征层 相关参数权重
        self.features_weight_1 = tf.Variable(tf.truncated_normal([3, 3, 512,  self.default_box_size[0] * (self.classes_size + 4)], 0, 1), name='features_weight_1')
        self.features_bias_1 = tf.Variable(tf.truncated_normal([self.default_box_size[0] * (self.classes_size + 4)], 0, 1), name='features_bias_1')
        self.features_weight_2 = tf.Variable(tf.truncated_normal([3, 3, 1024, self.default_box_size[1] * (self.classes_size + 4)], 0, 1), name='features_weight_2')
        self.features_bias_2 = tf.Variable(tf.truncated_normal([self.default_box_size[1] * (self.classes_size + 4)], 0, 1), name='features_bias_2')
        self.features_weight_3 = tf.Variable(tf.truncated_normal([3, 3, 512,  self.default_box_size[2] * (self.classes_size + 4)], 0, 1), name='features_weight_3')
        self.features_bias_3 = tf.Variable(tf.truncated_normal([self.default_box_size[2] * (self.classes_size + 4)], 0, 1), name='features_bias_3')
        self.features_weight_4 = tf.Variable(tf.truncated_normal([3, 3, 256,  self.default_box_size[3] * (self.classes_size + 4)], 0, 1), name='features_weight_4')
        self.features_bias_4 = tf.Variable(tf.truncated_normal([self.default_box_size[3] * (self.classes_size + 4)], 0, 1), name='features_bias_4')
        self.features_weight_5 = tf.Variable(tf.truncated_normal([3, 3, 256,  self.default_box_size[4] * (self.classes_size + 4)], 0, 1), name='features_weight_5')
        self.features_bias_5 = tf.Variable(tf.truncated_normal([self.default_box_size[4] * (self.classes_size + 4)], 0, 1), name='features_bias_5')
        self.features_weight_6 = tf.Variable(tf.truncated_normal([1, 1, 256,  self.default_box_size[5] * (self.classes_size + 4)], 0, 1), name='features_weight_6')
        self.features_bias_6 = tf.Variable(tf.truncated_normal([self.default_box_size[5] * (self.classes_size + 4)], 0, 1), name='features_bias_6')
    
        # vvg16卷积层 1
        self.conv_1_1 = tf.nn.conv2d(self.input, self.conv_weight_1_1, self.conv_strides_1, padding='SAME', name='conv_1_1')
        self.conv_1_1 = tf.nn.relu(tf.add(self.conv_1_1, self.conv_bias_1_1), name='relu_1_1')
        self.conv_1_1 = self.batch_normalization(self.conv_1_1)
        self.conv_1_2 = tf.nn.conv2d(self.conv_1_1, self.conv_weight_1_2, self.conv_strides_1, padding='SAME', name='conv_1_2')
        self.conv_1_2 = tf.nn.relu(tf.add(self.conv_1_2, self.conv_bias_1_2), name='relu_1_2')
        self.conv_1_2 = self.batch_normalization(self.conv_1_2)
        self.conv_1_2 = tf.nn.max_pool(self.conv_1_2, self.pool_size, self.pool_strides, padding='SAME',   name='pool_1_2')
        print('##   conv_1_2 shape: ' + str(self.conv_1_2.get_shape().as_list()))
        # vvg16卷积层 2
        self.conv_2_1 = tf.nn.conv2d(self.conv_1_2, self.conv_weight_2_1, self.conv_strides_1, padding='SAME', name='conv_2_1')
        self.conv_2_1 = tf.nn.relu(tf.add(self.conv_2_1, self.conv_bias_2_1), name='relu_2_1')
        self.conv_2_1 = self.batch_normalization(self.conv_2_1)
        self.conv_2_2 = tf.nn.conv2d(self.conv_2_1, self.conv_weight_2_2, self.conv_strides_1, padding='SAME', name='conv_2_2')
        self.conv_2_2 = tf.nn.relu(tf.add(self.conv_2_2, self.conv_bias_2_2), name='relu_2_2')
        self.conv_2_2 = self.batch_normalization(self.conv_2_2)
        self.conv_2_2 = tf.nn.max_pool(self.conv_2_2, self.pool_size, self.pool_strides, padding='SAME',   name='pool_2_2')
        print('##   conv_2_2 shape: ' + str(self.conv_2_2.get_shape().as_list()))
        # vvg16卷积层 3
        self.conv_3_1 = tf.nn.conv2d(self.conv_2_2, self.conv_weight_3_1, self.conv_strides_1, padding='SAME', name='conv_3_1')
        self.conv_3_1 = tf.nn.relu(tf.add(self.conv_3_1, self.conv_bias_3_1), name='relu_3_1')
        self.conv_3_1 = self.batch_normalization(self.conv_3_1)
        self.conv_3_2 = tf.nn.conv2d(self.conv_3_1, self.conv_weight_3_2, self.conv_strides_1, padding='SAME', name='conv_3_2')
        self.conv_3_2 = tf.nn.relu(tf.add(self.conv_3_2, self.conv_bias_3_2), name='relu_3_2')
        self.conv_3_2 = self.batch_normalization(self.conv_3_2)
        #self.conv_3_3 = tf.nn.conv2d(self.conv_3_2, self.conv_weight_3_3, self.conv_strides_1, padding='SAME', name='conv_3_3')
        #self.conv_3_3 = tf.nn.relu(tf.add(self.conv_3_3, self.conv_bias_3_3), name='relu_3_3')
        #self.conv_3_3 = self.batch_normalization(self.conv_3_3)
        #self.conv_3_3 = tf.nn.max_pool(self.conv_3_3, self.pool_size, self.pool_strides, padding='SAME', name='pool_3_3')
        print('##   conv_3_2 shape: ' + str(self.conv_3_2.get_shape().as_list()))
        # vvg16卷积层 4
        self.conv_4_1 = tf.nn.conv2d(self.conv_3_2, self.conv_weight_4_1, self.conv_strides_1, padding='SAME', name='conv_4_1')
        self.conv_4_1 = tf.nn.relu(tf.add(self.conv_4_1, self.conv_bias_4_1), name='relu_4_1')
        self.conv_4_1 = self.batch_normalization(self.conv_4_1)
        self.conv_4_2 = tf.nn.conv2d(self.conv_4_1, self.conv_weight_4_2, self.conv_strides_1, padding='SAME', name='conv_4_2')
        self.conv_4_2 = tf.nn.relu(tf.add(self.conv_4_2, self.conv_bias_4_2), name='relu_4_2')
        self.conv_4_2 = self.batch_normalization(self.conv_4_2)
        self.conv_4_3 = tf.nn.conv2d(self.conv_4_2, self.conv_weight_4_3, self.conv_strides_1, padding='SAME', name='conv_4_3')
        self.conv_4_3 = tf.nn.relu(tf.add(self.conv_4_3, self.conv_bias_4_3), name='relu_4_3')
        self.conv_4_3 = self.batch_normalization(self.conv_4_3)
        self.conv_4_3 = tf.nn.max_pool(self.conv_4_3, self.pool_size, self.pool_strides, padding='SAME',   name='pool_4_3')
        print('##   conv_4_3 shape: ' + str(self.conv_4_3.get_shape().as_list()))
        # vvg16卷积层 5
        self.conv_5_1 = tf.nn.conv2d(self.conv_4_3, self.conv_weight_5_1, self.conv_strides_1, padding='SAME', name='conv_5_1')
        self.conv_5_1 = tf.nn.relu(tf.add(self.conv_5_1, self.conv_bias_5_1), name='relu_5_1')
        self.conv_5_1 = self.batch_normalization(self.conv_5_1)
        self.conv_5_2 = tf.nn.conv2d(self.conv_5_1, self.conv_weight_5_2, self.conv_strides_1, padding='SAME', name='conv_5_2')
        self.conv_5_2 = tf.nn.relu(tf.add(self.conv_5_2, self.conv_bias_5_2), name='relu_5_2')
        self.conv_5_2 = self.batch_normalization(self.conv_5_2)
        self.conv_5_3 = tf.nn.conv2d(self.conv_5_2, self.conv_weight_5_3, self.conv_strides_1, padding='SAME', name='conv_5_3')
        self.conv_5_3 = tf.nn.relu(tf.add(self.conv_5_3, self.conv_bias_5_3), name='relu_5_3')
        self.conv_5_3 = self.batch_normalization(self.conv_5_3)
        self.conv_5_3 = tf.nn.max_pool(self.conv_5_3, self.pool_size, self.pool_strides, padding='SAME',   name='pool_5_3')
        print('##   conv_5_3 shape: ' + str(self.conv_5_3.get_shape().as_list()))
        # ssd卷积层 6
        self.conv_6_1 = tf.nn.conv2d(self.conv_5_3, self.conv_weight_6_1, self.conv_strides_1, padding='SAME', name='conv_6_1')
        self.conv_6_1 = tf.nn.relu(tf.add(self.conv_6_1, self.conv_bias_6_1), name='relu_6_1')
        self.conv_6_1 = self.batch_normalization(self.conv_6_1)
        print('##   conv_6_1 shape: ' + str(self.conv_6_1.get_shape().as_list()))
        # ssd卷积层 7
        self.conv_7_1 = tf.nn.conv2d(self.conv_6_1, self.conv_weight_7_1, self.conv_strides_1, padding='SAME', name='conv_7_1')
        self.conv_7_1 = tf.nn.relu(tf.add(self.conv_7_1, self.conv_bias_7_1), name='relu_7_1')
        self.conv_7_1 = self.batch_normalization(self.conv_7_1)
        print('##   conv_7_1 shape: ' + str(self.conv_7_1.get_shape().as_list()))
        # ssd卷积层 8
        self.conv_8_1 = tf.nn.conv2d(self.conv_7_1, self.conv_weight_8_1, self.conv_strides_1, padding='SAME', name='conv_8_1')
        self.conv_8_1 = tf.nn.relu(tf.add(self.conv_8_1, self.conv_bias_8_1), name='relu_8_1')
        self.conv_8_1 = self.batch_normalization(self.conv_8_1)
        self.conv_8_2 = tf.nn.conv2d(self.conv_8_1, self.conv_weight_8_2, self.conv_strides_2, padding='SAME', name='conv_8_2')
        self.conv_8_2 = tf.nn.relu(tf.add(self.conv_8_2, self.conv_bias_8_2), name='relu_8_2')
        self.conv_8_2 = self.batch_normalization(self.conv_8_2)
        print('##   conv_8_2 shape: ' + str(self.conv_8_2.get_shape().as_list()))
        # ssd卷积层 9
        self.conv_9_1 = tf.nn.conv2d(self.conv_8_2, self.conv_weight_9_1, self.conv_strides_1, padding='SAME', name='conv_9_1')
        self.conv_9_1 = tf.nn.relu(tf.add(self.conv_9_1, self.conv_bias_9_1), name='relu_9_1')
        self.conv_9_1 = self.batch_normalization(self.conv_9_1)
        self.conv_9_2 = tf.nn.conv2d(self.conv_9_1, self.conv_weight_9_2, self.conv_strides_2, padding='SAME', name='conv_9_2')
        self.conv_9_2 = tf.nn.relu(tf.add(self.conv_9_2, self.conv_bias_9_2), name='relu_9_2')
        self.conv_9_2 = self.batch_normalization(self.conv_9_2)
        print('##   conv_9_2 shape: ' + str(self.conv_9_2.get_shape().as_list()))
        # ssd卷积层 10
        self.conv_10_1 = tf.nn.conv2d(self.conv_9_2, self.conv_weight_10_1, self.conv_strides_1, padding='SAME', name='conv_10_1')
        self.conv_10_1 = tf.nn.relu(tf.add(self.conv_10_1, self.conv_bias_10_1), name='relu_10_1')
        self.conv_10_1 = self.batch_normalization(self.conv_10_1)
        self.conv_10_2 = tf.nn.conv2d(self.conv_10_1, self.conv_weight_10_2, self.conv_strides_2, padding='SAME', name='conv_10_2')
        self.conv_10_2 = tf.nn.relu(tf.add(self.conv_10_2, self.conv_bias_10_2), name='relu_10_2')
        self.conv_10_2 = self.batch_normalization(self.conv_10_2)
        print('##   conv_10_2 shape: ' + str(self.conv_10_2.get_shape().as_list()))
        # ssd卷积层 11
        self.conv_11_1 = tf.nn.conv2d(self.conv_10_2, self.conv_weight_11_1, self.conv_strides_1, padding='SAME', name='conv_11_1')
        self.conv_11_1 = tf.nn.relu(tf.add(self.conv_11_1, self.conv_bias_11_1), name='relu_11_1')
        self.conv_11_1 = self.batch_normalization(self.conv_11_1)
        self.conv_11_2 = tf.nn.conv2d(self.conv_11_1, self.conv_weight_11_2, self.conv_strides_3, padding='SAME', name='conv_11_2')
        self.conv_11_2 = tf.nn.relu(tf.add(self.conv_11_2, self.conv_bias_11_2), name='relu_11_2')
        self.conv_11_2 = self.batch_normalization(self.conv_11_2)
        print('##   conv_11_2 shape: ' + str(self.conv_11_2.get_shape().as_list()))

        # 第 1 层 特征层，来源于conv_4_3
        self.features_1 = tf.nn.conv2d(self.conv_4_3, self.features_weight_1, self.conv_strides_1, padding='SAME', name='conv_features_1')
        self.features_1 = tf.nn.relu(tf.add(self.features_1, self.features_bias_1),name='relu_features_1')
        self.features_1 = self.batch_normalization(self.features_1)
        print('##   features_1 shape: ' + str(self.features_1.get_shape().as_list()))
        # 第 2 层 特征层，来源于conv_7_1
        self.features_2 = tf.nn.conv2d(self.conv_7_1, self.features_weight_2, self.conv_strides_1, padding='SAME', name='conv_features_2')
        self.features_2 = tf.nn.relu(tf.add(self.features_2, self.features_bias_2),name='relu_features_2')
        self.features_2 = self.batch_normalization(self.features_2)
        print('##   features_2 shape: ' + str(self.features_2.get_shape().as_list()))
        # 第 3 层 特征层，来源于conv_8_2
        self.features_3 = tf.nn.conv2d(self.conv_8_2, self.features_weight_3, self.conv_strides_1, padding='SAME', name='conv_features_3')
        self.features_3 = tf.nn.relu(tf.add(self.features_3, self.features_bias_3),name='relu_features_3')
        self.features_3 = self.batch_normalization(self.features_3)
        print('##   features_3 shape: ' + str(self.features_3.get_shape().as_list()))
        # 第 4 层 特征层，来源于conv_9_2
        self.features_4 = tf.nn.conv2d(self.conv_9_2, self.features_weight_4, self.conv_strides_1, padding='SAME', name='conv_features_4')
        self.features_4 = tf.nn.relu(tf.add(self.features_4, self.features_bias_4),name='relu_features_4')
        self.features_4 = self.batch_normalization(self.features_4)
        print('##   features_4 shape: ' + str(self.features_4.get_shape().as_list()))
        # 第 5 层 特征层，来源于conv_10_2
        self.features_5 = tf.nn.conv2d(self.conv_10_2,self.features_weight_5, self.conv_strides_1, padding='SAME', name='conv_features_5')
        self.features_5 = tf.nn.relu(tf.add(self.features_5, self.features_bias_5),name='relu_features_5')
        self.features_5 = self.batch_normalization(self.features_5)
        print('##   features_5 shape: ' + str(self.features_5.get_shape().as_list()))
        # 第 6 层 特征层，来源于conv_11_2
        self.features_6 = tf.nn.conv2d(self.conv_11_2,self.features_weight_6, self.conv_strides_1, padding='SAME', name='conv_features_6')  
        self.features_6 = tf.nn.relu(tf.add(self.features_6, self.features_bias_6),name='relu_features_6')
        self.features_6 = self.batch_normalization(self.features_6)
        print('##   features_6 shape: ' + str(self.features_6.get_shape().as_list()))
        
        # 特征层集合
        self.feature_maps = [self.features_1, self.features_2, self.features_3, self.features_4, self.features_5, self.features_6]
        # 获取卷积后各个特征层的shape,以便生成feature和groundtruth格式一致的训练数据
        self.feature_maps_shape = [m.get_shape().as_list() for m in self.feature_maps]
        
        # 整理feature数据
        self.tmp_all_feature = []
        for i, fmap in zip(range(len(self.feature_maps)), self.feature_maps):
            width = self.feature_maps_shape[i][1]
            height = self.feature_maps_shape[i][2]
            # 这里reshape目的为定位和类别2方面回归作准备
            # reshape前 shape=[None, width, height, default_box*(classes+4)]
            # reshape后 shape=[None, width*height*default_box, (classes+4) ]
            self.tmp_all_feature.append(tf.reshape(fmap, [-1, (width * height * self.default_box_size[i]) , (self.classes_size + 4)]))
        # 合并每张图像产生的所有特征
        self.tmp_all_feature = tf.concat(self.tmp_all_feature, axis=1)
        # 这里正式拆分为定位和类别2类数据
        self.feature_class = self.tmp_all_feature[:,:,:self.classes_size]
        self.feature_location = self.tmp_all_feature[:,:,self.classes_size:]

        print('##   feature_class shape : ' + str(self.feature_class.get_shape().as_list()))
        print('##   feature_location shape : ' + str(self.feature_location.get_shape().as_list()))
        # 生成所有default boxs
        self.all_default_boxs = self.generate_all_default_boxs()
        self.all_default_boxs_len = len(self.all_default_boxs)
        print('##   all default boxs : ' + str(self.all_default_boxs_len))
        # 用于Testing返回值
        self.pred_set = [self.feature_class, self.feature_location]

        # 输入真实数据
        self.input_actual_data = []
        self.groundtruth_class = tf.placeholder(shape=[None,self.all_default_boxs_len], dtype=tf.int32,name='groundtruth_class')
        self.groundtruth_location = tf.placeholder(shape=[None,self.all_default_boxs_len,4], dtype=tf.float32,name='groundtruth_location')
        self.groundtruth_positives = tf.placeholder(shape=[None,self.all_default_boxs_len], dtype=tf.float32,name='groundtruth_positives')
        self.groundtruth_negatives = tf.placeholder(shape=[None,self.all_default_boxs_len], dtype=tf.float32,name='groundtruth_negatives')

        self.groundtruth_count = tf.add(self.groundtruth_positives , self.groundtruth_negatives)
        self.loss_location = tf.reduce_sum(tf.reduce_sum(self.smooth_L1(tf.subtract(self.groundtruth_location , self.feature_location)), reduction_indices=2) * self.groundtruth_positives, reduction_indices=1) / ((tf.reduce_sum(self.groundtruth_positives, reduction_indices = 1))+self.extra_decimal)
        self.loss_class = tf.reduce_sum((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.feature_class, labels=self.groundtruth_class) * self.groundtruth_count), reduction_indices=1) / (tf.reduce_sum(self.groundtruth_count, reduction_indices = 1)+self.extra_decimal)
        self.loss_all = tf.reduce_sum(tf.add(self.loss_class , self.loss_location))
        self.loss = [self.loss_all,self.loss_location,self.loss_class]
        # loss优化函数
        self.optimizer = tf.train.AdamOptimizer(0.001)
        #self.optimizer = tf.train.GradientDescentOptimizer(0.001)
        self.train = self.optimizer.minimize(self.loss_all)
         
    # 图像检测与训练
    # input_images : 输入图像数据，格式:[None,width,hight,channel]
    # actual_data : 标注数据，格式:[None,[None,top_X,top_Y,width,hight,classes]] , classes值范围[0,classes_size)
    def run(self, input_images, actual_data):
        # 训练部分
        if self.isTraining :
            if actual_data is None :
                raise Exception('actual_data参数不存在!')
            if len(input_images) != len(actual_data):
                raise Exception('input_images 与 actual_data参数长度不对应!')
        
            f_class, f_location = self.sess.run(self.pred_set, feed_dict={self.input : input_images})
            #print('f_class :【'+str(np.sum(f_class))+'|'+str(np.amax(f_class))+'|'+str(np.amin(f_class))+'】|f_location : 【'+str(np.sum(f_location))+'|'+str(np.amax(f_location))+'|'+str(np.amin(f_location))+'】')           
            self.input_actual_data = actual_data
            gt_class,gt_location,gt_positives,gt_negatives = self.generate_groundtruth_data(f_class)
            #print('gt_class :【'+str(np.sum(gt_class))+'|'+str(np.amax(gt_class))+'|'+str(np.amin(gt_class))+'】|gt_location : 【'+str(np.sum(gt_location))+'|'+str(np.amax(gt_location))+'|'+str(np.amin(gt_location))+'】')            
            #print('gt_positives :【'+str(np.sum(gt_positives))+'|'+str(np.amax(gt_positives))+'|'+str(np.amin(gt_positives))+'】|gt_negatives : 【'+str(np.sum(gt_negatives))+'|'+str(np.amax(gt_negatives))+'|'+str(np.amin(gt_negatives))+'】')            

            self.sess.run(self.train, feed_dict={
                self.input : input_images,
                self.groundtruth_class : gt_class,
                self.groundtruth_location : gt_location,
                self.groundtruth_positives : gt_positives,
                self.groundtruth_negatives : gt_negatives
            })

            loss_all,loss_location,loss_class = self.sess.run(self.loss, feed_dict={
                self.input : input_images,
                self.groundtruth_class : gt_class,
                self.groundtruth_location : gt_location,
                self.groundtruth_positives : gt_positives,
                self.groundtruth_negatives : gt_negatives
            })

            # 释放资源
            self.feature_class = None
            self.feature_location = None
            self.feature_maps = None
            self.tmp_all_feature = None

            return loss_all, loss_class, loss_location, f_class, f_location

        # 检测部分
        else :
            # 预测结果
            pred_class,pred_location = self.sess.run(self.pred_set, feed_dict={self.input : input_images})
            return pred_class , pred_location
            '''
            # 过滤冗余的预测结果
            top_size = int(self.all_default_boxs_len / 20)
            possibilities = []
            for c in pred_class[0] :
                possibilities.append(np.amax(np.exp(c)) / (np.sum(np.exp(c))+self.extra_decimal))
            indicies = np.argpartition(possibilities,-top_size)[-top_size:]
            indicies = indicies[0.1 < np.asarray(possibilities)[indicies]]
                
            pred_class = pred_class[0][indicies]
            pred_location = pred_location[0][indicies]

            result_location = []
            result_class = []
            
            for p_c, p_l in zip(pred_class, pred_location):
                p_class = np.argmax(p_c)
                if p_class == self.background_classes_val:
                    continue
                isFilter = False
                # 这里需要优化，过滤后的box可能不是最完整的!
                for r_class, r_loc in zip(result_class, result_location):
                    if r_class == p_class and self.jaccard(r_loc, p_l) > self.jaccard_value :
                        isFilter = True
                        break
                if isFilter == False:
                    result_location.append(p_l)
                    result_class.append(p_class)        
            
            return result_class , result_location
            '''

    # Batch Normalization算法
    #批量归一标准化操作，预防梯度弥散、消失与爆炸，同时替换dropout预防过拟合的操作
    def batch_normalization(self,input):
        bn_input_shape = input.get_shape()
        bn_input_params = bn_input_shape.as_list()[-1]
        bn_batch_mean, bn_batch_var = tf.nn.moments(input, list(range(len(bn_input_shape) - 1)))
        bn_ema = tf.train.ExponentialMovingAverage(decay=self.conv_bn_decay)
        bn_offset = tf.Variable(tf.zeros([bn_input_params]))
        bn_scale = tf.Variable(tf.ones([bn_input_params]))

        def mean_var_with_update():
            ema_apply_op = bn_ema.apply([bn_batch_mean, bn_batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(bn_batch_mean), tf.identity(bn_batch_var)

        bn_mean, bn_variance = tf.cond(tf.constant(self.isTraining), mean_var_with_update,lambda: (bn_ema.average(bn_batch_mean), bn_ema.average(bn_batch_var)))
        return tf.nn.batch_normalization(input, bn_mean, bn_variance, bn_offset, bn_scale, self.conv_bn_epsilon)

    # smooth_L1 算法
    def smooth_L1(self, x):
        return tf.where(tf.less_equal(tf.abs(x),1.0), tf.multiply(0.5, tf.pow(x, 2.0)), tf.subtract(tf.abs(x), 0.5))

    # 初始化、整理训练数据
    def generate_all_default_boxs(self):
        # 全部按照比例计算并生成一张图像产生的每个default box的坐标以及长宽
        # 用于后续的jaccard匹配
        all_default_boxes = []
        for index, map_shape in zip(range(len(self.feature_maps_shape)), self.feature_maps_shape):
            width = map_shape[1]
            height = map_shape[2]
            scale = self.default_box_scale[index]
            ratios = self.box_aspect_ratio[index]
            for x in range(width):
                for y in range(height):
                    for i,ratio in zip(range(len(ratios)), ratios):
                        top_x = x / float(width)
                        top_y = y / float(height)
						# 原论文的width、height计算公式错误，应为以下公式
                        box_width = np.sqrt(scale * ratio)
                        box_height = np.sqrt(scale / ratio)
                        all_default_boxes.append([top_x, top_y, box_width, box_height])
        
        return all_default_boxes

    # 整理生成groundtruth数据
    def generate_groundtruth_data(self,f_class):
        input_actual_data_len = len(self.input_actual_data)
        
        # 生成空数组，用于保存groundtruth
        gt_class = np.zeros((input_actual_data_len, self.all_default_boxs_len))
        gt_location = np.zeros((input_actual_data_len, self.all_default_boxs_len, 4))
        gt_positives = np.zeros((input_actual_data_len, self.all_default_boxs_len))
        gt_negatives = np.zeros((input_actual_data_len, self.all_default_boxs_len))

        def extract_highest_indicies(pre_class, max_length):
            loss_confs = []
            for c in pre_class:
                # softmax归一化，减去max预防溢出
                max_v = np.amax(c)
                pred = np.exp(c - max_v) / (np.sum(np.exp(c - max_v))/self.extra_decimal)
                loss_confs.append(np.amax(pred))
                
            max_length = min(len(loss_confs),max_length)
            return np.argpartition(loss_confs, -max_length)[-max_length:] 
        
        for img_index in range(input_actual_data_len):
            # 初始化正例训练数据
            positives_count = 0
            for pre_actual in self.input_actual_data[img_index]:
                gt_class_val = np.amax(pre_actual[4:])
                gt_box_val = pre_actual[:4]
                for boxe_index in range(self.all_default_boxs_len):
                    jacc = self.jaccard(gt_box_val, self.all_default_boxs[boxe_index])
                    if jacc > self.jaccard_value or jacc == self.jaccard_value:
                        gt_class[img_index][boxe_index] = gt_class_val
                        gt_location[img_index][boxe_index] = gt_box_val
                        gt_positives[img_index][boxe_index] = 1
                        gt_negatives[img_index][boxe_index] = 0
                        positives_count += 1

            # 从分类预测的所有boxs中，获取分类置信度最高的前 (符合匹配的3倍数量) 位的int位置索引数组
            # positives_count+1 预防positives_count=0，jaccard有可能没有匹配任何defualt box
            indicies = extract_highest_indicies(f_class[img_index], ((positives_count + 1) * 3))
            # 初始化负例训练数据
            for max_box_index in indicies:
                if np.sum(gt_location[img_index][max_box_index]) == 0 : 
                    gt_class[img_index][max_box_index] = self.background_classes_val
                    gt_location[img_index][max_box_index] = [0, 0, 0, 0]
                    gt_positives[img_index][max_box_index] = 0
                    gt_negatives[img_index][max_box_index] = 1

        return gt_class, gt_location, gt_positives, gt_negatives
                     
    def jaccard(self, rect1, rect2):
        rect1_ = [x if x >= 0 else 0 for x in rect1]
        rect2_ = [x if x >= 0 else 0 for x in rect2]
        s = rect1_[2] * rect1_[3] + rect2_[2] * rect2_[3]
        # rect1 and rect2 => A∧B
        intersect = 0
        top_x = max(rect1_[0], rect2_[0])
        top_y = max(rect1_[1], rect2_[1])
        bottom_x = min(rect1_[0] + rect1_[2], rect2_[0] + rect2_[2])
        bottom_y = min(rect1_[1] + rect1_[3], rect2_[1] + rect2_[3])
        if bottom_y > top_y and bottom_x > top_x:
            intersect = (bottom_y - top_y) * (bottom_x - top_x)
        # rect1 or rect2 => A∨B
        union = s - intersect
        # A∧B / A∨B
        return intersect / union
    
