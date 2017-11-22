import os
import gc
import pickle
import random
import skimage.io
import skimage.transform
import numpy as np
import tensorflow as tf
import ssd300
import time

'''
SSD检测
'''
def testing():
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        ssd_model = ssd300.SSD300(sess,False)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        if os.path.exists('./session_params/session.ckpt.index') :
            saver.restore(sess, './session_params/session.ckpt')
            image, actual = get_traindata_voc2007(1)
            for img, act in zip(image, actual):
                f_class,f_location = ssd_model.run([img],None)
                print('pic end time : '+time.asctime(time.localtime(time.time())))
                
                for a in act :
                    print('actual:'+str(a))
                    for c,l in zip(f_class[0],f_location[0]):
                        if(np.argmax(c)==a[4]):
                            print('f_class:'+str(a[4]))
                            print('f_location:'+str(l))
                            print('----------------------')
        else:
            print('No Data Exists!')
            
        sess.close()
    
'''
SSD训练
'''
def training():
    batch_size = 15
    running_count = 0
    
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        ssd_model = ssd300.SSD300(sess,True)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        if os.path.exists('./session_params/session.ckpt.index') :
            print('\nStart Restore')
            saver.restore(sess, './session_params/session.ckpt')
            print('\nEnd Restore')
         
        print('\nStart Training')
        min_loss_location = 100000.
        min_loss_class = 100000.
        while((min_loss_location+min_loss_class) > 0.05 and running_count < 100000 ):
            running_count += 1
            
            train_data, actual_data = get_traindata_voc2007(batch_size)
            
            if len(train_data) > 0:
                loss_all,loss_location,loss_class = ssd_model.run(train_data, actual_data)
                l = np.sum(loss_location)
                c = np.sum(loss_class)
                if min_loss_location>l:
                    min_loss_location = l
                if min_loss_class>c:
                    min_loss_class = c
                print('Running:【' + str(running_count) + '】 | Loss All:【' + str(min_loss_location+min_loss_class) +'/'+ str(loss_all) + '】 | Location:【' + str(min_loss_location) +'/'+ str(np.sum(loss_location)) + '】 | Class:【' + str(min_loss_class) +'/'+ str(np.sum(loss_class)) + '】')
                #print('loss_location:'+str(loss_location))
                #print('loss_class:'+str(loss_location))
                
                # 每训练100次保存ckpt
                if running_count%100 == 0:
                    saver.save(sess, './session_params/session.ckpt')
                    gc.collect()
            else:
                print('No Data Exists!')
                break
            
        saver.save(sess, './session_params/session.ckpt')
        sess.close()
        gc.collect()
            
    print('End Training')
    
'''
获取voc2007训练图片数据
train_data：训练批次图像，格式[None,width,height,3]
actual_data：图像标注数据，格式[None,[None,top_x,top_y,width,height,classes]]
'''
def get_traindata_voc2007(batch_size):
    train_data = []
    actual_data = []
    train_img = None
    train_img_key = None
    with open('./train_datasets/voc2007.pkl', 'rb') as train_file:
        train_img = pickle.load(train_file)
        train_img_key = ([] if train_img is None else train_img.keys())
        if len(train_img_key) > batch_size :
            train_img_key = random.sample(train_img_key, batch_size)
    
    if len(train_img_key) > 0:
        for img_file_name in train_img_key:
            img = skimage.io.imread('./train_datasets/voc2007/' + img_file_name)
            img = skimage.transform.resize(img, (300, 300))
            img = img / 255
            img = np.array(img, dtype=np.float32)
            img = img.reshape((300, 300, 3))
            train_data.append(img)
            actual = train_img[img_file_name]
            actual_item = []
            for obj in actual:
                label = np.argmax(obj[4:])
                loc = obj[:4]
                actual_item.append([loc[0], loc[1], loc[2] - loc[0], loc[3] - loc[1], label])
            actual_data.append(actual_item)

    return train_data, actual_data
    
'''
主程序入口
'''
if __name__ == '__main__':
    print('\nStart Running')
    # 检测
    #testing()
    # 训练
    training()
    print('\nEnd Running')


