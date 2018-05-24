#By @Kevin Xu
#kevin28520@gmail.com

# 11.08 2017 更新
# 最近入驻了网易云课堂讲师，我的第一门课《使用亚马逊云计算训练深度学习模型》。
# 有兴趣的同学可以学习交流。
# * 我的网易云课堂主页： http://study.163.com/provider/400000000275062/index.htm

# 深度学习QQ群, 1群满): 153032765
# 2群：462661267
#The aim of this project is to use TensorFlow to process our own data.
#    - input_data.py:  read in data and generate batches
#    - model: build the model architecture
#    - training: train

# I used Ubuntu with Python 3.5, TensorFlow 1.0*, other OS should also be good.
# With current settings, 10000 traing steps needed 50 minutes on my laptop.


# data: cats vs. dogs from Kaggle
# Download link: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
# data size: ~540M

# How to run?
# 1. run the training.py once
# 2. call the run_training() in the console to train the model.

# Note: 
# it is suggested to restart your kenel to train the model multiple times 
#(in order to clear all the variables in the memory)
# Otherwise errors may occur: conv1/weights/biases already exist......


#%%

import tensorflow as tf
import numpy as np
import os
import math

#%%

# you need to change this to your data directory  'C:/tf/test/1.jpg'
train_dir = 'C:/tf/train/'

def get_files(file_dir, ratio):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.') # ['dog', '973', 'jpg']
        if name[0]=='cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print('There are %d cats\nThere are %d dogs' %(len(cats), len(dogs)))
    
    image_list = np.hstack((cats, dogs)) # (25000,)
    label_list = np.hstack((label_cats, label_dogs))  # (25000,)
    
    #shuffle을 여기서 하는 이유는 validation가 모두 dogs이 되는 문제를 없애기 위해
    temp = np.array([image_list, label_list]) # 2*25000  여기서 int ->string으로 강제 형변환 발생
    temp = temp.transpose() # 25000*2
    np.random.shuffle(temp)
    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])
    #all_image_list = image_list
    #all_label_list = label_list
    
    n_sample = len(all_label_list)
    n_val = math.ceil(n_sample*ratio) # number of validation samples
    n_train = n_sample - n_val # number of training samples
    print('There are %d train datas\nThere are %d validation datas' %(n_train, n_val))
    
    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]
    
    return tra_images,tra_labels,val_images,val_labels

#cats = np.array([1,2,3])
#print("image_list:" + str(cats.shape))
#tra_images, tra_labels, val_images, val_labels = get_files(train_dir, 0.2)
#%%

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    
    label = input_queue[1] # label
    
    image_contents = tf.read_file(input_queue[0])  # image
    image = tf.image.decode_jpeg(image_contents, channels=3)

    ######################################
    # data argumentation should go to here
    ######################################
    
    #image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H) #reserve aspect ratio
    image = tf.image.resize_images(image, [image_W, image_H])
    
    # if you want to test the generated batches of images, you might want to comment the following line.
    image = tf.image.per_image_standardization(image)
    
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity)
        
    #you can also use shuffle_batch 
#   image_batch, label_batch = tf.train.shuffle_batch([image,label],
#                                                      batch_size=BATCH_SIZE,
#                                                      num_threads=64,
#                                                      capacity=CAPACITY,
#                                                      min_after_dequeue=CAPACITY-1)
    
    label_batch = tf.reshape(label_batch, [batch_size])
    #image_batch = tf.cast(image_batch, tf.int32)
    #image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch


test_dir = 'C:/tf/test/'
def get_test_files(image_W, image_H, maxCount=0, batch_size=1, capacity=256):
    '''
    Returns:
        list of images
    '''
    datas = []

    index=0
    for file in os.listdir(test_dir):
        #name = file.split(sep='.') # ['dog', '973', 'jpg']
        datas.append(test_dir + file)
        index+=1
        if maxCount!=0 and index >= maxCount:
            break
    print('There are %d datas' %(len(datas)))
    
    input_queue = tf.train.slice_input_producer([datas])
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    
    #image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H) #reserve aspect ratio
    image = tf.image.resize_images(image, [image_W, image_H])
    image = tf.image.per_image_standardization(image)
    image_batch, filename_batch = tf.train.batch([image, input_queue[0]], batch_size= 1, num_threads= 64, capacity = capacity)
    return image_batch, filename_batch, index



 
#%% TEST
# To test the generated batches of images
# When training the model, DO comment the following codes


import matplotlib.pyplot as plt

BATCH_SIZE = 2
CAPACITY = 256
IMG_W = 208
IMG_H = 208
ratio=0.2
'''
# train file 확인
tra_images, tra_labels, val_images, val_labels = get_files(train_dir, ratio)
tra_image_batch, tra_label_batch = get_batch(tra_images, tra_labels, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    try:
        while not coord.should_stop() and i<1:
            
            # imageSize, width, height, rgb
            img, label = sess.run([tra_image_batch, tra_label_batch])
            
            # just test one batch
            for j in np.arange(BATCH_SIZE):
                print('label: %d' %label[j])
                #plt.imshow(img[j,:,:,:])
                #plt.show()
            i+=1
            
    except tf.errors.OutOfRangeError:
        print('error!')
    finally:
        coord.request_stop()
        print('done!')
    coord.join(threads)
'''


# test file 확인
test_image, filename_batch, test_image_num = get_test_files(IMG_W, IMG_H, 0)
print("Test File Count: %d" %test_image_num)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    try:
        while not coord.should_stop():
            
            # imageSize, width, height, rgb
            img, filename = sess.run([test_image, filename_batch])
            
            print(filename)   
            plt.imshow(img[0,:,:,:])
            plt.show()
            
    except tf.errors.OutOfRangeError:
        print('error!')
    finally:
        coord.request_stop()
        print('done!')
    coord.join(threads)


#%%