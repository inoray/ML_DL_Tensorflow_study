#By @Kevin Xu
#kevin28520@gmail.com
#Youtube: https://www.youtube.com/channel/UCVCSn4qQXTDAtGWpWAe4Plw
#
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

import os
import numpy as np
import tensorflow as tf
import input_data
import model

#%%

N_CLASSES = 2
IMG_W = 208  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 208
RATIO = 0.2
BATCH_SIZE = 64
CAPACITY = 2000
MAX_STEP = 6000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001 #0.0001 # with current parameters, it is suggested to use learning rate<0.0001


#%%
def run_training():
    
    # you need to change the directories to yours.
    train_dir = 'C:/tf/train/'
    logs_train_dir = 'C:/tf/logs/train/'
    logs_val_dir = 'C:/tf/logs/val/'
    #train_dir = '/home/kevin/tensorflow/cats_vs_dogs/data/train/'
    #logs_train_dir = '/home/kevin/tensorflow/cats_vs_dogs/logs/train/'
    
    train, train_label, val, val_label = input_data.get_files(train_dir, RATIO)
    
    train_batch, train_label_batch = input_data.get_batch(train,
                                                          train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE, 
                                                          CAPACITY)      
    
    val_batch, val_label_batch = input_data.get_batch(val,
                                                  val_label,
                                                  IMG_W,
                                                  IMG_H,
                                                  BATCH_SIZE, 
                                                  CAPACITY)
     
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    keep_prob = tf.placeholder(tf.float32)
     
    logits = model.inference(x, BATCH_SIZE, N_CLASSES, keep_prob)
    loss = model.losses(logits, y_)
    acc = model.evaluation(logits, y_)
    train_op = model.trainning(loss, learning_rate)
    
    with tf.Session() as sess:
        
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)
    
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                        break
                
                tra_images, tra_labels = sess.run([train_batch, train_label_batch])    
                _, tra_loss, tra_acc = sess.run([train_op, loss, acc], feed_dict={x:tra_images, y_:tra_labels,keep_prob: 0.7})
                
                if step>0:   
                    if step % 50 == 0:
                        print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
                        summary_str = sess.run(summary_op, feed_dict={x: tra_images, y_: tra_labels,keep_prob: 0.7})
                        train_writer.add_summary(summary_str, step)
                    
                    if step % 200 == 0 or (step + 1) == MAX_STEP:
                            val_images, val_labels = sess.run([val_batch, val_label_batch])
                            val_loss, val_acc = sess.run([loss, acc], feed_dict={x:val_images, y_:val_labels,keep_prob: 1})
                            print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc*100.0))
                            summary_str = sess.run(summary_op, feed_dict={x: val_images, y_: val_labels,keep_prob: 1})
                            val_writer.add_summary(summary_str, step)  
                            
                    if step % 2000 == 0 or (step + 1) == MAX_STEP:
                        checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)
                        print('Save checkpoint  : %s' %(checkpoint_path))
                    
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)
        #sess.close()

run_training()    

#%% Evaluate one image
# when training, comment the following codes.

from PIL import Image
import matplotlib.pyplot as plt

def get_one_image(train, train_label):
    '''Randomly pick one image from training data
    Return: ndarray
    '''
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]
    
    print("img_dir:" + img_dir)
    print("train_label:" + str(train_label[ind]))
    
    image = Image.open(img_dir)
    plt.imshow(image)
    image = image.resize([208, 208])
    image = np.array(image)
    return image

def evaluate_one_image():
    '''Test one image against the saved models and parameters
    '''
    
    # you need to change the directories to yours.
    train_dir = 'C:/tf/train/'
    train, train_label, val, val_label = input_data.get_files(train_dir, 0.1)
    #train, train_label = input_data.get_files(train_dir)
    image_array = get_one_image(train, train_label)
    
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2
        
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 208, 208, 3])
        
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
        
        logit = tf.nn.softmax(logit)
        
        x = tf.placeholder(tf.float32, shape=[208, 208, 3])
        
        
        # you need to change the directories to yours.
        #logs_train_dir = '/home/kevin/tensorflow/cats_vs_dogs/logs/train/' 
        logs_train_dir = 'C:/tf/logs/train/'
                
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            
            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            if max_index==0:
                print('This is a cat with possibility %.6f' %prediction[:, 0])
            else:
                print('This is a dog with possibility %.6f' %prediction[:, 1])


#%%

def evaluate_all_image():

    test_dir = 'C:/tf/test/'
    
    logs_train_dir = 'C:/tf/logs/train/'
    
    image_batch, filename_batch, test_image_num = input_data.get_test_files(IMG_W, IMG_H, 0)
    
    x = tf.placeholder(tf.float32, shape=[1, IMG_W, IMG_H, 3])
    
    logits = model.inference(x, 1, N_CLASSES)
    logits = tf.nn.softmax(logits)
    #loss = model.losses(logits, y_)
    #acc = model.evaluation(logits, y_)
    
    with tf.Session() as sess:
        
        saver = tf.train.Saver()
        
        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')
    
        #prediction = sess.run(logit, feed_dict={x: image_array})
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
    
        try:
            while not coord.should_stop():
        
                image_array, filename = sess.run([image_batch, filename_batch])
                prediction = sess.run([logits], feed_dict={x: image_array})
                max_index = np.argmax(prediction)
                
                filename = str(filename)
                names = filename.split(sep='/')[-1] # ['dog', '973', 'jpg']
                id = names.split(sep='.')[0]
                
                if max_index==0:
                    print('%s, 0, cat, %.6f' % (id, prediction[0][0, 0]))
                else:
                    print('%s, 1, dog, %.6f' % (id, prediction[0][0, 1]))
                        
        except tf.errors.OutOfRangeError:
            print('error!')
        finally:
            coord.request_stop()
            print('done!')
        coord.join(threads)

#evaluate_all_image()
    
#evaluate_one_image()
