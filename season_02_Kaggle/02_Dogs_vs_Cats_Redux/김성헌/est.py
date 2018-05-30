import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import display, Image, HTML
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import time

import cv2

# Use a custom OpenCV function to read the image, instead of the standard
# TensorFlow `tf.read_file()` operation.
def _read_py_function(filename, label):
  image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_GRAYSCALE)
  return image_decoded, label

# Use standard TensorFlow operations to resize the image to a fixed shape.
def _resize_function(image_decoded, label):
  image_decoded.set_shape([None, None, None])
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

def _parse_function (filename, label):
    #print('filename: ', filename)
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    #image_resized = tf.image.resize_images(image_decoded, [150, 150])
    image_decoded.set_shape([150, 150, 3])
    return image_decoded, label


def input_fn(filenames, labels, batch_size = 16):
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    #dataset = dataset.map(_parse_function)
    dataset = dataset.map(lambda filename, label: tuple(tf.py_func(
        _read_py_function, [filename, label], [tf.uint8, label.dtype])))

    return dataset


def model_fn(features, labels, mode):
    training = True

    if mode == tf.estimator.ModeKeys.TRAIN:
        training = True
    elif mode == tf.estimator.ModeKeys.EVAL:
        training = False
    elif mode == tf.estimator.ModeKeys.PREDICT:
        training = False

    train_op = None
    loss = None
    eval_metric_ops = None

    conv1_1 = tf.layers.conv2d(inputs=features, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1_1, pool_size=[2, 2], padding="SAME", strides=2)

    conv2_1 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2_1, pool_size=[2, 2], padding="SAME", strides=2)

    conv3_1 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3_1, pool_size=[2, 2], padding="SAME", strides=2)

    initializer = tf.contrib.layers.xavier_initializer()

    # 150 -> 75 -> 38 -> 19 -> 10 -> 5
    # Dense Layer with Relu
    flat6 = tf.layers.flatten(pool3) # tf.reshape(pool3, [-1, 256 * 19 * 19])
    fc6 = tf.layers.dense(inputs=flat6, units=1000, activation=tf.nn.relu, kernel_initializer=initializer)
    dropout6 = tf.layers.dropout(inputs=fc6, rate=0.5, training=training)

    # flat7 = tf.reshape(dropout6, [-1, 1000])
    fc7 = tf.layers.dense(inputs=dropout6, units=500, activation=tf.nn.relu, kernel_initializer=initializer)
    dropout7 = tf.layers.dropout(inputs=fc7, rate=0.5, training=training)

    logits = tf.layers.dense(inputs=dropout7, units=2)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={
                    "prob":tf.nn.softmax(logits)})
    else:
        global_step = tf.train.get_global_step()
        loss = tf.losses.softmax_cross_entropy(labels, logits)
        train_op = tf.train.AdamOptimizer(0.0001).minimize(loss, global_step)

        accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1), tf.argmax(tf.nn.softmax(logits), axis=1))
        eval_metric_ops = {"acc": accuracy}
        return tf.estimator.EstimatorSpec(
                mode=mode,
                train_op=train_op,
                loss=loss,
                eval_metric_ops=eval_metric_ops)


if __name__ == "__main__":
    est = tf.estimator.Estimator(model_fn)

    DATA_DIR = "../data/"
    TRAIN_DIR = DATA_DIR + "train_resize/"

    train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
    train_dogs = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
    train_cats = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

    print("train_dogs: ", len(train_dogs))
    print("train_cats: ", len(train_cats))

    train_dogs = train_dogs[:100]
    train_cats = train_cats[:100]

    train_dog_cat = train_dogs + train_cats

    label_dog = [0 for i in range(len(train_dogs))]
    label_cat = [1 for i in range(len(train_cats))]
    label = label_dog + label_cat
    label_one_hot = np.eye(2)[label]

    x_train, x_valid, y_train, y_valid = train_test_split (
        train_dog_cat, label_one_hot, test_size=0.3, random_state=42)

    print("train: ", len(x_train))
    print("valid: ", len(x_valid))


    for epoch in range(10):
        est.train(input_fn(x_train, y_train))
        est.evaluate(input_fn(x_train, y_train))

    '''
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array([list(range(10))], np.float32)},
            num_epochs=1,
            shuffle=False)

    predictions = est.predict(pred_input_fn)
    for i in predictions:
        print(i["prob"])
        '''