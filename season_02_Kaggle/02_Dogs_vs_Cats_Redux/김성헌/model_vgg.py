import tensorflow as tf
import cv2
import numpy as np

class Model_Vgg19:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name

    def build_net(self, image_size):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.training = tf.placeholder(tf.bool)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
            self.Y = tf.placeholder(tf.float32, [None, 2])
            self.learning_rate = tf.placeholder(tf.float32)

            # Convolutional Layer #1
            conv1_1 = tf.layers.conv2d(inputs=self.X, filters=32, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=32, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[2, 2], padding="SAME", strides=2)

            conv2_1 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2, 2], padding="SAME", strides=2)

            conv3_1 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=128, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            #conv3_3 = tf.layers.conv2d(inputs=conv3_2, filters=128, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            #conv3_4 = tf.layers.conv2d(inputs=conv3_3, filters=128, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3_2, pool_size=[2, 2], padding="SAME", strides=2)

            #conv4_1 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            #conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=512, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            #conv4_3 = tf.layers.conv2d(inputs=conv4_2, filters=512, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            #conv4_4 = tf.layers.conv2d(inputs=conv4_3, filters=512, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            #pool4 = tf.layers.max_pooling2d(inputs=conv4_3, pool_size=[2, 2], padding="SAME", strides=2)

            #conv5_1 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            #conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=512, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            #conv5_3 = tf.layers.conv2d(inputs=conv5_2, filters=512, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            #conv5_4 = tf.layers.conv2d(inputs=conv5_3, filters=512, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            #pool5 = tf.layers.max_pooling2d(inputs=conv5_3, pool_size=[2, 2], padding="SAME", strides=2)

            initializer = tf.contrib.layers.xavier_initializer()

            # 150 -> 75 -> 38 -> 19 -> 10 -> 5
            # Dense Layer with Relu
            flat6 = tf.layers.flatten(pool3) #tf.reshape(pool3, [-1, 256 * 19 * 19])
            #fc6 = tf.layers.dense(inputs=flat6, units=6400, activation=tf.nn.relu, kernel_initializer=initializer)
            fc6 = tf.layers.dense(inputs=flat6, units=1000, activation=tf.nn.relu, kernel_initializer=initializer)
            dropout6 = tf.layers.dropout(inputs=fc6, rate=0.5, training=self.training)

            #flat7 = tf.reshape(dropout6, [-1, 1000])
            #fc7 = tf.layers.dense(inputs=flat7, units=3200, activation=tf.nn.relu, kernel_initializer=initializer)
            fc7 = tf.layers.dense(inputs=dropout6, units=500, activation=tf.nn.relu, kernel_initializer=initializer)
            dropout7 = tf.layers.dropout(inputs=fc7, rate=0.5, training=self.training)

            # Logits (no activation) Layer: L7 Final FC 625 inputs -> 2 outputs
            self.logits = tf.layers.dense(inputs=dropout7, units=2)

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        self.softmax_out = tf.nn.softmax(self.logits)
        self.prediction = tf.argmax(tf.nn.softmax(self.logits), axis=1)
        correct_prediction = tf.equal(self.prediction, tf.argmax(self.Y, axis=1))
        self.correct_count = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # tensorboard data
        tf.summary.scalar("cost", self.cost)
        tf.summary.scalar("accuracy", self.accuracy)
        self.summary = tf.summary.merge_all()

    def probability(self, x_test):
        return self.sess.run(self.softmax_out, feed_dict={self.X: x_test, self.training: False})

    def predict(self, x_test):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: False})

    def countCorrect(self, x_test, y_test):
        return self.sess.run(self.correct_count, feed_dict={self.X: x_test, self.Y: y_test, self.training: False})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: False})

    def get_cost(self, x_test, y_test):
        return self.sess.run(self.cost, feed_dict={self.X: x_test, self.Y: y_test, self.training: False})

    def _summary(self, x_test, y_test):
        return self.sess.run(self.summary, feed_dict={self.X: x_test, self.Y: y_test, self.training: False})

    def train(self, x_data, y_data, learning_rate, training=True):
        return self.sess.run([self.summary, self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.learning_rate: learning_rate, self.training: training})

'''
class DataSet:
    def __init__(self, dataset, batch_size):
        self._dataset = dataset
        self._batch_size = batch_size



   # def getBatch(self, batch_idx):

    def getBatchRange (data_size, batch_size, batch_idx):
        begin_idx = batch_idx * batch_size
        end_idx = begin_idx + batch_size
        end_idx = min(end_idx, data_size)
        return begin_idx, end_idx

    def getImageData (image_file_list, image_width = 150, image_height = 150, channel = 3):
        data = np.ndarray((len(image_file_list), image_width, image_height, channel), dtype=np.uint8)
        for i, image_file in enumerate(image_file_list):
            data[i] = cv2.imread(image_file)
        return data

    def getBatchData (x_data, y_data, batch_size, batch_idx):
        begin_idx, end_idx = getBatchRange(len(x_data), batch_size, batch_idx)

        batch_x = x_data[begin_idx : end_idx]
        batch_y = y_data[begin_idx : end_idx]
        batch_y = np.reshape(batch_y, [-1,2])

        # 리사이즈 이미지 로드
        batch_x_image = getImageData(batch_x)

        return batch_x_image, batch_y

    def getBatchData_x (x_data, batch_size, batch_idx):
        begin_idx, end_idx = getBatchRange(len(x_data), batch_size, batch_idx)

        batch_x = x_data[begin_idx : end_idx]

        # 리사이즈 이미지 로드
        batch_x_image = getImageData(batch_x)

        return batch_x, batch_x_image
        '''