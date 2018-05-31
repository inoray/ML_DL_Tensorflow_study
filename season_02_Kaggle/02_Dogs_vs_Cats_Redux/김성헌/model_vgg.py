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
        return self.sess.run(self.prediction, feed_dict={self.X: x_test, self.training: False})

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

import random
class ShkimDataSet:
    def __init__(self, dataset, batch_size):
        """데이터 관리

        Args:
            dataset: [image_files, labels]
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._batch_idx = 0
        self._dataset_count = len(dataset)
        self._data_count = len(dataset[0])
        self._image_shape = [150, 150, 3]
        self._resize = False

    def setImageInfo(self, image_shape, resize=False):
        """이미지 정보 설정

        Args:
            image_shape: [width, height, channel]
            resize: True, False. 이미지 축소수행 유무
        """
        self._image_shape = image_shape
        self._resize = resize

    def init_iterator(self):
        self._batch_idx = 0

    def countData(self):
        return self._data_count

    def countBatch(self):
        return int(self.countData() / self._batch_size) + 1

    def getBatchRange (self, data_size, batch_size, batch_idx):
        begin_idx = batch_idx * batch_size
        end_idx = begin_idx + batch_size
        end_idx = min(end_idx, data_size)
        return begin_idx, end_idx

    def getImageData (self, image_file_list, image_shape=[150, 150, 3]):
        data = np.ndarray((len(image_file_list), image_shape[0], image_shape[1], image_shape[2]), dtype=np.uint8)
        for i, image_file in enumerate(image_file_list):
            data[i] = cv2.imread(image_file)
        return data

    def next_batch(self):
        begin_idx, end_idx = self.getBatchRange(self._data_count, self._batch_size, self._batch_idx)
        self._batch_idx += self._batch_idx

        batch_x = self._dataset[0][begin_idx : end_idx]
        batch_x_image = self.getImageData(batch_x, self._image_shape)

        if self._dataset_count == 1:
            return batch_x_image
        elif self._dataset_count == 2:
            batch_y = self._dataset[1][begin_idx : end_idx]
            batch_y = np.reshape(batch_y, [-1,2])
            return batch_x_image, batch_y


def train (model, train_dataset, valid_dataset, learning_rate, epochs):
    sess = model.sess
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    start_time = time.time()
    train_count = train_dataset.countData()
    total_batch_train = train_dataset.countBatch()
    print('train_count: ', train_count, 'total_batch_train: ', total_batch_train)
    valid_count = valid_dataset.countData()
    total_batch_valid = valid_dataset.countBatch()
    print('valid_count: ', valid_count, 'total_batch_valid: ', total_batch_valid)

    # usage
    # tensorboard --logdir=./log
    train_writer = tf.summary.FileWriter("./log/train")
    valid_writer = tf.summary.FileWriter("./log/valid")
    train_writer.add_graph(sess.graph)
    train_global_step = 0
    valid_global_step = 0

    print('Learning started. It takes sometime.')
    for epoch in range(epochs):
        avg_cost_train = 0
        avg_cost_valid = 0
        accuracy_train = 0
        accuracy_valid = 0

        correct_count_train = 0
        correct_count_valid = 0

        start_time_epoch = time.time()

        train_dataset.init_iterator()
        for i in range(total_batch_train):
            batch_x_image, batch_y = train_dataset.next_batch()

            s, c, _ = model.train(batch_x_image, batch_y, learning_rate)
            avg_cost_train += c / total_batch_train
            correct_count_train += model.countCorrect(batch_x_image, batch_y)

            train_global_step += 1
            if train_global_step % 1000:
                train_writer.add_summary(s, global_step=train_global_step)

        accuracy_train = correct_count_train / train_count

        valid_dataset.init_iterator()
        for i in range(total_batch_valid):
            batch_x_image, batch_y = valid_dataset.next_batch ()
            c = model.get_cost(batch_x_image, batch_y)
            avg_cost_valid += c / total_batch_valid

            correct_count_valid += model.countCorrect(batch_x_image, batch_y)
            #s = model.summary(batch_x_image, batch_y)
            #valid_writer.add_summary(s, global_step=valid_global_step)
            #valid_global_step += 1

        accuracy_valid = correct_count_valid / valid_count

        print('Epoch:', '%04d' % (epoch + 1)
            , 'train [cost: ', '{:.9f}'.format(avg_cost_train), ', acc: %.4f]' % accuracy_train
            , 'valid [cost: ', '{:.9f}'.format(avg_cost_valid), ', acc: %.4f]' % accuracy_valid
            , " %.2f seconds" % (time.time() - start_time_epoch))

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            checkpoint_path = os.path.join('./model/', 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=epoch + 1)
            print('Save checkpoint  : %s' %(checkpoint_path))

    print('Learning Finished!')
    print("--- %.2f seconds ---" %(time.time() - start_time))


def eval(model, eval_dataset):
    sess = model.sess
    sess.run(tf.global_variables_initializer())

    start_time = time.time()
    avg_cost = 0
    eval_count = eval_dataset.countData()
    total_batch_eval = eval_dataset.countBatch()
    print('eval_count: ', eval_count, ', total_batch_eval: ', total_batch_eval)

    eval_dataset.init_iterator()
    correct_count = 0
    for i in range(total_batch_eval):
        batch_x_image, batch_y = eval_dataset.next_batch()
        c = model.get_cost(batch_x_image, batch_y)
        avg_cost += c / total_batch_eval
        correct_count += model.countCorrect(batch_x_image, batch_y)

    accuracy = correct_count / eval_count

    print('Evaluation Finished!')
    print("cost: ", "{:.9f}".format(avg_cost), ", accuracy: %.4f" % accuracy)
    print("--- %.2f seconds ---" %(time.time() - start_time))

    return avg_cost, accuracy


def predict(model, test_dataset):
    start_time = time.time()

    test_count = test_dataset.countData()
    total_batch = test_dataset.countBatch()

    test_dataset.init_iterator()
    predict_list = []
    probability_list = []
    for i in range(total_batch):
        batch_x_image = test_dataset.next_batch()
        batch_predict = model.predict(batch_x_image)
        batch_prob = model.probability(batch_x_image)
        predict_list.extend(batch_predict)
        probability_list.extend(batch_prob)

    print("--- %.2f seconds ---" %(time.time() - start_time))
    return predict_list, probability_list


if __name__ == "__main__":

    import os
    from sklearn.model_selection import train_test_split
    import time

    DATA_DIR = "../data/"
    TRAIN_DIR = DATA_DIR + "train_resize/"

    train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
    train_dogs = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
    train_cats = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

    print("train_dogs: ", len(train_dogs))
    print("train_cats: ", len(train_cats))

    train_dogs = train_dogs[:20]
    train_cats = train_cats[:20]

    train_dog_cat = train_dogs + train_cats

    label_dog = [0 for i in range(len(train_dogs))]
    label_cat = [1 for i in range(len(train_cats))]
    label = label_dog + label_cat
    label_one_hot = np.eye(2)[label]

    x_train, x_valid, y_train, y_valid = train_test_split (
        train_dog_cat, label_one_hot, test_size=0.3, random_state=42)

    print("train: ", len(x_train))
    print("valid: ", len(x_valid))

    learning_rate = 0.0001
    epochs = 2
    batch_size = 10

    train_dataset = ShkimDataSet([x_train, y_train], batch_size = batch_size)
    valid_dataset = ShkimDataSet([x_valid, y_valid], batch_size = batch_size)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth =True)))
    model = Model_Vgg19(sess, "model")
    model.build_net(150)

    train (model, train_dataset, valid_dataset, learning_rate, epochs)
    eval (model, valid_dataset)

    import pandas as pd
    import re
    TEST_DIR = DATA_DIR + "test_resize/"
    test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]
    test_images.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

    test_images = test_images[:10]

    test_dataset = ShkimDataSet([test_images], batch_size = batch_size)
    predict_list, probability_list = predict(model, test_dataset)
    hp = np.array(probability_list)
    df = pd.DataFrame({"id": range(1, len(probability_list) + 1), "label": hp[:, 0], "class": predict_list})
    df.to_csv('submission.csv', index=False)
