import tensorflow as tf

class Vgg:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name

    def build_net(self, image_shape=[150, 150, 3], class_count=2):
        """모델빌드

        args:
            image_shape: image shape. [height, width, chnnel]
        """
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.training = tf.placeholder(tf.bool)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], image_shape[2]])
            self.Y = tf.placeholder(tf.float32, [None, class_count])
            self.learning_rate = tf.placeholder(tf.float32)

            '''
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
            flat6 = tf.layers.flatten(net) #tf.reshape(pool3, [-1, 256 * 19 * 19])
            #fc6 = tf.layers.dense(inputs=flat6, units=6400, activation=tf.nn.relu, kernel_initializer=initializer)
            fc6 = tf.layers.dense(inputs=flat6, units=1000, activation=tf.nn.relu, kernel_initializer=initializer)
            dropout6 = tf.layers.dropout(inputs=fc6, rate=0.5, training=self.training)

            #flat7 = tf.reshape(dropout6, [-1, 1000])
            #fc7 = tf.layers.dense(inputs=flat7, units=3200, activation=tf.nn.relu, kernel_initializer=initializer)
            fc7 = tf.layers.dense(inputs=dropout6, units=500, activation=tf.nn.relu, kernel_initializer=initializer)
            dropout7 = tf.layers.dropout(inputs=fc7, rate=0.5, training=self.training)

            # Logits (no activation) Layer: L7 Final FC 625 inputs -> 2 outputs
            self.logits = tf.layers.dense(inputs=dropout7, units=class_count)
            '''

            dropout_rate = 0.5
            seed = 777
            net = self.X
            n_filters = 64
            for i in range(3):
                net = self.conv_bn_activ_dropout(name="3x3conv{}-{}".format(i+1, 1), x=net,
                                            n_filters=n_filters, kernel_size=[3,3], strides=1,
                                            dropout_rate=dropout_rate, training=self.training, seed=seed)
                net = self.conv_bn_activ_dropout(name="3x3conv{}-{}".format(i+1, 2), x=net,
                                            n_filters=n_filters, kernel_size=[3,3], strides=1,
                                            dropout_rate=dropout_rate, training=self.training, seed=seed)
                if i == 2:
                    net = self.conv_bn_activ_dropout(name="1x1conv", x=net,
                                                n_filters=n_filters, kernel_size=[1,1], strides=1,
                                                dropout_rate=dropout_rate, training=self.training, seed=seed)
                n_filters *= 2
                net = self.conv_bn_activ_dropout(name="5x5stridepool{}".format(i+1), x=net,
                                            n_filters=n_filters, kernel_size=[5,5], strides=2,
                                            dropout_rate=dropout_rate, training=self.training, seed=seed)

            initializer = tf.contrib.layers.xavier_initializer()

            net = tf.contrib.layers.flatten(net)
            # Logits (no activation) Layer: L7 Final FC 625 inputs -> 2 outputs
            self.logits = tf.layers.dense(inputs=net, units=class_count, kernel_initializer=initializer)

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y))

        global_step = tf.Variable(0, trainable=False)
        learningRate = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                                global_step= global_step,
                                                decay_steps=5000,
                                                decay_rate= 0.1,
                                                staircase=True)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost, global_step=global_step, name="optimizer")

        self.probability = tf.nn.softmax(self.logits)
        self.prediction = tf.argmax(tf.nn.softmax(self.logits), axis=1)
        correct_prediction = tf.equal(self.prediction, tf.argmax(self.Y, axis=1))
        self.correct_count = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # tensorboard data
        tf.summary.scalar("cost", self.cost)
        tf.summary.scalar("accuracy", self.accuracy)
        self.summary = tf.summary.merge_all()

    def conv_bn_activ_dropout(self, name, x, n_filters, kernel_size, strides, dropout_rate, training, seed,
                            padding='SAME', activ_fn=tf.nn.relu):
        with tf.variable_scope(name):
            net = tf.layers.conv2d(x, n_filters, kernel_size, strides=strides, padding=padding, use_bias=False,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed))
            net = tf.layers.batch_normalization(net, training=training)
            net = activ_fn(net)
            if dropout_rate > 0.0:
                net = tf.layers.dropout(net, rate=dropout_rate, training=training, seed=seed)
        return net

    def predict(self, x_test):
        return self.sess.run([self.prediction, self.probability], feed_dict={self.X: x_test, self.training: False})

    def eval(self, x_test, y_test):
        return self.sess.run([self.accuracy, self.cost], feed_dict={self.X: x_test, self.Y: y_test, self.training: False})

    def train(self, x_data, y_data, learning_rate, training=True):
        return self.sess.run([self.summary, self.accuracy, self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.learning_rate: learning_rate, self.training: training})

    def _summary(self, x_test, y_test):
        return self.sess.run(self.summary, feed_dict={self.X: x_test, self.Y: y_test, self.training: False})


def conv_bn_activ_dropout(x, n_filters, kernel_size, strides, dropout_rate, training, seed,
                          padding='SAME', activ_fn=tf.nn.relu, name="conv_bn_act_dr"):
    #with tf.variable_scope(name):
    net = tf.layers.conv2d(x, n_filters, kernel_size, strides=strides, padding=padding, use_bias=False,
                           kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed))
    net = tf.layers.batch_normalization(net, training=training)
    net = activ_fn(net)
    if dropout_rate > 0.0:
        net = tf.layers.dropout(net, rate=dropout_rate, training=training, seed=seed)
    return net


def conv_bn_activ(x, training, n_filters, kernel_size, strides=1, seed=777,
                  padding='SAME', activ_fn=tf.nn.relu, name="conv_bn_act"):
    return conv_bn_activ_dropout(x, n_filters, kernel_size, strides, 0.0, training, seed,
                                 padding=padding, activ_fn=activ_fn, name=name)


def stem(x, training, name="stem"):
    with tf.variable_scope(name):
        b1 = conv_bn_activ(x, training, 32, [1, 1])
        b1 = conv_bn_activ(b1, training, 48, [3, 3])
        b2 = conv_bn_activ(x, training, 32, [1, 1])
        b2 = conv_bn_activ(x, training, 32, [1, 7])
        b2 = conv_bn_activ(b2, training, 32, [7, 1])
        b2 = conv_bn_activ(b2, training, 48, [3, 3])
        net = tf.concat([b1, b2], axis=-1)
        print(net)
        return net


def inception_A(x, training, name="inception_A"):
    # num of channels : 24 x 4 = 96
    with tf.variable_scope(name):
        b1 = tf.layers.average_pooling2d(x, [3, 3], 1, padding='SAME')
        b1 = conv_bn_activ(b1, training, 24, [1, 1])
        b2 = conv_bn_activ(x, training, 24, [1, 1])
        b3 = conv_bn_activ(x, training, 16, [1, 1])
        b3 = conv_bn_activ(b3, training, 24, [3, 3])
        b4 = conv_bn_activ(x, training, 16, [1, 1])
        b4 = conv_bn_activ(b4, training, 24, [3, 3])
        b4 = conv_bn_activ(b4, training, 24, [3, 3])
        net = tf.concat([b1, b2, b3, b4], axis=-1)
        print(net)
        return net


def inception_B(x, training, name="inception_B"):
    # num of channels : 32 + 96 + 64 + 64 = 256
    with tf.variable_scope(name):
        b1 = tf.layers.average_pooling2d(x, [3, 3], 1, padding='SAME')
        b1 = conv_bn_activ(b1, training, 32, [1, 1])
        b2 = conv_bn_activ(x, training, 96, [1, 1])
        b3 = conv_bn_activ(x, training, 48, [1, 1])
        b3 = conv_bn_activ(b3, training, 56, [1, 7])
        b3 = conv_bn_activ(b3, training, 64, [7, 1])
        b4 = conv_bn_activ(x, training, 48, [1, 1])
        b4 = conv_bn_activ(b4, training, 48, [1, 7])
        b4 = conv_bn_activ(b4, training, 56, [7, 1])
        b4 = conv_bn_activ(b4, training, 56, [1, 7])
        b4 = conv_bn_activ(b4, training, 64, [7, 1])
        net = tf.concat([b1, b2, b3, b4], axis=-1)
        print(net)
        return net


def inception_C(x, training, name="inception_C"):
    # num of channels : 64 * 6 = 384
    with tf.variable_scope(name):
        b1 = tf.layers.average_pooling2d(x, [3, 3], 1, padding='SAME')
        b1 = conv_bn_activ(b1, training, 64, [1, 1])
        b2 = conv_bn_activ(x, training, 64, [1, 1])
        b3 = conv_bn_activ(x, training, 96, [1, 1])
        b3_1 = conv_bn_activ(b3, training, 64, [1, 3])
        b3_2 = conv_bn_activ(b3, training, 64, [3, 1])
        b4 = conv_bn_activ(x, training, 96, [1, 1])
        b4 = conv_bn_activ(b4, training, 112, [1, 3])
        b4 = conv_bn_activ(b4, training, 128, [3, 1])
        b4_1 = conv_bn_activ(b4, training, 64, [3, 1])
        b4_2 = conv_bn_activ(b4, training, 64, [1, 3])
        net = tf.concat([b1, b2, b3_1, b3_2, b4_1, b4_2], axis=-1)
        print(net)
        return net

def reduction_A(x, training, name="reduction_A"):
    # num of channels : 96 + 64 + 96 = 256
    with tf.variable_scope(name):
        b1 = tf.layers.max_pooling2d(x, [3, 3], 2, padding='SAME')
        b2 = conv_bn_activ(x, training, 96, [3, 3], strides=2)
        b3 = conv_bn_activ(x, training, 48, [1, 1])
        b3 = conv_bn_activ(b3, training, 56, [3, 3])
        b3 = conv_bn_activ(b3, training, 64, [3, 3], strides=2)
        net = tf.concat([b1, b2, b3], axis=-1)
        print(net)
        return net


def reduction_A(x, training, name="reduction_A"):
    # num of channels : 96 + 64 + 96 = 256
    with tf.variable_scope(name):
        b1 = tf.layers.max_pooling2d(x, [3, 3], 2, padding='SAME')
        b2 = conv_bn_activ(x, training, 96, [3, 3], strides=2)
        b3 = conv_bn_activ(x, training, 48, [1, 1])
        b3 = conv_bn_activ(b3, training, 56, [3, 3])
        b3 = conv_bn_activ(b3, training, 64, [3, 3], strides=2)
        net = tf.concat([b1, b2, b3], axis=-1)
        print(net)
        return net


def reduction_B(x, training, name="reduction_B"):
    # num of channes : 256 + 48 + 80 = 384
    with tf.variable_scope(name):
        b1 = tf.layers.max_pooling2d(x, [3, 3], 2, padding='SAME')
        b2 = conv_bn_activ(x, training, 48, [1, 1])
        b2 = conv_bn_activ(b2, training, 48, [3, 3], strides=2)
        b3 = conv_bn_activ(x, training, 64, [1, 1])
        b3 = conv_bn_activ(b3, training, 64, [1, 7])
        b3 = conv_bn_activ(b3, training, 80, [7, 1])
        b3 = conv_bn_activ(b3, training, 80, [3, 3], strides=2)
        net = tf.concat([b1, b2, b3], axis=-1)
        print(net)
        return net


class Inception_slim:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name

    def build_net(self, image_shape=[150, 150, 3], class_count=2):
        """모델빌드

        args:
            image_shape: image shape. [height, width, chnnel]
        """
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.training = tf.placeholder(tf.bool)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], image_shape[2]])
            self.Y = tf.placeholder(tf.float32, [None, class_count])
            self.learning_rate = tf.placeholder(tf.float32)

            dropout_rate = 0.5
            seed = 777
            n_filters = 64

            net = self.X
            with tf.variable_scope("stem"):
                net = stem(net, self.training)
            with tf.variable_scope("inception-A"):
                for i in range(2):
                    net = inception_A(net, self.training, name="inception_block_a{}".format(i))
            with tf.variable_scope("reduction-A"):
                net = reduction_A(net, self.training)
            with tf.variable_scope("inception-B"):
                for i in range(3):
                    net = inception_B(net, self.training, name="inception_block_b{}".format(i))
            with tf.variable_scope("reduction-B"):
                net = reduction_B(net, self.training)
            with tf.variable_scope("inception-C"):
                for i in range(1):
                    net = inception_C(net, self.training, name="inception_block_c{}".format(i))
            with tf.variable_scope("fc"):
                net = tf.layers.average_pooling2d(name="gap", inputs=net, pool_size=[7, 7],
                                                strides=7, padding='SAME')

            net = tf.reshape(net, [-1, 384])
            net = tf.layers.dropout(net, rate=0.2, training=self.training, seed=seed)

            initializer = tf.contrib.layers.variance_scaling_initializer(seed=seed)
            self.logits = tf.layers.dense(inputs=net, units=class_count, kernel_initializer=initializer)

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y))

        global_step = tf.Variable(0, trainable=False)
        learningRate = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                                global_step= global_step,
                                                decay_steps=5000,
                                                decay_rate= 0.1,
                                                staircase=True)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost, global_step=global_step, name="optimizer")

        self.probability = tf.nn.softmax(self.logits)
        self.prediction = tf.argmax(tf.nn.softmax(self.logits), axis=1)
        correct_prediction = tf.equal(self.prediction, tf.argmax(self.Y, axis=1))
        self.correct_count = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # tensorboard data
        tf.summary.scalar("cost", self.cost)
        tf.summary.scalar("accuracy", self.accuracy)
        self.summary = tf.summary.merge_all()

    def predict(self, x_test):
        return self.sess.run([self.prediction, self.probability], feed_dict={self.X: x_test, self.training: False})

    def eval(self, x_test, y_test):
        return self.sess.run([self.accuracy, self.cost], feed_dict={self.X: x_test, self.Y: y_test, self.training: False})

    def train(self, x_data, y_data, learning_rate, training=True):
        return self.sess.run([self.summary, self.accuracy, self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.learning_rate: learning_rate, self.training: training})

    def _summary(self, x_test, y_test):
        return self.sess.run(self.summary, feed_dict={self.X: x_test, self.Y: y_test, self.training: False})
