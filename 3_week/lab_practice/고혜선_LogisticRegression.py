import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

# x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
# y_data = [[0], [0], [0], [1], [1], [1]]
# x = tf.placeholder(tf.float32, shape=[None, 2])
# y = tf.placeholder(tf.float32, shape=[None, 1])
# w = tf.Variable(tf.random_normal([2, 1]), name='weight')
# b = tf.Variable(tf.random_normal([1]), name='bias')

# xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
# x_data = xy[:, 0:-1]
# y_data = xy[:, [-1]]
# print(x_data.shape, y_data.shape)

filename_queue = tf.train.string_input_producer(['data-03-diabetes.csv'], shuffle=False, name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_default = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], ]
xy = tf.decode_csv(value, record_defaults=record_default)

x_data, y_data = tf.train.batch([xy[0:-1] , xy[-1:]], batch_size=800)

x = tf.placeholder(tf.float32, shape=[None, 8])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)
cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(10001) :
        x_batch, y_batch = sess.run([x_data, y_data])
        cost_val, _ = sess.run([cost, train], feed_dict={x: x_batch, y: y_batch})
        if step % 200 == 0:
            print(step, cost_val)

    coord.request_stop()
    coord.join(threads)

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={x: x_batch, y:y_batch})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: " , a)
