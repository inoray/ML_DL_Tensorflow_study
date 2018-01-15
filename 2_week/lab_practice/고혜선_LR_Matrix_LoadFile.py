import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data, len(y_data))

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x, w) + b
cost = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range (2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x : x_data, y : y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\mPrediction:\n", hy_val)

print("-------------------------------\n")
print("Your score: ", sess.run(hypothesis, feed_dict={x: [[10, 70, 101]]}))
print("other score: ", sess.run(hypothesis, feed_dict={x: [[60, 70, 110], [90, 100, 80]]}))


