import tensorflow as tf
import numpy as np

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0],     [1],     [1],    [0]], dtype=np.float32)

X = tf.placeholder(tf.float32, shape=[None ,2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

numOfnodes = 3

W1 = tf.Variable(tf.random_normal([2, numOfnodes]), dtype=tf.float32, name='weight1')
b1 = tf.Variable(tf.random_normal([1, numOfnodes]), dtype=tf.float32, name='bias1')

W2 = tf.Variable(tf.random_normal([numOfnodes, 1]), dtype=tf.float32, name='weight2')
b2 = tf.Variable(tf.random_normal([1, 1]), dtype=tf.float32, name='bias2')

# define hypothesis
g = tf.nn.sigmoid(tf.matmul(X, W1)+ b1)
hypothesis = tf.nn.sigmoid(tf.matmul(g, W2)+ b2)

# define cost
cost = - tf.reduce_mean( Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis) )

# define gradient
gradient = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# init network
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# calc accuracy
predict = tf.cast(hypothesis>0.5, dtype=tf.float32)
accuracy = tf.reduce_mean( tf.cast( tf.equal(predict, Y), dtype=tf.float32 ))

# implement logics
for step in range(10001):
    acc_, cost_, _ = sess.run([accuracy, cost, gradient], feed_dict={X: x_data, Y: y_data})

    if step % 200 == 0:
        print(step, "Cost : ", cost_, "Accuracy : ", acc_)

h_ = sess.run(hypothesis, feed_dict={X: x_data, Y: y_data})
print("hypothesis : ", h_)




