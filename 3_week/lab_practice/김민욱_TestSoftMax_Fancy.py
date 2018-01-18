import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]


X = tf.placeholder(tf.float32, shape=[None, 16])
Y = tf.placeholder(tf.int32, shape=[None, 1]) # Integer - 0~7 shape = (none, 1))

nb_classes = 7
Y_one_hot = tf.one_hot(Y, nb_classes) # 한차원을 더한 값을 리턴한다. one hot shape = (none, 1, 7)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) # 차원을 줄여야 한다. shape = (none, 7)

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# define hypothesis
logits = tf.matmul(X, W) + b # hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)
hypothesis = tf.nn.softmax(logits) # for Test

# define cost
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)  # cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))

# define gradient
gradient = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Calc Accuracy
predict = tf.arg_max(hypothesis, 1)
correct_predict = tf.equal(predict, tf.arg_max(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, dtype=tf.float32))

# init network
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# go implement
for step in range(2001):
    cost_, accu_, _ = sess.run([cost, accuracy, gradient], feed_dict={X: x_data, Y:y_data})

    if step % 200 == 0:
        print("Step: {:5}\tLoss: {:.3f}\tAcc:{:.2%}".format(step, cost_, accu_) )

pred = sess.run(predict, feed_dict={X: x_data})
for p, y in zip(pred, y_data.flatten()):
    print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))






