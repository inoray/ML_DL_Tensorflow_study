import tensorflow as tf

x_data = [[1, 2], [2, 3], [3,1], [4,3], [5,3], [6,2]]
y_data = [[0], [0], [0], [1], [1], [1]]

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# define hypothesis
# hypothesis = tf.sigmoid(tf.matmul(X, W) + b )
hypothesis = tf.div(1. ,  1 + tf.exp(tf.matmul(X, W)+b) )

# define cost
cost = - tf.reduce_mean( Y * tf.log(hypothesis) + (1 - Y)*tf.log(1- hypothesis) )

# define gradient
gradient = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy Computation
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32 ))

# init network
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Write Main Source
for step in range(10000):
    cost_, acc_, pre_, _ = sess.run([cost, accuracy, predicted, gradient], feed_dict={X: x_data, Y:y_data})

    if step % 200 == 0:
        print(step, "Cost : ", cost_, "Correct : \n", pre_, "Accuracy : ", acc_)

print("Other Prediction1 : ",  sess.run([hypothesis, predicted], feed_dict={X:[[3, 2]]}) )
print("Other Prediction2 : ",  sess.run([hypothesis, predicted], feed_dict={X:[[3, 4]]}) )