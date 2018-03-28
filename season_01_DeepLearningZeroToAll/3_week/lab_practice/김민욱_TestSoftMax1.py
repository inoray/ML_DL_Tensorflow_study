import tensorflow as tf

x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3,1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 3])

W = tf.Variable(tf.random_normal([4, 3]), name='weight')
b = tf.Variable(tf.random_normal([3]), name='bias')

# define hypothesis
hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)

# define cost
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))

#define gradient
gradient = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Init network
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# predict
predict = tf.cast(hypothesis>0.5, dtype=tf.float32)
accracy = tf.reduce_mean(tf.cast(tf.equal(predict, Y), dtype=tf.float32))

# write go
for step in range(2001):
    acc_, pre_, cost_, _ = sess.run([accracy,predict, cost, gradient], feed_dict={X: x_data, Y: y_data})

    if step % 200 == 0:
        print(step, cost_, pre_, acc_)

a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
print( a, sess.run(tf.argmax(a, 1)))

