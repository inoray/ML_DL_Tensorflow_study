import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import random
import matplotlib.pyplot as plt
tf.set_random_seed(777)  # for reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_class = 10

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, nb_class])

W = tf.Variable(tf.random_normal([784, nb_class]))
b = tf.Variable(tf.random_normal([nb_class]))

# define Hypothesis
logits = tf.matmul(X, W)+b
hypothesis = tf.nn.softmax(tf.matmul(X, W)+b)

# define cost function
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
cost = tf.reduce_mean(cost_i)

# define gradient
gradient = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# init session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# implement logic
training_epoch = 15
batch_size = 100

for epoch in range(training_epoch):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_X, batch_Y = mnist.train.next_batch(batch_size)

        c, _ = sess.run([cost, gradient], feed_dict={X: batch_X, Y: batch_Y})
        avg_cost += c / total_batch
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print("Learning finished")


# Test Model
predict = tf.argmax(hypothesis, 1)
answer = tf.argmax(Y, 1)
accuracy = tf.reduce_mean( tf.cast(tf.equal(predict, answer), dtype=tf.float32) )

print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))

# Get one and predict
r = random.randint(0, mnist.test.num_examples-1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
print("Prediction: ", sess.run(predict, feed_dict={X:mnist.test.images[r:r+1]}) )

# View
plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
plt.show()

sess.close()








