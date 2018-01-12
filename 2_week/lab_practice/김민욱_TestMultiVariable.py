import tensorflow as tf

x1_data = [ 73., 93., 89., 96., 73.]
x2_data = [ 80., 88., 91., 98., 66.]
x3_data = [ 73., 93., 90., 100., 70.]

y_data = [152., 185., 180., 196., 142.]

x1 = tf.placeholder(tf.float32, [None])
x2 = tf.placeholder(tf.float32, [None])
x3 = tf.placeholder(tf.float32, [None])
Y  = tf.placeholder(tf.float32, [None])

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b  = tf.Variable(tf.random_normal([1]), name='bais')

# define hyperthesis
hypothesis = x1*w1 + x2*w2 + x3*w3 + b

# define cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# define gradient function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
gradient = optimizer.minimize(cost)

# intialize network
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Go learning
for step in range(5001):
    cost_, h_, _ = sess.run([cost, hypothesis, gradient], feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost : ", cost_, "\nPrediction\n", h_)

