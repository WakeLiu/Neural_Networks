import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# read mnist data from tensorflow example
# if one_hot is False, it will return 2 instead of [0,0,1,0,0,0,0,0,0,0]
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels



print ("x_train.shape: ",x_train.shape)
print ("y_train.shape: ",y_train.shape)
print (y_train[0])


# config
lr =  0.05# learning rate
train_steps = 1000 #input by wake
batch_size = 100 #input by wake
logs_path = 'tensorboard/'
n_features = x_train.shape[1] # 784
n_labels = y_train.shape[1] # 10

#added by wake, input_size, hidden
input_size = 800
hidden = 1200

#tensor (added by wake)
x = tf.placeholder("float", [None, input_size])
y = tf.placeholder("float", [None, input_size])


# Define weights (added by wake)
Weight = {
    'A' : tf.Variable(tf.random_normal([input_size, hidden])),
    'A_bias': tf.Variable(tf.random_normal([hidden])),

    'B' : tf.Variable(tf.random_normal([hidden, input_size])),
    'B_bias': tf.Variable(tf.random_normal([input_size]))
}


# our model (added by Wake)
temp = tf.matmul(x, Weight['A']) + Weight['A_bias']
output = tf.matmul(temp, Weight['B']) + Weight['B_bias']


# Define loss and optimizer (added by wake)
cost = tf.reduce_mean(tf.square(output-y))
optimizer = tf.train.AdamOptimizer(lr).minimize(cost)


with tf.Session(config=tf.ConfigProto()) as sess:
    # with tf.name_scope('inputs'):
    #
    # with tf.name_scope('labels'):
    #
    # # build variables
    # with tf.name_scope('params'):
    #
    # # build model
    # with tf.name_scope('model'):
    #
    # # define loss
    # with tf.name_scope('loss'):
    #
    # # Gradient Descent
    # with tf.name_scope('gd'):
    #
    # with tf.name_scope('accuracy'):

    # start run
    initializer = tf.global_variables_initializer()
    sess.run(initializer)

    # display data
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logs_path, graph = sess.graph)

    for step in range(train_steps):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict = {x: x_batch, y: y_batch})

        if step % 50 == 0:
            l, summary = sess.run([loss, merged], feed_dict = {x: x_batch, y: y_batch})
            ac = sess.run(acc, feed_dict={x: x_test, y: y_test})
            writer.add_summary(summary, step)

            print("Accuracy: %.2f, Loss: %.2f" % (ac, l))
