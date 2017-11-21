import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# read mnist data from tensorflow example
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

# look into the feature of the data
# [data_num, feature_num]
print('--- x train shape ---')
print(x_train.shape)
# [data_num, output_num]
print('--- y train shape ---')
print(y_train.shape)
print('--- x train shape ---')
print(x_test.shape)
print('--- y test shape ---')
print(y_test.shape)


print('---- input ----')
print(x_train[0, :])
print('---- output ----')
print(np.argmax(y_train[1, :]))

# config
lr = 0.01 # learning rate
train_steps = 1000
batch_size = 100
logs_path = 'tensorboard/'
n_features = x_train.shape[1] # 784
n_labels = y_train.shape[1] # 10

with tf.Session(config=tf.ConfigProto()) as sess:
    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, [None, n_features], name = 'digit_input')
    
    with tf.name_scope('labels'):
        y = tf.placeholder(tf.float32, [None, n_labels], name = 'digit_label')

    # build variables
    with tf.name_scope('params'):
        w = tf.Variable(tf.zeros([n_features, n_labels]), name = 'weight')
        b = tf.Variable(tf.zeros([n_labels]), name = 'bias')
    
    # build model
    with tf.name_scope('model'):
        prediction = tf.nn.softmax(tf.matmul(x, w) + b)
    
    # define loss
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices = 1))
        tf.summary.scalar('Loss', loss)
    
    # Gradient Descent
    with tf.name_scope('gd'):
        optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('Accuracy', acc)
    
    # start run
    initializer = tf.global_variables_initializer()
    sess.run(initializer)
    
    # display data
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logs_path, graph = sess.graph)
    
    for step in xrange(train_steps):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict = {x: x_batch, y: y_batch})
    
        if step % 50 == 0:
            l, summary = sess.run([loss, merged], feed_dict = {x: x_batch, y: y_batch})
            ac = sess.run(acc, feed_dict={x: x_test, y: y_test})
            writer.add_summary(summary, step)
            
            print("Accuracy: %.2f, Loss: %.2f" % (ac, l))
