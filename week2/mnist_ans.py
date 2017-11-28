import tensorflow as tf
import time
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Agg')
from tensorflow.examples.tutorials.mnist import input_data

# read mnist data from tensorflow example
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

# config
lr = 1e-4 # learning rate
train_steps = 1000
batch_size = 100
logs_path = 'tensorboard/'
n_features = x_train.shape[1] # 784
n_labels = y_train.shape[1] # 10

with tf.Session(config=tf.ConfigProto()) as sess:
  with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, shape=[None, n_features])
  with tf.name_scope('Label'):
    y = tf.placeholder(tf.float32, shape=[None, n_labels])

  with tf.name_scope('conv_1'):
    # [batch, height, width, channel]
    input_conv1 = tf.reshape(x, [-1, 28, 28, 1])
    w_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
    b_conv1 = tf.Variable(tf.constant(0.1, shape = [32]))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(input_conv1, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  with tf.name_scope('conv_2'):
    w_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
    b_conv2 = tf.Variable(tf.constant(0.1, shape = [64]))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  with tf.name_scope('fc_1'):
    input_fc1 = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    w_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
    b_fc1 = tf.Variable(tf.constant(0.1, shape = [1024]))
    h_fc1 = tf.nn.relu(tf.matmul(input_fc1, w_fc1) + b_fc1)

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  with tf.name_scope('output'):
    w_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
    b_fc2 = tf.Variable(tf.constant(0.1, shape = [10]))
    y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

  with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_conv, labels = y))
    tf.summary.scalar('loss', loss)

  with tf.name_scope('optimizer'):
    train = tf.train.AdamOptimizer(lr).minimize(loss)

  with tf.name_scope('accuracy'):
    correct = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

  sess.run(tf.global_variables_initializer())

  merged = tf.summary.merge_all()
  writer = tf.summary.FileWriter(logs_path, graph = sess.graph)

  tic = time.time()
  for i in range(20000):
    x_train, y_train = mnist.train.next_batch(batch_size)
    train.run(feed_dict={x: x_train, y: y_train, keep_prob: 0.5})

    if i % 100 == 0:
      acc, l = sess.run([accuracy, loss], feed_dict={x: x_train, y: y_train, keep_prob: 1.0})
      print('Step %d, accuracy %.2f, loss %.2f' % (i, acc, l))
      summary = sess.run(merged, feed_dict={x: x_train, y: y_train, keep_prob: 1.0})
      writer.add_summary(summary, i)
      writer.flush()

  toc = time.time()
  acc_test = sess.run(accuracy, feed_dict={x: x_test, y: y_test, keep_prob: 1.0})
  print('Test accuracy %.2f' % acc_test)
  print('Spend %.2f sec' % (toc - tic))

  img_l1, img_l2 = sess.run([h_pool1, h_pool2], feed_dict={x: x_train[:1], y: y_test[:1], keep_prob: 1.0})

  img_l1 = np.transpose(img_l1[0], (2, 0, 1))
  img_l2 = np.transpose(img_l2[0], (2, 0, 1))

  fig = plt.figure(figsize=(16,16))

  for i, im in enumerate(img_l1):
    ax = fig.add_subplot(6,6,i+1)
    ax.matshow(im, cmap = plt.get_cmap('gray'))
  plt.show()

  fig = plt.figure(figsize=(16,16))
  for i, im in enumerate(img_l2):
    ax = fig.add_subplot(8,8,i+1)
    ax.matshow(im, cmap = plt.get_cmap('gray'))
  plt.show()
