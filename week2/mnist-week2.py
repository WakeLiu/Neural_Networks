import tensorflow as tf
import time
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
  
  with tf.name_scope('Label'):

  with tf.name_scope('conv_1'):
    # [batch, height, width, channel]

  with tf.name_scope('conv_2'):

  with tf.name_scope('fc_1'):

  with tf.name_scope('dropout'):

  with tf.name_scope('output'):

  with tf.name_scope('loss'):

  with tf.name_scope('optimizer'):

  with tf.name_scope('accuracy'):

  sess.run(tf.global_variables_initializer())

  merged = tf.summary.merge_all()
  writer = tf.summary.FileWriter(logs_path, graph = sess.graph)

  tic = time.time()
  for i in range(20000):
    x_train, y_train = mnist.train.next_batch(batch_size)
    train.run(feed_dict={x: x_train, y: y_train, keep_prob: 0.5})

    if i % 50 == 0:
      acc, l = sess.run([accuracy, loss], feed_dict={x: x_train, y: y_train, keep_prob: 1.0})
      print('Step %d, accuracy %.2f, loss %.2f' % (i, acc, l))
      summary = sess.run(merged, feed_dict={x: x_train, y: y_train, keep_prob: 1.0})
      writer.add_summary(summary, i)
      writer.flush()

  toc = time.time()
  acc_test = sess.run(accuracy, feed_dict={x: x_test, y: y_test, keep_prob: 1.0})
  print('Test accuracy %.2f' % acc_test)
  print('Spend %.2f sec' % (toc - tic))
