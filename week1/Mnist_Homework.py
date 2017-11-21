
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')

# read mnist data from tensorflow example
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
x_train = mnist.train.images #train_img
y_train = mnist.train.labels #train_labels
x_test = mnist.test.images #test_img
y_test = mnist.test.labels #test_labels




# In[2]:



input_size = 784
x = tf.placeholder("float", [None, 784])

#y is used to put the correct answer
y = tf.placeholder("float", [None, 10])


# In[3]:


# config
lr =  0.5# learning rate

batch_size = 100
logs_path = 'tensorboard/'
n_features = x_train.shape[1] # 784
n_labels = y_train.shape[1] # 10
hidden = 10

#print out the features and labels
print (n_features, n_labels)


# In[4]:


# Define weights
Weight = {
    'A' : tf.Variable(tf.random_normal([784, hidden])),
    'A_bias': tf.Variable(tf.random_normal([hidden])),

    'B' : tf.Variable(tf.random_normal([hidden, 10])),
    'B_bias': tf.Variable(tf.random_normal([10]))
}

#Our model
temp = tf.matmul(x, Weight['A']) + Weight['A_bias']
#output = tf.matmul(temp, Weight['B']) + Weight['B_bias']

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
output = tf.nn.softmax(tf.matmul(x, W) + b)

print (output)
print (y)


# In[6]:


# Define loss and optimizer
cost = tf.reduce_mean(tf.square(output-y))
#cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(output), reduction_indices=[1]))

#89
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)

#88
#optimizer = tf.train.AdadeltaOptimizer(lr).minimize(cost)

#69
#optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

#AdagradDAOptimizer, 89
#optimizer = tf.train.AdagradDAOptimizer(lr).minimize(cost)

#MomentumOptimizer
#optimizer = tf.train.MomentumOptimizer(lr).minimize(cost)
print (cost)


# In[9]:


train_steps = 1000

if True:
    sess = tf.Session()
#with tf.Session(config=tf.ConfigProto()) as sess:
    """
    with tf.name_scope('inputs'):

    with tf.name_scope('labels'):

    # build variables
    with tf.name_scope('params'):

    # build model
    with tf.name_scope('model'):

    # define loss
    with tf.name_scope('loss'):

    # Gradient Descent
    with tf.name_scope('gd'):

    with tf.name_scope('accuracy'):
    """

    # start run
    initializer = tf.global_variables_initializer()
    sess.run(initializer)

    # display data
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logs_path, graph = sess.graph)
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print (correct_prediction)

    for step in range(train_steps):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict = {x: x_batch, y: y_batch})

        if step % 50 == 0:
            loss = sess.run(cost, feed_dict={x: x_batch, y: y_batch})
            summary = sess.run(merged, feed_dict = {x: x_batch, y: y_batch})
            ac = sess.run(acc, feed_dict={x: x_test, y: y_test})
            writer.add_summary(summary, step)

            print("Accuracy: %.2f, Loss: %.2f" % (ac,loss))




# In[ ]:


print(sess.run(output[0,:], feed_dict = {x: mnist.test.images, y: mnist.test.labels}))


# In[ ]:


print(correct_prediction)
