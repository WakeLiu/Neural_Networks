import tensorflow as tf
from numpy import random

import numpy as np

# import "imsave" function
from scipy.misc import *

# training information
save_route = 'save_hw3/save_net.ckpt'

# Model Parameters
training_iters = 500000000
batch_size = 1
display_step = 5000

# Training Parameters
resume = 1 # 1: yes, else: repeat
save_step = 1000
object_limit = 10

# learning rate
learning_rate = 0.001 # 0.001

# Network Parameters
n_inputs = 10 # input frame size
n_steps = 20 # timesteps
p_steps = int(n_steps/2)
n_hiddens = 64 # hidden layer num of features (128 hidden nodes)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_inputs])
y = tf.placeholder("float", [None, n_steps/2, n_inputs])

RNN_W = {
    'Wi' : tf.Variable(tf.random_normal([n_inputs, n_hiddens])),
    'bi' : tf.Variable(tf.random_normal([n_hiddens])),
    'Wo' : tf.Variable(tf.random_normal([n_hiddens, n_inputs])),
    'bo' : tf.Variable(tf.random_normal([n_inputs]))
}

def RNN(x):

    # reshape -> 2D
    x = tf.reshape(x, [-1, n_inputs])

    # input layer
    x = tf.matmul(x, RNN_W['Wi']) + RNN_W['bi']

    # reshape -> 3D
    rnn_input = tf.reshape(x, [-1, n_steps, n_hiddens])

    # rnn cell initialization
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hiddens)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    # rnn part
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, rnn_input, initial_state=init_state, time_major=False)

    # reshape
    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))

    results = []

    for i in range(p_steps, n_steps, 1):
        temp = tf.matmul(outputs[i], RNN_W['Wo']) + RNN_W['bo']
        results.append(temp)

    results = tf.unstack(tf.transpose(results, [1,0,2]))

    return results

# training
rnn_pred = RNN(x)

# Define loss and optimizer
cost = tf.reduce_mean(tf.square(rnn_pred-y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# save
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 0

    # resume or not
    if resume == 1:
        save_path = saver.save(sess, save_route)
        print("Save to path: ", save_path)
    else:
        saver.restore(sess, save_route)

    # Training
    while step * batch_size < training_iters:

        # random batch video
        data = np.random.randint(0, object_limit, size=int(n_steps/2))

        # data pair
        b_x, b_y = [], []

        for i in range(0, batch_size, 1):

            # initialization
            data_x = np.zeros((n_steps, n_inputs))
            data_y = np.zeros((p_steps, n_inputs))

            for j in range(0, p_steps, 1):
                data_x[j, data[j]] = 1.0
                data_y[p_steps-1-j, data[j]] = 1.0

            b_x.append(data_x)
            b_y.append(data_y)

        sess.run(optimizer, feed_dict={x: b_x, y: b_y})

        # display
        if step % display_step == 0:

            # output loss
            loss = sess.run(cost, feed_dict={x: b_x, y: b_y})
            print("Iter " + str(step*batch_size) + ", loss= " + "{:.6f}".format(loss))

            # output answer and predict
            pre = sess.run(rnn_pred, feed_dict={x: b_x, y: b_y})
            show_pre = np.random.randint(0, 1, size=(p_steps))

            for i in range(0, p_steps, 1):
                maxv = -1
                maxi = -1
                for j in range(0, n_inputs, 1):
                    if pre[0][i, j] > maxv:
                        maxv = pre[0][i, j]
                        maxi = j

                show_pre[p_steps-1-i] = int(maxi)

            print (" input: " + str(data))
            print ("output: " + str(show_pre))

        # save
        if step % save_step == 0:
            save_path = saver.save(sess, save_route)

        step += 1

    print("Optimization Finished!")
