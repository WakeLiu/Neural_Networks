from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections

training_file = 'belling.txt'

def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = np.reshape(np.array([content[i].split() for i in range(len(content))]), [-1, ])
    return content
  
def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

training_data = read_data(training_file)
print("Loaded training data...")
dictionary, reverse_dictionary = build_dataset(training_data)
vocab_size = len(dictionary)

learning_rate = 0.001
training_iters = 50000
display_step = 1000
n_input = 3

# number of units in RNN cell
n_hidden = 512

with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, [None, n_input, 1])

with tf.name_scope('Output'):
    y = tf.placeholder(tf.float32, [None, vocab_size])

with tf.name_scope('RNN'):
    # reshape x to [-1, n_input]
    rnn_input = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    rnn_input = tf.split(rnn_input, n_input, 1)

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    # rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, rnn_input, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    w = tf.Variable(tf.random_normal([n_hidden, vocab_size]))
    b = tf.Variable(tf.random_normal([vocab_size]))
    pred = tf.matmul(outputs[-1], w) + b

with tf.name_scope('Loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    
with tf.name_scope('Optimizer'):
    optimizer = tf.train.RMSPropOptimizer(learning_rate =  learning_rate).minimize(cost)
    
with tf.name_scope('Accuracy'):
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as session:
    session.run(init)
    offset = random.randint(0,n_input+1)
    step, acc_total, loss_total = 0, 0, 0

    while step < training_iters:
        # Generate a minibatch. Add some randomness on selection process.
        if offset > (len(training_data) - n_input - 1):
            offset = random.randint(0, n_input+1)
        '''
        symbols_in format:
        [[[ 4]
          [18]
          [ 1]]]
        '''
        symbols_in = [[dictionary[str(training_data[i])]] for i in range(offset, offset + n_input)]
        symbols_in = np.reshape(np.array(symbols_in), [-1, n_input, 1])
        
        '''
        symbols_out format(onehot vector):
        [[0,0,0,......,1,0,...,0,0,0]]
        '''
        symbols_out = np.zeros([vocab_size], dtype=float)
        symbols_out[dictionary[str(training_data[offset+n_input])]] = 1.0
        symbols_out = np.reshape(symbols_out,[1,-1])
        
        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                feed_dict={x: symbols_in, y: symbols_out})
        loss_total += loss
        acc_total += acc
        if (step + 1) % display_step == 0:
            print("Iter= %d, Average Loss= %.4f, Average Accuracy= %.2f" % (step+1, loss_total/display_step,  100*acc_total/display_step))
            acc_total, loss_total = 0, 0
            words_in = [training_data[i] for i in range(offset, offset + n_input)]
            words_out = training_data[offset + n_input]
            words_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
            print("%s - %s vs %s" % (' '.join(words_in), words_out, words_out_pred))
        step += 1
        offset += (n_input+1)
    
    sentence = 'long ago ,'
    symbols_in_keys = [dictionary[word] for word in sentence.split(' ')]
    for i in range(len(training_data) - n_input):
        keys = np.reshape(np.array(symbols_in_keys[-n_input:]), [-1, n_input, 1])
        onehot_pred = session.run(pred, feed_dict={x: keys})
        onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
        symbols_in_keys.append(onehot_pred_index)
    print(' '.join([reverse_dictionary[key] for key in symbols_in_keys]))

    
