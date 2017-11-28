from __future__ import print_function
import tensorflow as tf
import numpy as np
import random

def unpickle(file):
  import cPickle
  fo = open(file, 'rb')
  dict = cPickle.load(fo)
  fo.close()
  if 'data' in dict:
    dict['data'] = dict['data'].reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2).reshape(-1, 32*32*3)

  return dict

def load_data_one(f):
  batch = unpickle(f)
  data = batch['data']
  labels = batch['labels']
  print("Loading %s: %d" % (f, len(data)))
  return data, labels

def load_data(files, data_dir, label_count):
  data, labels = load_data_one(data_dir + '/' + files[0])
  for f in files[1:]:
    data_n, labels_n = load_data_one(data_dir + '/' + f)
    data = np.append(data, data_n, axis=0)
    labels = np.append(labels, labels_n, axis=0)
  labels = np.array([ [ float(i == label) for i in xrange(label_count) ] for label in labels ])
  return data, labels

def run_in_batch_avg(session, tensors, batch_placeholders, feed_dict={}, batch_size=128):                              
  res = [ 0 ] * len(tensors)                                                                                           
  batch_tensors = [ (placeholder, feed_dict[ placeholder ]) for placeholder in batch_placeholders ]                    
  total_size = len(batch_tensors[0][1])                                                                                
  batch_count = (total_size + batch_size - 1) / batch_size                                                             
  for batch_idx in xrange(batch_count):                                                                                
    current_batch_size = None                                                                                          
    for (placeholder, tensor) in batch_tensors:                                                                        
      batch_tensor = tensor[ batch_idx*batch_size : (batch_idx+1)*batch_size ]                                         
      current_batch_size = len(batch_tensor)                                                                           
      feed_dict[placeholder] = tensor[ batch_idx*batch_size : (batch_idx+1)*batch_size ]                               
    tmp = session.run(tensors, feed_dict=feed_dict)                                                                    
    res = [ r + t * current_batch_size for (r, t) in zip(res, tmp) ]                                                   
  return [ r / float(total_size) for r in res ]

def _random_crop(batch, crop_shape, padding=None):
        oshape = np.shape(batch[0])
        
        if padding:
            oshape = (oshape[0] + 2*padding, oshape[1] + 2*padding)
        new_batch = []
        npad = ((padding, padding), (padding, padding), (0, 0))
        for i in range(len(batch)):
            new_batch.append(batch[i])
            if padding:
                new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                          mode='constant', constant_values=0)
            nh = random.randint(0, oshape[0] - crop_shape[0])
            nw = random.randint(0, oshape[1] - crop_shape[1])
            new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                                        nw:nw + crop_shape[1]]
        return new_batch

def _random_flip_leftright(batch):
        for i in range(len(batch)):
            if bool(random.getrandbits(1)):
                batch[i] = np.fliplr(batch[i])
        return batch

# training parameter
data_dir = './cifar-10-batches-py'
image_size = 32
image_dim = image_size * image_size * 3
meta = unpickle(data_dir + '/batches.meta')
label_names = meta['label_names']
label_count = len(label_names)

print("label_count %i " % label_count) 

train_files = [ 'data_batch_%d' % d for d in xrange(1, 6) ]
train_data, train_labels = load_data(train_files, data_dir, label_count)
#print("Train:", np.shape(train_data), np.shape(train_labels))
#print("Test:", np.shape(test_data), np.shape(test_labels))
test_data, test_labels = load_data([ 'test_batch' ], data_dir, label_count)
pi = np.random.permutation(len(train_data))
train_data, train_labels = train_data[pi], train_labels[pi]
train_data = train_data.reshape((-1, 32, 32, 3))

mean = np.array([125.3,123.0,113.9])
var = np.array([63.0, 62.1, 66.7])
train_data = train_data - mean[None,None,None,:]
train_data /= var[None,None,None,:]

test_data = test_data.reshape((-1, 32, 32, 3))
test_data = test_data - mean[None,None,None,:]
test_data /= var[None,None,None,:]

data = { 'train_data': train_data,
      'train_labels': train_labels,
      'test_data': test_data.reshape((-1, 32, 32, 3)),
      'test_labels': test_labels }

#define placeholder
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

######################################
# define your model here
######################################

# learning rate
learning_rate = tf.placeholder(tf.float32)

# the loss function 
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

# optimizer SGD
train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cross_entropy+ l2 * 0.0001)

# prediction
correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as session:
    batch_size = 128
    lr = 0.1
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    train_data, train_labels = data['train_data'], data['train_labels']
    batch_count = len(train_data) / batch_size
    batches_data = np.split(train_data[:batch_count * batch_size], batch_count)
    batches_labels = np.split(train_labels[:batch_count * batch_size], batch_count)
    print("Batch per epoch: ", batch_count)
    for epoch in xrange(1, 1+164):
      if epoch == 81: lr = 0.01
      if epoch == 121: lr = 0.001
      for batch_idx in np.random.permutation(range(batch_count)):    #xrange(batch_count):
        xs_, ys_ = batches_data[batch_idx], batches_labels[batch_idx]
        xs_ = _random_crop(xs_, [32, 32, 3], padding=4)
        xs_ = _random_flip_leftright(xs_)
        batch_res = session.run([ train_step, accuracy ],
          feed_dict = {x: xs_, y_: ys_, learning_rate: lr, keep_prob: 0.5 })
        #if batch_idx % 100 == 0: print(epoch, batch_idx, batch_res[1:])

      save_path = saver.save(session, 'NIN_%d.ckpt' % epoch)
      test_results = run_in_batch_avg(session, [accuracy], [x, y_],
          feed_dict = { x: data['test_data'], y_: data['test_labels'], keep_prob: 1.0 })
      print(" %i th epoch accuracy:" % (epoch), test_results)

session.close()
