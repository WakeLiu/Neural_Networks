
"""Builds the model.

Inputs:
  image_feature
  input_seqs
  keep_prob 
  target_seqs 
  input_mask 
Outputs:
  total_loss 
  preds 
"""

import tensorflow as tf

def build_model(config, mode, inference_batch = None, glove_vocab = None):

    """Basic setup.

    Args:
      config: Object containing configuration parameters.
      mode: "train" or "inference".
      inference_batch: if mode is 'inference', we will need to provide the batch_size of input data. Otherwise, leave it as None. 
      glove_vocab: if we need to use glove word2vec to initialize our vocab embeddings, we will provide with a matrix of [config.vocab_size, config.embedding_size]. If not, we leave it as None. 
    """
    assert mode in ["train", "inference"]
    if mode == 'inference' and inference_batch is None:
        raise ValueError("When inference mode, inference_batch must be provided!")
    config = config

    # To match the "Show and Tell" paper we initialize all variables with a
    # random uniform initializer.
    initializer = tf.random_uniform_initializer(
        minval=-config.initializer_scale,
        maxval=config.initializer_scale)

    # An int32 Tensor with shape [batch_size, padded_length].
    input_seqs = tf.placeholder(tf.int32, [None, None], name='input_seqs')

    # An int32 Tensor with shape [batch_size, padded_length].
    target_seqs = tf.placeholder(tf.int32, [None, None], name='target_seqs')    
    
    # A float32 Tensor with shape [1]
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # An int32 0/1 Tensor with shape [batch_size, padded_length].
    input_mask = tf.placeholder(tf.int32, [None, None], name='input_mask')
    
    # A float32 Tensor with shape [batch_size, image_feature_size].
    image_feature = tf.placeholder(tf.float32, [None, config.image_feature_size], name='image_feature')

    # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
    seq_embedding = None

    # A float32 scalar Tensor; the total loss for the trainer to optimize.
    total_loss = None

    # A float32 Tensor with shape [batch_size * padded_length].
    target_cross_entropy_losses = None

    # A float32 Tensor with shape [batch_size * padded_length].
    target_cross_entropy_loss_weights = None

    # Collection of variables from the inception submodel.
    inception_variables = []

    # Global step Tensor.
    global_step = None
    
    """Sets up the global step Tensor."""
    global_step = tf.Variable(
    initial_value=0,
    name="global_step",
    trainable=False,
    collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
    
    ### Builds the input sequence embeddings ###
    # Inputs:
    #   self.input_seqs
    # Outputs:
    #   self.seq_embeddings
    ############################################

    with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
        if glove_vocab is None:
            embedding_map = tf.get_variable(
                name="map",
                shape=[config.vocab_size, config.embedding_size],
                initializer=initializer)
        else:
            init = tf.constant(glove_vocab.astype('float32'))
            embedding_map = tf.get_variable(
                name="map",
                initializer=init)
        seq_embedding = tf.nn.embedding_lookup(embedding_map, input_seqs)

    ############ Builds the model ##############
    # Inputs:
    #   self.image_feature
    #   self.seq_embeddings
    #   self.target_seqs (training and eval only)
    #   self.input_mask (training and eval only)
    # Outputs:
    #   self.total_loss (training and eval only)
    #   self.target_cross_entropy_losses (training and eval only)
    #   self.target_cross_entropy_loss_weights (training and eval only)
    ############################################

    ##############################################
    # TO DO A (LSTM initialization)
    ##############################################

    with tf.variable_scope("lstm", initializer=initializer) as lstm_scope:
    
      # Feed the image embeddings to set the initial LSTM state.
      if mode == 'train':
          zero_state = lstm_cell.zero_state(
              batch_size=config.batch_size, dtype=tf.float32)
      elif mode == 'inference':
          zero_state = lstm_cell.zero_state(
              batch_size=inference_batch, dtype=tf.float32)
              
      with tf.variable_scope('image_embeddings'):
          ##############################################
          # TO DO B (fully-connected)
          ##############################################

      ##############################################
      # TO DO C (LSTM predict)
      ##############################################


    with tf.variable_scope('logits'):
        ##############################################
        # TO DO D (fully-connected)
        ##############################################

    ###### for inference & validation only #######
    softmax = tf.nn.softmax(logits)
    preds = tf.argmax(softmax, 1)
    ##############################################
    
    # for training only below 
    targets = tf.reshape(target_seqs, [-1])
    weights = tf.to_float(tf.reshape(input_mask, [-1]))

    # Compute losses.
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                            logits=logits)
    batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                        tf.reduce_sum(weights),
                        name="batch_loss")
    tf.contrib.losses.add_loss(batch_loss)
    total_loss = tf.contrib.losses.get_total_loss()
    
    # target_cross_entropy_losses = losses  # Used in evaluation.
    # target_cross_entropy_loss_weights = weights  # Used in evaluation.

    return dict(
        total_loss = total_loss, 
        global_step = global_step, 
        image_feature = image_feature, 
        input_mask = input_mask, 
        target_seqs = target_seqs, 
        input_seqs = input_seqs, 
        final_state = final_state,
        initial_state = initial_state, 
        softmax = softmax,
        preds = preds, 
        keep_prob = keep_prob, 
        saver = tf.train.Saver()
    )

