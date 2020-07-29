import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np



def BiRNN_new(x, c, l, num_layers, num_hidden, meta_data,
              num_classes, timesteps, keep_prob, uncertainty, 
              is_train=True, use_sequence_lengths=False, use_embedding=True,
              sequence_length=66, dict_size=32):

        if use_embedding:
            emb = tf.keras.layers.Embedding(dict_size, num_hidden, input_length=sequence_length)
            x = emb(x)

        num_layers=num_layers
        with tf.name_scope("birnn"):
            #lstm_fw_cell = [tf.contrib.rnn.DropoutWrapper(rnn.BasicLSTMCell(num_hidden), input_keep_prob = keep_prob) for _ in range(num_layers)]
            #lstm_bw_cell = [tf.contrib.rnn.DropoutWrapper(rnn.BasicLSTMCell(num_hidden), input_keep_prob = keep_prob) for _ in range(num_layers)]
            #lstm_fw_cell = [rnn.ConvLSTMCell(1,[66,32],128, (5,1)) for _ in range(num_layers)]	
            #lstm_bw_cell = [rnn.ConvLSTMCell(1,[66,32],128, (5,1)) for _ in range(num_layers)]
            lstm_fw_cell = [rnn.BasicLSTMCell(num_hidden) for _ in range(num_layers)]
            lstm_bw_cell = [rnn.BasicLSTMCell(num_hidden) for _ in range(num_layers)]
            #

        if use_sequence_lengths:
            rnn_outputs_all, final_fw, final_bw = rnn.stack_bidirectional_dynamic_rnn(
                lstm_fw_cell, lstm_bw_cell, x, sequence_length=l, dtype=tf.float32)
        else:
            rnn_outputs_all, final_fw, final_bw = rnn.stack_bidirectional_dynamic_rnn(
                lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
            final_fw = final_fw[-1].h
            final_bw = final_bw[-1].h
		
		
        out = tf.concat([final_fw, final_bw], 1)
        print(out, final_bw, final_fw)
        feat = tf.concat([out, c], axis=1)

        #add another layer !
        l1 = tf.contrib.slim.fully_connected(feat, num_hidden)
        # l1 = tf.layers.dropout(l1, rate=keep_prob, training=is_train)

        # l2 = tf.contrib.slim.fully_connected(l1, num_hidden)
        preds = tf.contrib.slim.fully_connected(l1, num_classes, activation_fn=None)
        # maybe we need to return somethin for attention score
        return preds, l1, None, None, None, None

def BiRNN(x, c, num_hidden, meta_data, num_classes, timesteps):

	# Define weights
	weights = {
	# Hidden layer weights => 2*n_hidden +1 because of forward + backward cells + charge

	'l1' : tf.Variable(tf.random_normal([2*num_hidden+meta_data.shape[1], num_hidden])),
	'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))

	}

	biases = {
	'l1': tf.Variable(tf.random_normal([num_hidden])),
	'out': tf.Variable(tf.random_normal([num_classes]))
	}

	# Prepare data shape to match `rnn` function requirements
	# Current data input shape: (batch_size, timesteps, n_input)
	# Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

	# Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
	x = tf.unstack(x, timesteps, 1)

	#add encoder

	# Define lstm cells with tensorflow
	# Forward direction cell
	lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
	# Backward direction cell
	lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

	# Get lstm cell output
	outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
		dtype=tf.float32)

	#add decoder
	feat = tf.concat([outputs[-1], c], axis=1)   
	l1 = tf.nn.relu(tf.matmul(feat, weights['l1']) + biases['l1'])
	preds = tf.matmul(l1, weights['out']) + biases['out']

	# Linear activation, using rnn inner loop last output
	return preds, l1, weights, biases, _

#def conv_net(x, c, l, num_hidden, meta_data, num_classes, timesteps, keep_prob, uncertainty):
def conv_net(x, c, l, num_layers, num_hidden, meta_data, num_classes, timesteps, keep_prob, uncertainty, is_train=True):

	# Define a scope for reusing the variables
	with tf.variable_scope('ConvNet'):
		x = x[...,tf.newaxis]
		# Convolution Layer with 32 filters and a kernel size of 5
		net = tf.layers.conv1d(x, 32, 8, activation=tf.nn.relu, padding='SAME')
		net = tf.layers.conv1d(net, 64, 6, activation=tf.nn.relu, padding='SAME')
		net = tf.layers.conv1d(net, 128, 5, activation=tf.nn.relu, padding='SAME')
		net = tf.layers.conv1d(net, 128, 3, activation=tf.nn.relu, padding='SAME')
		net = tf.layers.conv1d(net, 128, 3, activation=tf.nn.relu, padding='SAME')
		net = tf.layers.conv1d(net, 128, 3, activation=tf.nn.relu, padding='SAME')
		s = tf.shape(net)
		print(net)

		net = tf.layers.average_pooling1d(
										net,
										(66),
										strides=1,
										)
		print(net)

		net = tf.contrib.layers.flatten(net)
		net = tf.concat([net, c], axis=1)   

	# Fully connected layer (in tf contrib folder for now
	fc1 = tf.layers.dense(net, num_hidden)
	# Apply Dropout (if is_training is False, dropout is not applied)
	# Output layer, class prediction
	preds = tf.layers.dense(fc1, num_classes, activation=None)
	print(preds)
	#sys.exit()
	return preds, fc1, None, None, None, None

def mlp(x, c, l, num_layers, num_hidden, meta_data, num_classes, timesteps, keep_prob, uncertainty, is_train=True):
    # Define a scope for reusing the variables
    with tf.variable_scope('MLP'):
        net = tf.contrib.layers.flatten(x)
        net = tf.concat([net, c], axis=1)   
        fc = net
        for _ in range(num_layers):
            fc = tf.layers.dense(fc, num_hidden, activation=tf.nn.relu)
            fc = tf.layers.dropout(fc, rate=keep_prob)
        preds = tf.layers.dense(fc, num_classes, activation=None)

           # Fully connected layer (in tf contrib folder for now
           # Apply Dropout (if is_training is False, dropout is not applied)
           # Output layer, class prediction
        return preds, fc, None, None, None, None

def logreg(x, c, l, num_layers, num_hidden, meta_data, num_classes, timesteps, keep_prob, uncertainty, is_train=True):

    # Define a scope for reusing the variables
    with tf.variable_scope('LogReg'):
        net = tf.contrib.layers.flatten(x)
        net = tf.concat([net, c], axis=1)   
        preds = tf.layers.dense(net, num_classes)
        # Fully connected layer (in tf contrib folder for now
        # Apply Dropout (if is_training is False, dropout is not applied)
        # Output layer, class prediction
        return preds, tf.zeros_like(preds), None, None, None, None
