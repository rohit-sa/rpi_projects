from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides= [1, 1, 1, 1], padding= 'SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
	strides=[1, 2, 2, 1], padding='SAME')

def model(X):
	
	X= tf.reshape(X,[-1, 28, 28, 1])
	with tf.name_scope('layer1'):
		conv1 = tf.layers.conv2d(inputs= X,
		filters= 32,
		kernel_size= [5, 5],
		strides= [1, 1],
		padding= 'same',
		activation= tf.nn.relu,
		kernel_initializer= tf.truncated_normal_initializer(stddev= 0.1),
		bias_initializer= tf.zeros_initializer())
	#~ W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev= 0.1))
	#~ b1 = tf.Variable(tf.constant(0.1, shape= [32]))
	#~ conv1 = tf.nn.relu(conv2d(X, W1) + b1)

	with tf.name_scope('max_pooling1'):
		pool1 = tf.layers.max_pooling2d(inputs= conv1,
		pool_size= [2, 2],
		padding= 'same',
		strides= 2)
		
	#~ pool1 = max_pool_2x2(conv1)	
	
	with tf.name_scope('layer2'):
		conv2 = tf.layers.conv2d(inputs= pool1,
		filters= 64,
		kernel_size= [5, 5],
		strides= [1, 1],
		padding= 'same',
		activation= tf.nn.relu,
		kernel_initializer= tf.truncated_normal_initializer(stddev= 0.1),
		bias_initializer= tf.zeros_initializer())
	#~ W2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev= 0.1))
	#~ b2 = tf.Variable(tf.constant(0.1, shape= [64]))
	#~ conv2 = tf.nn.relu(conv2d(pool1, W2) + b2)
	
	with tf.name_scope('max_pooling2'):
		pool2 = tf.layers.max_pooling2d(inputs= conv2,
		pool_size= [2, 2],
		padding= 'same',
		strides= 2)		
		
	#~ pool2 = max_pool_2x2(conv2)		
	pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
		
	with tf.name_scope('fully_connected_layer1'):
		fc1 = tf.layers.dense(inputs= pool2_flat,
		units= 1024,
		activation= tf.nn.relu,
		kernel_initializer= tf.truncated_normal_initializer(stddev= 0.1),
		bias_initializer= tf.zeros_initializer())
	#~ W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev= 0.1))
	#~ b_fc1 = tf.Variable(tf.constant(0.1, shape= [1024]))
	#~ fc1 = tf.nn.relu(tf.matmul(pool2_flat, W_fc1) + b_fc1)
		
	with tf.name_scope('fully_connected_layer2'):
		fc2 = tf.layers.dense(inputs= fc1,
		units= 10,
		activation= None,
		kernel_initializer= tf.truncated_normal_initializer(stddev= 0.1),
		bias_initializer= tf.zeros_initializer())
	#~ W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev= 0.1))
	#~ b_fc2 = tf.Variable(tf.constant(0.1, shape= [10]))
	#~ fc2 = tf.matmul(fc1, W_fc2) + b_fc2
		
	out = fc2
	return out		

def main():
	if os.path.exists('MINST_data/') and len(os.listdir('MINST_data/')) == 4:
		data = input_data.read_data_sets("MNIST_data/", one_hot=True)
	else:
		from tensorflow.examples.tutorials.mnist import input_data
		data = input_data.read_data_sets("MNIST_data/", one_hot=True)
	
	train_set = data.train
	valid_set = data.validation
	test_set = data.test
	
	learning_rate = 0.001
	batch_size = 64
	
	with tf.name_scope('input'):
		X = tf.placeholder(tf.float32, [None, 784])
		label = tf.placeholder(tf.float32, [None, 10])
		
		
	Y_logit = model(X)	
	Y = tf.nn.softmax(Y_logit)
	
	with tf.name_scope('loss'):
		#~ cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits= Y_logit, labels=label)
		#~ loss = tf.reduce_mean(cross_entropy)
		loss = tf.reduce_mean(-tf.reduce_sum(label * tf.log(Y), reduction_indices=[1]))
	with tf.name_scope('evaluation'):
		correct = tf.equal(tf.argmax(Y,1), tf.argmax(label,1))
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) 
	tf.summary.scalar('Accuracy', accuracy)
	
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	
	merged = tf.summary.merge_all()
	init_op = tf.global_variables_initializer()
	
	with tf.Session() as sess:
		writer = tf.summary.FileWriter('output', sess.graph)
		sess.run(init_op)
		for i in range(500):
			batch_X, batch_Y = train_set.next_batch(batch_size)
			train_data = {X: batch_X, label: batch_Y }
			summary,_ = sess.run([merged, optimizer], feed_dict = train_data)
			writer.add_summary(summary, i)
			a = sess.run([accuracy], feed_dict = train_data)
			if (i+1)%100 == 0:
				print('Train accuracy= {}'.format(a))
		test_data = {X: test_set.images, label: test_set.labels}
		a, c = sess.run([accuracy, loss], feed_dict = test_data)
		print('Test accuracy= {}'.format(a))
		writer.close()
	return
	
if __name__ == '__main__':
	main()
