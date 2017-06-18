from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import cv2

tf.logging.set_verbosity(tf.logging.INFO)


def model(X):
	X = tf.divide(X, 255)
	X = tf.reshape(X,[-1, 28, 28, 1])
	conv1 = tf.layers.conv2d(inputs= X,
		filters= 32,
		kernel_size= [5, 5],
		strides= [1, 1],
		padding= 'same',
		activation= tf.nn.relu,
		kernel_initializer= tf.truncated_normal_initializer(stddev= 0.1),
		bias_initializer= tf.zeros_initializer(),
		name= 'conv_layer1')

	pool1 = tf.layers.max_pooling2d(inputs= conv1,
		pool_size= [2, 2],
		padding= 'same',
		strides= 2,
		name= 'max_pooling1')
	
	conv2 = tf.layers.conv2d(inputs= pool1,
		filters= 64,
		kernel_size= [5, 5],
		strides= [1, 1],
		padding= 'same',
		activation= tf.nn.relu,
		kernel_initializer= tf.truncated_normal_initializer(stddev= 0.1),
		bias_initializer= tf.zeros_initializer(),
		name= 'conv_layer2')
	
	pool2 = tf.layers.max_pooling2d(inputs= conv2,
		pool_size= [2, 2],
		padding= 'same',
		strides= 2,
		name= 'max_pooling2')
		
	pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
		
	
	fc1 = tf.layers.dense(inputs= pool2_flat,
		units= 1024,
		activation= tf.nn.relu,
		kernel_initializer= tf.truncated_normal_initializer(stddev= 0.1),
		bias_initializer= tf.zeros_initializer(),
		name= 'fully_connected_layer1')
		
	fc2 = tf.layers.dense(inputs= fc1,
		units= 10,
		activation= None,
		kernel_initializer= tf.truncated_normal_initializer(stddev= 0.1),
		bias_initializer= tf.zeros_initializer(),
		name= 'fully_connected_layer2')
		
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
	batch_size = 128
	
	with tf.name_scope('input'):
		X = tf.placeholder(tf.float32, [None, 784], name= 'in_image')
		label = tf.placeholder(tf.float32, [None, 10], name= 'in_label')
	
	with tf.variable_scope('inference'):
		Y_logit = model(X)	
		Y = tf.nn.softmax(Y_logit, name= 'predict')
		
	tf.add_to_collection('logit_predict', Y_logit)
	
	with tf.name_scope('loss'):
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits= Y_logit, labels=label)
		loss = tf.reduce_mean(cross_entropy)
	with tf.name_scope('evaluation'):
		correct = tf.equal(tf.argmax(Y,1), tf.argmax(label,1))
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) 
	tf.summary.scalar('Accuracy', accuracy)
	
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	
	merged = tf.summary.merge_all()
	init_op = tf.global_variables_initializer()
	saver = tf.train.Saver()
	
	with tf.Session() as sess:
		writer = tf.summary.FileWriter('output', sess.graph)
		sess.run(init_op)
		for i in range(1000):
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
		saver.save(sess,'model')
		
		image = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
		image_flat = np.reshape(image, (-1,784))
		feed_dict = {X: image_flat}
		scores = sess.run(Y, feed_dict)
		print(np.argmax(scores,1))
	return
		
if __name__ == '__main__':
	main()
