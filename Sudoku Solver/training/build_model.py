from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

def model(X):
	
	with tf.name_scope('layer1'):
		with tf.name_scope('weights'):
			W1 = tf.Variable(tf.truncated_normal([784, 200], stddev = 0.5), name = 'weights')
		with tf.name_scope('biases'):
			b1 = tf.Variable(tf.zeros([200]), name = 'biases')
		
		h1 = tf.add(tf.matmul(X, W1),b1)
		h1 = tf.nn.relu(h1)
		
	with tf.name_scope('layer2'):
		with tf.name_scope('weights'):
			W2 = tf.Variable(tf.truncated_normal([200, 10], stddev = 0.5), name = 'weights')
		with tf.name_scope('biases'):
			b2 = tf.Variable(tf.zeros([10]), name = 'biases')
		
		out = tf.add(tf.matmul(h1, W2),b2)
		
	return out	
	
def model1(X):
	
	with tf.name_scope('layer1'):
		fc1 = tf.layers.dense(inputs= X,
		units= 200,
		activation= tf.nn.relu,
		kernel_initializer= tf.truncated_normal_initializer(stddev= 0.5),
		bias_initializer= tf.zeros_initializer())
		
	with tf.name_scope('layer2'):
		fc2 = tf.layers.dense(inputs= fc1,
		units= 10,
		activation= None,
		kernel_initializer= tf.truncated_normal_initializer(stddev= 0.5),
		bias_initializer= tf.zeros_initializer())
		
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
	with tf.name_scope('input'):
		X = tf.placeholder(tf.float32, [None, 784])
		label = tf.placeholder(tf.float32, [None, 10])
		
	Y_logit = model1(X)	
	Y = tf.nn.softmax(Y_logit)
	
	with tf.name_scope('loss'):
		loss = tf.nn.softmax_cross_entropy_with_logits(logits= Y_logit, labels=label)
		#~ loss = -tf.reduce_sum(label * tf.log(Y))
		#~ loss = tf.losses.softmax_cross_entropy(onehot_labels= label, logits= Y_logit)
	with tf.name_scope('evaluation'):
		correct = tf.equal(tf.argmax(Y,1), tf.argmax(label,1))
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) 
	tf.summary.scalar('Accuracy', accuracy)
	#~ tf.summary.scalar('Loss', loss)
	optimizer = tf.train.GradientDescentOptimizer(0.003).minimize(loss)
	
	merged = tf.summary.merge_all()
	init_op = tf.global_variables_initializer()
	
	with tf.Session() as sess:
		writer = tf.summary.FileWriter('output', sess.graph)
		sess.run(init_op)
		for i in range(1000):
			batch_X, batch_Y = train_set.next_batch(100)
			train_data = {X: batch_X, label: batch_Y }
			summary,_ = sess.run([merged, optimizer], feed_dict = train_data)
			writer.add_summary(summary, i)
			a = sess.run([accuracy], feed_dict = train_data)
		test_data = {X: test_set.images, label: test_set.labels}
		a, c = sess.run([accuracy, loss], feed_dict = test_data)
		print(a)
		writer.close()
	return
	
if __name__ == '__main__':
	main()
