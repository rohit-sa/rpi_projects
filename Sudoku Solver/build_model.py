from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gzip
import struct
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

#~ tf.logging.set_verbosity(tf.logging.INFO)

"""
function " download_dataset "
input parameters: save path
output parameters: download successful 
Notes: Downloads MISNT dataset into save path
"""
def download_dataset(save_path):
	import requests
	website = 'http://yann.lecun.com/exdb/mnist/'
	file_name = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
	't10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz']
	
	url = [website+name for name in file_name]
	for i in range(len(url)):
		print('Downloading ' + file_name[i])
		response =requests.get(url[i])
		if response.status_code != requests.codes.ok:
			print('Check internet connectivity')
			break
		with open(save_path+file_name[i],'wb') as f:
			f.write(response.content)
	print('Download complete')
	return True
"""
function " get_dataset "
input parameters: none
output parameters: train and test images, labels
Notes: checks if dataset is present else downloads it. Extracts
and returns images and labels
"""
def get_dataset():
	save_path = 'training/MINST_data/'
	if not os.path.exists('training/MINST_data/') or len(os.listdir('training/MINST_data/')) != 4:
		os.makedirs(save_path)
		download_dataset(save_path)
	
	train_images = unzip_images(save_path+'train-images-idx3-ubyte.gz')
	test_images = unzip_images(save_path+'t10k-images-idx3-ubyte.gz')
	train_labels = unzip_labels(save_path+'train-labels-idx1-ubyte.gz')
	test_labels = unzip_labels(save_path+'t10k-labels-idx1-ubyte.gz')	
	
	return	train_images, train_labels, test_images, test_labels
"""
function " unzip_images "
input parameters: file_name
output parameters: images
Notes: upacks bytes in images in shape total number, rows, cols
Also adds blank image to dataset.
"""
def unzip_images(file_name):
	images = None
	with gzip.open(file_name,'rb') as file_content:
		file_content.seek(4)
		num_images = struct.unpack('>L',file_content.read(4))[0]
		rows = struct.unpack('>L',file_content.read(4))[0]
		cols = struct.unpack('>L',file_content.read(4))[0]
		total_size = rows*cols*num_images
		images = np.frombuffer(file_content.read(total_size),dtype = np.uint8).astype(np.float32)
		empty_images = np.zeros((total_size//10),dtype=np.float32)
		noise_index = np.random.randint(784,size=100)
		empty_images[noise_index] = 200
		images = np.concatenate((images,empty_images))
		num_images += num_images//10
		images = np.reshape(images,(num_images,rows,cols,1))
	return images
"""
function " unzip_labels "
input parameters: file_name
output parameters: labels
Notes: upacks bytes in labels i
Also adds label 10 for the blank image in dataset.
"""
def unzip_labels(file_name):
	labels = None
	with gzip.open(file_name,'rb') as file_content:
		file_content.seek(4)
		num_labels = struct.unpack('>L',file_content.read(4))[0]
		total_size = num_labels
		labels = np.frombuffer(file_content.read(total_size),dtype = np.uint8)
		empty_labels = np.zeros((total_size//10),dtype = np.uint8)
		empty_labels.fill(10)
		labels = np.concatenate((labels,empty_labels))
	return labels
	
"""
class " Pipeline "
Notes: creates a pipeline to get batches of images and labels
"""
class Pipeline(object):
	
	def __init__(self,data,label,batch_size=110):
		self.__num_images = data.shape[0]
		self.__data,self.__label = self.__shuffle(data,label,self.__num_images)
		self.__batch_size = batch_size
		self.__max_iter = self.__num_images//self.__batch_size
		self.__classes = 11
		self.__curr_iter = 0
		self.__one_hot_encoding()
		return	
		
	def __shuffle(self,data,label,num_images):
		shuffle_index = list(range(num_images))
		np.random.shuffle(shuffle_index)	
		return data[shuffle_index], label[shuffle_index]
		
	def __one_hot_encoding(self):
		num_label = self.__label.shape[0]
		one_hot_label = np.zeros((num_label,self.__classes))
		one_hot_label[np.arange(num_label),self.__label] = 1
		self.__label = one_hot_label
	
	def get_batches(self):
		if self.__curr_iter+self.__batch_size >= self.__num_images:
			self.__data,self.__label = self.__shuffle(self.__data,self.__label,self.__num_images)
			self.__curr_iter = 0
		indicies = list(range(self.__curr_iter,self.__curr_iter+self.__batch_size))
		self.__curr_iter += self.__batch_size
		return self.__data[indicies], self.__label[indicies]
		
	def get_dataset(self,size=None):
		if size == None:
			return self.__data, self.__label
		return self.__data[:size,:,:,:], self.__label[:size,:]
"""
function " model "
input parameters: X
output parameters: logits
Notes: normalize input image of shape 28,28
standard layer distribution
outputs logit preditions needs further softmax to get preditions
"""		
def model(X):
	X = tf.divide(X, 255)
	#~ X = tf.reshape(X,[-1, 28, 28, 1])
	conv1 = tf.layers.conv2d(inputs= X,
		filters= 32,
		kernel_size= [5, 5],
		strides= [1, 1],
		padding= 'same',
		activation= tf.nn.relu,
		kernel_initializer= tf.truncated_normal_initializer(stddev= 0.1),
		bias_initializer= tf.constant_initializer(0.1),
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
		bias_initializer= tf.constant_initializer(0.1),
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
		bias_initializer= tf.constant_initializer(),
		name= 'fully_connected_layer1')
		
	fc2 = tf.layers.dense(inputs= fc1,
		units= 11,
		activation= None,
		kernel_initializer= tf.truncated_normal_initializer(stddev= 0.1),
		bias_initializer= tf.zeros_initializer(),
		name= 'fully_connected_layer2')
		
	logit_pred = fc2
	return logit_pred
	
	
def main():
		
	learning_rate = 0.001
	batch_size = 256
	epochs = 500
	
	train_images, train_labels, test_images, test_labels =  get_dataset()
	train_pipe = Pipeline(train_images,train_labels,batch_size)
	test_pipe = Pipeline(test_images,test_labels)
	
	
	with tf.name_scope('input'):
		X = tf.placeholder(tf.float32, [None,28,28,1], name= 'image')
		y_true = tf.placeholder(tf.float32, [None, 11], name= 'label')
	
	with tf.variable_scope('inference'):
		logit_pred = model(X)	
		y_pred = tf.nn.softmax(logit_pred, name= 'predict')
		
	tf.add_to_collection('logit_predict', logit_pred)
	
	with tf.name_scope('loss'):
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits= logit_pred, labels=y_true)
		loss = tf.reduce_mean(cross_entropy)
	with tf.name_scope('evaluation'):
		correct_count = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_true,1))
		accuracy = tf.reduce_mean(tf.cast(correct_count, tf.float32)) 
	tf.summary.scalar('Accuracy', accuracy)
	
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	
	merged = tf.summary.merge_all()
	init_op = tf.global_variables_initializer()
	saver = tf.train.Saver()
	
	with tf.Session() as sess:
		
		writer = tf.summary.FileWriter('output', sess.graph)
		sess.run(init_op)
		
		for i in range(epochs):
			train_X, train_y = train_pipe.get_batches()
			train_data = {X: train_X, y_true: train_y }
			summary,_ = sess.run([merged, optimizer], feed_dict = train_data)
			writer.add_summary(summary, i)
			a = sess.run([accuracy], feed_dict = train_data)
			if (i+1)%(epochs//10) == 0:
				print('Train accuracy= {}'.format(a))
		test_X, test_y = test_pipe.get_dataset(5000)
		test_data = {X: test_X, y_true: test_y}
		a, c = sess.run([accuracy, loss], feed_dict = test_data)
		print('Test accuracy= {}'.format(a))
		writer.close()
		saver.save(sess,'training/model')
	return
	
if __name__ == '__main__':
	main()
