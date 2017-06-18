from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import cv2

tf.logging.set_verbosity(tf.logging.INFO)

def main():
	Y = tf.placeholder(tf.float32, [1,10])
	prediction = tf.nn.softmax(Y)
	with tf.Session() as sess:
		model = tf.train.import_meta_graph('model.meta')
		model.restore(sess, tf.train.latest_checkpoint('./'))
		graph = tf.get_default_graph()
		#~ predict = graph.get_operation_by_name('inference/predict')
		logit_predict = tf.get_collection('logit_predict')
		X = graph.get_tensor_by_name('input/in_image:0')
		image = cv2.imread('5.png', cv2.IMREAD_GRAYSCALE)
		image_flat = np.reshape(image, (-1,784))
		feed_dict = {X: image_flat}
		scores = sess.run(logit_predict, feed_dict)
		predict = sess.run(prediction, feed_dict= {Y: scores[0]})
		#~ print(np.argmax(scores, axis= 1))
		#~ for op in tf.get_default_graph().get_operations():
			#~ print(op.name )
		print(scores[0])
		print(np.argmax(predict, axis= 1))
	return


if __name__ == '__main__':
	main()
