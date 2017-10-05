from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import copy
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def predict(image):
	y_logit = tf.placeholder(tf.float32, [None,11])
	y_pred = tf.nn.softmax(y_logit)
	with tf.Session() as sess:
		model = tf.train.import_meta_graph('training/model.meta')
		model.restore(sess, tf.train.latest_checkpoint('training/./'))
		graph = tf.get_default_graph()
		logit_pred = tf.get_collection('logit_predict')
		X = graph.get_tensor_by_name('input/image:0')
		image = np.reshape(image, (-1,28,28,1))
		feed_dict = {X: image}
		scores = sess.run(logit_pred, feed_dict)
		result = sess.run(y_pred, feed_dict= {y_logit: scores[0]})
		number_pred = np.argmax(result, axis= 1)
	return number_pred

def clean_image(cell):
	#~ mask_im = np.zeros(cell.shape, dtype = 'uint8')
	out_im = np.zeros((28,28), dtype = 'uint8')
	_, contours, hierarchy = cv2.findContours(cell.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	if len(contours) == 0:
		return None
	largest_contour = max(contours, key= cv2.contourArea)
	row, col = cell.shape
	if ( cv2.contourArea(largest_contour) < .04 *row * col ):
		return None
	x,y,w,h = cv2.boundingRect(largest_contour)
	mask_im = cell[y:y+h,x:x+w]
	if h > 28 or w > 28:
		return cv2.resize(mask_im, (28, 28))
	else:
		out_im[14-h//2:14-h//2+h, 14-w//2:14-w//2+w] = mask_im
		return out_im

def l2_dist(pt1, pt2):
	return np.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2))

def pipeline(in_image):
	rows, cols = in_image.shape
	filtered_im = cv2.GaussianBlur(in_image, (5, 5), 0)
	binarized_im = cv2.adaptiveThreshold(filtered_im,255,
	cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,3)
								
	connectivity = 4
	cc_info = cv2.connectedComponentsWithStats(binarized_im,
											connectivity, cv2.CV_32S)
	labels_im = cc_info[1]
	stats = cc_info[2]
	stats_area = stats[:, cv2.CC_STAT_AREA]
	# Label of largest area excluding the background
	largest_label = np.argmax(stats_area[1:])+1
	# Extract connected component with largest area
	binarized_im[labels_im == largest_label] = 255
	binarized_im[labels_im != largest_label] = 0
	
	
	positions = np.where(binarized_im == 255)
	positions = np.asarray(positions).T
	positions[:,0], positions[:,1] = positions[:,1], positions[:,0].copy()
	positions = positions.tolist()
	width_adj = lambda pos: cols - 1 - pos[0]
			
	tl_positions = positions
	tr_positions = [[width_adj(pos), pos[1]] for pos in tl_positions]
	
	tr_positions.sort(key= lambda pt: pt[0]+pt[1])
	tl_positions.sort(key= lambda pt: pt[0]+pt[1])
	
	tr = [width_adj(tr_positions[0]), tr_positions[0][1]]
	bl = [width_adj(tr_positions[-1]), tr_positions[-1][1]]
	tl = tl_positions[0]
	br = tl_positions[-1]

	rect = np.array([tl, tr, br, bl], dtype = "float32")
	pts = rect.astype(int)
	width1 = l2_dist(br, bl)
	width2 = l2_dist(tr, tl)
	height1 = l2_dist(br, tr)
	height2 = l2_dist(tl, bl)
	max_width = max(int(width1), int(width2))
	max_height = max(int(height1), int(height2))
	dst = np.array([[0, 0],[max_height - 1, 0],[max_height - 1, max_width - 1], [0, max_width - 1]], dtype = "float32")
	
	transform_mat = cv2.getPerspectiveTransform(rect, dst)
	sudoku_ext_im = cv2.warpPerspective(in_image, transform_mat, (max_height,max_width))
	sudoku_bin_im = cv2.adaptiveThreshold(sudoku_ext_im, 255,
	cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,3)
	
	cv2.imshow('sudoku', sudoku_bin_im )
	cv2.waitKey(0)
	
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(int(max_width/9),1))
	horizontal = cv2.erode(sudoku_bin_im,kernel,iterations = 1)
	horizontal = cv2.dilate(horizontal,kernel,iterations = 2)
	
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(1,int(max_height/9)))
	vertical = cv2.erode(sudoku_bin_im,kernel,iterations = 1)
	vertical = cv2.dilate(vertical,kernel,iterations = 2)
	
	intersection_lines = cv2.bitwise_or(vertical, horizontal)
	numbers_im = sudoku_bin_im - intersection_lines
	numbers_im = cv2.medianBlur(numbers_im,3)
	
	cell_height = int(max_height/9)
	cell_width = int(max_width/9)
	
	return numbers_im,cell_height,cell_width

def main():		
	sudoku_im = cv2.imread('sudoku-0.jpg', cv2.IMREAD_GRAYSCALE)
	rows, cols = sudoku_im.shape
	
	numbers_im,cell_height,cell_width = pipeline(sudoku_im)
		
	numbers = []
	cell_resize = []
	
	
	cv2.destroyAllWindows()
	return

def test():
	image = cv2.imread('t.png', cv2.IMREAD_GRAYSCALE)
	image = np.reshape(image, (-1,28,28,1))
	print(predict(image))

if __name__ == '__main__':
	main()
	#~ test()
