import cv2
import copy
import numpy as np

def sum_sort(pt1, pt2):
	d1 = pt1[0] + pt1[1]
	d2 = pt2[0] + pt2[1]
	if d1 > d2:
		return 1
	elif d2 > d1:
		return -1
	else:
		return 0

def l2_dist(pt1, pt2):
	return np.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2))

def main():		
	sudoku_im = cv2.imread('sudoku-original.jpg', cv2.IMREAD_GRAYSCALE)
	rows, cols = sudoku_im.shape
	transformed_sudoku_im = cv2.GaussianBlur(sudoku_im, (5, 5), 0)
	transformed_sudoku_im = cv2.adaptiveThreshold(transformed_sudoku_im, 255,
												  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
												  cv2.THRESH_BINARY_INV, 15, 2)
	connectivity = 4
	output = cv2.connectedComponentsWithStats(transformed_sudoku_im,
											connectivity, cv2.CV_32S)
	labels_im = output[1]
	stats = output[2]
	stats_area = stats[:, cv2.CC_STAT_AREA]
	largest_label = np.argmax(stats_area[1:])+1
	b_box_x, b_box_y, b_box_w, b_box_h = stats[largest_label, :4]
	transformed_sudoku_im[labels_im == largest_label] = 255
	transformed_sudoku_im[labels_im != largest_label] = 0
	
	positions = np.where(transformed_sudoku_im == 255)
	positions = np.asarray(positions).T
	positions[:,0], positions[:,1] = positions[:,1], positions[:,0].copy()
	positions = positions.tolist()
	width_adj = lambda pos: cols - 1 - pos[0]
			
	tl_positions = positions
	tr_positions = [[width_adj(pos), pos[1]] for pos in tl_positions]
	
	tl_positions.sort(sum_sort)
	tr_positions.sort(sum_sort)
	
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
	sudoku_ext_im = cv2.warpPerspective(sudoku_im, transform_mat, (max_height,max_width))
	sudoku_bin_im = cv2.adaptiveThreshold(sudoku_ext_im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)
	
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(max_width/9,1))
	horizontal = cv2.erode(sudoku_bin_im,kernel,iterations = 1)
	horizontal = cv2.dilate(horizontal,kernel,iterations = 2)
	
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(1,max_height/9))
	vertical = cv2.erode(sudoku_bin_im,kernel,iterations = 1)
	vertical = cv2.dilate(vertical,kernel,iterations = 2)
	
	intersection_lines = cv2.bitwise_or(vertical, horizontal)
	numbers = sudoku_bin_im - intersection_lines
	numbers = cv2.medianBlur(numbers,5)
	
	cell_height = max_height/9
	cell_width = max_width/9
	
	for i in range(9):
		#~ print(i*cell_height)
		for j in range(9):
			cell = sudoku_ext_im[i*cell_width:(i+1)*cell_width, j*cell_height:(j+1)*cell_height]
			cv2.rectangle(numbers, (j*cell_height,i*cell_width), ((j+1)*cell_height,(i+1)*cell_width), 255, 2)
			#~ cv2.imshow('asdf', cell)
			#~ cv2.waitKey(0)
			
	#~ cv2.polylines(sudoku_im,[pts],True,255)
	cv2.imshow('sudoku', numbers )
	#~ cv2.imshow('sudoku1', numbers )
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return

if __name__ == '__main__':
	main()
