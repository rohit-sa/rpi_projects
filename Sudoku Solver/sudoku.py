import cv2
import numpy as np

sudoku_im = cv2.imread('sudoku-original.jpg', cv2.IMREAD_GRAYSCALE)
height, width = sudoku_im.shape
transformed_sudoku_im = cv2.GaussianBlur(sudoku_im, (5, 5), 0)
transformed_sudoku_im = cv2.adaptiveThreshold(transformed_sudoku_im, 255,
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, 15, 2)
#~ transformed_sudoku_im = cv2.medianBlur(transformed_sudoku_im, 3)
kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
#~ kernel = np.ones((3, 3), np.uint8)
transformed_sudoku_im = cv2.dilate(transformed_sudoku_im, kernel)
connectivity = 4
output = cv2.connectedComponentsWithStats(transformed_sudoku_im,
										connectivity, cv2.CV_32S)
labels_im = output[1]
stats = output[2]
stats_area = stats[:, cv2.CC_STAT_AREA]
largest_label = np.argmax(stats_area[1:])+1
b_box_x, b_box_y, b_box_w, b_box_h = stats[largest_label, :4]
cv2.rectangle(sudoku_im,(b_box_x, b_box_y),
				(b_box_x + b_box_w, b_box_y + b_box_h),  255, 2)
#~ crop_sudoku_im = sudoku_im[b_box_y: b_box_y + b_box_h,
						#~ b_box_x: b_box_x + b_box_w]
transformed_sudoku_im[labels_im == largest_label] = 255
transformed_sudoku_im[labels_im != largest_label] = 0
#~ dst = cv2.cornerHarris(transformed_sudoku_im,4,3,0.04)
#~ closing = cv2.morphologyEx(transformed_sudoku_im, cv2.MORPH_CLOSE, kernel)
#~ dst = cv2.dilate(dst,None)
#~ sudoku_im[dst > 0.1*dst.max()] = 255
#~ minLineLength = 100
#~ maxLineGap = 10
#~ lines = cv2.HoughLinesP(transformed_sudoku_im,1,np.pi/180,100,minLineLength,maxLineGap)
#~ print(lines[:,0])
#~ for x1,y1,x2,y2 in lines[:,0]:
	#~ cv2.line(sudoku_im,(x1,y1),(x2,y2),255,2)
#~ im2, contours, hierarchy = cv2.findContours(transformed_sudoku_im, 
												#~ cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow('sudoku', sudoku_im )
cv2.waitKey(0)
cv2.destroyAllWindows()
