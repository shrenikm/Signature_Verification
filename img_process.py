import numpy as np 
import cv2

img = cv2.imread("testing2_white.jpg", 0)

cv2.namedWindow("image", cv2.WINDOW_NORMAL)

rows, cols = img.shape

ret, imgth = cv2.threshold(img, 135, 255, cv2.THRESH_BINARY)


imgth_resized = cv2.resize(imgth, (50, 50)) 

cv2.imwrite("negative_example1.jpg", imgth_resized)

cv2.imshow("image", imgth)



cv2.waitKey(0)


cv2.destroyAllWindows()