import numpy as np 
import cv2

im = cv2.imread("positive_example1.jpg", 0)

cv2.namedWindow("image", cv2.WINDOW_NORMAL)

fil = open("positive_example1.dat", "a")

rows, cols = im.shape

print rows, cols

im = im.flatten()

for i in xrange(0, (rows*cols)):
	fil.write(str(im[i])+ " ")
fil.write("\n")


fil.close()

cv2.waitKey(0)

cv2.destroyAllWindows()