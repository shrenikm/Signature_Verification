import numpy as np 
import cv2

img = cv2.imread("paper_2_ninv.jpg", 0)

cv2.namedWindow("image", cv2.WINDOW_NORMAL)

rows, cols = img.shape

# ret, imgth = cv2.threshold(img, 135, 255, cv2.THRESH_BINARY)

img = cv2.bitwise_not(img)


img_resized = cv2.resize(img, (50, 50)) 

cv2.imwrite("paper_2_final.jpg", img_resized)

cv2.imshow("image", img_resized)



cv2.waitKey(0)


cv2.destroyAllWindows()