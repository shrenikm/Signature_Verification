import numpy as np 
import cv2 

img_name = "nnex2.jpg"
dat_name = img_name[:-3]+"dat"

im = cv2.imread(img_name, 0)

rows, cols = im.shape

print rows, cols

fil = open(dat_name, "a")

imr = cv2.resize(im, (50, 50))


cv2.imwrite(img_name, imr)

imr = imr.flatten()

# Saving to file

for i in xrange(2500):
	fil.write(str(imr[i])+" ")
fil.write("\n")

fil.close()

cv2.waitKey(0)

cv2.destroyAllWindows()




