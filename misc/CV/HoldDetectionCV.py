import cv2
import numpy as np

im = cv2.imread('./Images/BoardCutout.png')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

lower_color_b = (10, 0, 0)
upper_color_b = (255, 50, 50)

lower_color_g = (0, 10, 0)
upper_color_g = (50, 255, 50)

ret,thresh = cv2.threshold(im, 80, 255, 0)

mask_b = cv2.inRange(im, lower_color_b, upper_color_b)
mask_g = cv2.inRange(im, lower_color_g, upper_color_g)

mask = mask_b | mask_g
maskbgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

th3 = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)

kernel = np.ones((8,8),np.uint8)
closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)

contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for n, cnt in enumerate(contours):
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    im = cv2.drawContours(im,[box],0,(0,0,255),2)

im = cv2.drawContours(im, contours, -1, (0, 255, 0), 3)

cv2.imshow('image', im)
cv2.imshow('thresh', thresh)
cv2.imshow('image2', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()