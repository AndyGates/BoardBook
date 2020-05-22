import cv2
import numpy as np

def getColorThreshold(img):
    lower_color_b = (10, 0, 0)
    upper_color_b = (255, 50, 50)

    lower_color_g = (0, 10, 0)
    upper_color_g = (50, 255, 50)

    mask_b = cv2.inRange(img, lower_color_b, upper_color_b)
    mask_g = cv2.inRange(img, lower_color_g, upper_color_g)

    mask = mask_b | mask_g
    maskbgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return maskbgr

def resizeAndShow(img, size, name):
    img = cv2.resize(img, size)
    cv2.imshow(name, img)

def findAndDrawContours(img):
    retimg = img.copy()

    contours, hierarchy = cv2.findContours(retimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#    for n, cnt in enumerate(contours):
#        rect = cv2.minAreaRect(cnt)
#        box = cv2.boxPoints(rect)
#        box = np.int0(box)
#        retimg = cv2.drawContours(retimg,[box],0,(0,0,255),2)

    retimg = cv2.drawContours(retimg, contours, -1, (0, 255, 0), 3)
    return retimg



im = cv2.imread('./misc/Images/BoardMasked.png')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

img_b = im[:,:,0]
imgray = cv2.GaussianBlur(imgray, (15, 15), 0)


#ret, thresh = cv2.threshold(img_b,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


#thresh = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, 5)
#edged=cv2.Canny(thresh,50,200)
#kernel = np.ones((3,3),np.uint8)

morph = im.copy()
for r in range(1, 4):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mgrad = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)

b, g, r = cv2.split(morph)

_, b = cv2.threshold(b, 20, 200, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
_, g = cv2.threshold(g, 20, 250, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
_, r = cv2.threshold(r, 60, 200, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

merged = cv2.merge([b,g,r])

#closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE, kernel, iterations = 2)
#sure_fg = cv2.erode(closing,kernel,iterations=2)

#opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
#sure_bg = cv2.dilate(opening,kernel,iterations=2)

#md_lim = 0.12
#dt = cv2.distanceTransform(sure_bg,cv2.DIST_L2,5)
#dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
#dt = cv2.normalize(dt, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
#ret, sure_fg = cv2.threshold(dt,int(float(md_lim)/(100.0)*dt.max()),255,0)

#sure_fg = np.uint8(sure_fg)
#unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
#ret, labels = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
#labels = labels+1

#labels[unknown==255] = 0

#labels = cv2.watershed(im,labels)
#im[labels == -1] = [255,0,0]

#kernel = np.ones((3,3),np.uint8)
#closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#closing = cv2.GaussianBlur(closing, (3, 3), 0)

#cntimg = findAndDrawContours(imgray)

mggray = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(mggray,180,255,cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
im = cv2.drawContours(im, contours, -1, (0, 255, 0), 3)

(h, w) = im.shape[:2]
aspect = w/h
ns = (int(600*aspect), 600)

resizeAndShow(thresh, ns, 'col')
resizeAndShow(im, ns, 'image')

cv2.waitKey(0)
cv2.destroyAllWindows()