import cv2
import numpy as np
import json

def resizeAndShow(img, size, name):
    img = cv2.resize(img, size)
    cv2.imshow(name, img)

def findAndDrawContours(mask, img):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    data = {
        'holds' : []
    }

    (h, w) = img.shape[:2]

    for n, cnt in enumerate(contours[1:]):
        r, c = drawBoundingRing(cnt, img)
        boundingPoly = drawBoundingPoly(cnt, img)

        path = list(map(lambda p : [p[0, 0] / w, p[0, 1] / w], boundingPoly))
        print(path)
        
        hold = {
            'x'     : c[0] / w,
            'y'     : c[1] / h,
            'size'  : r / w,
            'path'  : path
        }

        data['holds'].append(hold)

    #img = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    with open('data.json', 'w') as outfile:
        json.dump(data, outfile, indent=4)

def drawBoundingBox(contour, img):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img,[box],0,(0,0,255),2)

def drawBoundingRing(contour, img):
    (cX, cY), cR = cv2.minEnclosingCircle(contour)
    radius = int(cR)
    center = (int(cX), int(cY))
    #cv2.circle(img, center, 7, (255, 255, 255), -1) 
    #cv2.circle(img, center, radius, (255, 255, 255), 2)
    return cR, (cX, cY)

def drawBoundingPoly(contour, img):
    epsilon = 0.01*cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,epsilon,True)
    img = cv2.drawContours(img, [approx], -1, (255, 0, 0), 3)
    return approx
    
im = cv2.imread('./misc/Images/BoardMasked.png')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY_INV)

kernel = np.ones((5,5),np.uint8)
close_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

#contours, hierarchy = cv2.findContours(close_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#im = cv2.drawContours(im, contours, -1, (0, 255, 0), 3)
findAndDrawContours(close_mask, im)

(h, w) = im.shape[:2]
aspect = w/h
ns = (int(800*aspect), 800)

resizeAndShow(close_mask, ns, 'col')
resizeAndShow(im, ns, 'image')

cv2.waitKey(0)
cv2.destroyAllWindows()