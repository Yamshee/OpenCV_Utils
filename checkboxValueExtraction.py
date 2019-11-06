import random
import smtplib
import json
import urllib
import pandas as pd
import cv2
import os
from functools import reduce
import datetime
import openpyxl as px
import numpy as np
from string import punctuation
from PIL import Image
import requests
import pytesseract


pytesseract.pytesseract.tesseract_cmd ="C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"
class UnionFind:
    """Union-find data structure. Items must be hashable."""

    def __init__(self):
        """Create a new empty union-find structure."""
        self.weights = {}
        self.parents = {}

    def __getitem__(self, obj):
        """X[item] will return the token object of the set which contains `item`"""

        # check for previously unknown object
        if obj not in self.parents:
            self.parents[obj] = obj
            self.weights[obj] = 1
            return obj

        # find path of objects leading to the root
        path = [obj]
        root = self.parents[obj]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def union(self, obj1, obj2):
        """Merges sets containing obj1 and obj2."""
        roots = [self[obj1], self[obj2]]
        heavier = max([(self.weights[r],r) for r in roots])[1]
        for r in roots:
            if r != heavier:
                self.weights[heavier] += self.weights[r]
                self.parents[r] = heavier

def groupTPL(TPL, distance=20):
    U = UnionFind()

    for (i, x) in enumerate(TPL):
        #print(i, x)
        for j in range(i + 1, len(TPL)):
            y = TPL[j]
            #both x and y coordinates
            #  if max(abs(x[0] - y[0]), abs(x[1] - y[1])) <= distance:
            #print(x[0] , y[0])
            if abs(x[0] - y[0])<= distance and abs(x[1] - y[1])<= distance and ( abs(x[2] - y[2])<= distance or abs(x[3] - y[3])<= distance):
                #print('True')
                U.union(x, y)

    disjSets = {}
    for x in TPL:
        s = disjSets.get(U[x], set())
        s.add(x)
        disjSets[U[x]] = s

    return [list(x) for x in disjSets.values()]


def tess_extract(crop_img):
    text = pytesseract.image_to_string(crop_img, config='-c preserve_interword_spaces=1 --psm 6')
    #print(text)
    return(text)


def sort_contours(cnts, method="top-to-bottom"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)



def checkboxValueExtraction_Improved(img):
    checkboxlist = []
    # img = cv2.imread(r'/mnt/c/Users/yamshee.fatima/Downloads/image/sample1-a.jpg')
    # cv2.imwrite("checkboxes.jpg", CheckboxImgCrop)
    #img = cv2.imread(r'D:\checkboxes.jpg',0)

    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    tuplelist = []
    list1 = []
    print(len(contours))
    #a=groupTPL(list(boundingBoxes),distance=5)
    #print(a)
    c=[]
    for i in range(0, len(contours)):
        cnt = contours[i]
        epsilon = 0.02*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        size = cv2.contourArea(approx)

        extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
        extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
        distance = np.sqrt((extLeft[0] - extRight[0])**2 + (extLeft[1] - extRight[1])**2)

        if (7000 > size > 320) and len(approx)%2 == 0 and cv2.isContourConvex(approx):
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            # a square will have an aspect ratio that is approximately equal to one, otherwise, the shape is a rectangle
            if ar >= 0.95 and ar <= 2.00:
                c.append(cnt)

    print(len(c))
    contours,boundingBoxes = sort_contours(c, method="left-to-right")
    a=groupTPL(list(boundingBoxes), distance=20)
    #print(a)

    aa=[i[0] for i in a]
    print(len(aa))

    for i in range(0, len(c)):
        cnt = contours[i]
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)

        checkbox_img = img[y+8:y+h-8, x+8:x+w-8]

        #image = cv2.imread("detected-boxes"+str(i)+".jpg", 0)
        # get all non black Pixels
        cntNotBlack = cv2.countNonZero(checkbox_img)

        # get pixel count of image
        height, width = checkbox_img.shape
        cntPixels = height*width

        # compute all black pixels
        cntBlack = cntPixels - cntNotBlack

        ratio_black = np.sum(checkbox_img == 0) / checkbox_img.size
        #print('ratio_black: ', ratio_black)


        if cntBlack > 10:
                #print('Detected black Count: ', cntBlack)
                #cv2.imwrite("detected-boxes"+str(i)+".jpg", checkbox_img)
                tuplelist.append((x, y, w, h))
                checkbox_img_text = img[y-5:y+h+5, x+w:img.shape[1]]
                #cv2.imwrite('checkbox_img_text' + str(random.randint(1, 900)) + '.jpg', checkbox_img_text)
                #added later

                #print(type(checkbox_img_text))
                try:

                    extractedData = tess_extract(checkbox_img_text)
                    extractedData = extractedData.lstrip(punctuation).split('  ')[0]
                    #print('Check Box Data ', extractedData)
                    checkboxlist.append(extractedData.strip())
                except:
                    pass
    return list(set(checkboxlist))
