from __future__ import print_function
import numpy as np
import pandas as pd
import cv2 
import scipy

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

import os
import os.path
import random 

from math import sqrt
from skimage import data
from skimage.feature import blob_log
from skimage.color import rgb2gray


#The following code reads the image and performs laplacian of gaussian computer vision algorithm. 
#The graph is also created as a part of the function. 
from PIL import Image

def check_shape(image):
    image = cv2.imread(image)
    height, width, channels = image.shape 
    if (height*width > 330000000):
        return find_blobs_out(image, 7)
    else:
        
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    
        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)
        
        image = image[y:y+h,x:x+w]
        
        return find_blobs_focus(image, 7)

def splice_image(image, x, y):
    test = image
    my_slice = test[x:y, x:y]
    return my_slice

#Pick random number of positions in image and detects blobs 
def find_blobs_out(image, num):
    array = []
    #image = cv2.imread(image)
    height, width, channels = image.shape
    if(width>15000):
        image = cv2.resize(image, dsize=(10000, 10000), interpolation=cv2.INTER_LINEAR)
    placer = num - 1
    X = width/300
    Xfirst = 0
    Y = X+height/300
    Yfirst = X
    for i in range(placer):
        x_pick = int(random.uniform(Xfirst, X))
        y_pick = int(random.uniform(Yfirst, Y))
        array.append(LOGandDOG(splice_image(image, x_pick, y_pick)))
        Xfirst = X
        Yfirst = Y
        X = X + width/300
        Y = X+height/300
    return array

def find_blobs_focus(image, num):
    array = []
    #image = cv2.imread(image)
    height, width, channels = image.shape
    placer = num - 1
    X = width/50
    Xfirst = 0
    Y = X+height/50
    Yfirst = X
    for i in range(placer):
        x_pick = int(random.uniform(Xfirst, X))
        y_pick = int(random.uniform(Yfirst, Y))
        array.append(LOGandDOG(splice_image(image, x_pick, y_pick)))
        Xfirst = X
        Yfirst = Y
        X = X + width/50
        Y = X+height/50
    return array

def LOGandDOG(image): 
    #Reading image
    #image = cv2.imread(image)
    
    blobsLOG = blob_log(rgb2gray(image), max_sigma=30, num_sigma=10, threshold=.1)

    # Computing radii
    blobsLOG[:, 2] = blobsLOG[:, 2] * sqrt(2)


    #Setting plot
    blobList = [blobsLOG]
    colors = ['yellow']
    titles = ['Laplacian of Gaussian']
    sequence = zip(blobList, colors, titles)

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharex=True, sharey=True)

    ax = axes.ravel()
    
    #print("Loading image...")
    countofCircles = 0
    
    for index, (blobs, color, title) in enumerate(sequence):
        ax[index].set_title(title)
        ax[index].imshow(image, interpolation='nearest')
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            countofCircles=countofCircles+1
            ax[index].add_patch(c)
        ax[index].set_axis_off()
        
    #plt.show()
    #print('This is the count for the number of Laplacian blobs: ', countofCircles)
    return countofCircles 




