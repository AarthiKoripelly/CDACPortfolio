from keras.models import model_from_json
import numpy as np
import pandas as pd
import cv2 

import matplotlib.pyplot as plt

import os
import os.path
import random 


def LOG(image):
    image = np.array(image, dtype=np.uint8)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.2, 100, minRadius = 1, maxRadius = 100)
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
 
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            #draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(img, (x, y), r, (255, 255, 255), -1)
            cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            
    return img

#The following code reads the image and performs laplacian of gaussian computer vision algorithm. 
#The graph is also created as a part of the function. 
from PIL import Image

def check_shape(image):
    print("checkpoint")
    image = cv2.imread(image)
    height, width, channels = image.shape 

    if (height*width > 330000000):
        return large_classify(image, 15)
    else:
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    
        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)
    
        image = image[y:y+h,x:x+w]
        return small_classify(image, 15)

def splice_image(image, x, y):
    test = image
    my_slice = test[x:y, x:y]
    return my_slice

#Pick random number of positions in image and detects blobs 
def large_classify(image, num):
    with open('model_hough.json', 'r') as f:
        model = model_from_json(f.read())
    print("checkpoint")
    # Load weights into the new model
    model.load_weights('model_hough.h5')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("checkpoint")
    predictions = []
    image = cv2.resize(image, dsize=(10000, 10000))
    #height, width, channels = image.shape
    placer = num - 1
    X = 100
    Y = 200
    #image = rgb2gray(image)
    for i in range(placer):
        im = splice_image(image, X, Y)
        im = LOG(im)
        im = cv2.resize(im, dsize=(128, 128))
        im = im.reshape(-1, 128,128,1)
        predictions.append(model.predict_classes(im))
        X = X+100
        Y = Y+100
    return predictions 

def small_classify(image, num):
    with open('model_hough.json', 'r') as f:
        model = model_from_json(f.read())
    print("checkpoint")
    # Load weights into the new model
    model.load_weights('model_hough.h5')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("checkpoint")
    predictions = []
    #height, width, channels = image.shape
    placer = num - 1
    X = 100
    Y = 500
    #image = rgb2gray(image)
    for i in range(placer):
        im = splice_image(image, X, Y)
        im = LOG(im)
        im = cv2.resize(im, dsize=(128, 128))
        im = im.reshape(-1, 128,128,1)
        predictions.append(model.predict_classes(im))
        X = X+400
        Y = Y+400
    return predictions 

def analyze_image(image):
    res = check_shape(image)
    prob = 0
    total = 0
    for item in res:
        if(item==1):
            prob+=1
        total+=1
    return(prob/total)