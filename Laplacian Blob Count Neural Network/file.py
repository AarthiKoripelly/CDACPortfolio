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

from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler


from testing import check_shape 

def analyze_image(image):
    # Model reconstruction from JSON file
    with open('model.json', 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights('model.h5')
    
    data = check_shape(image)
    x = np.array(data).reshape((-1,1))
    
    sc = StandardScaler()
    x = sc.fit_transform(x)
    
    x = np.array(data).reshape((1,6))
    
    prediction = model.predict(x)
    
    if (prediction <1): 
        return False 
    else:
        return True
