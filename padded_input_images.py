import numpy as np
import re
import math
import cv2
from os.path import isfile, join
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import sys
from args import get_arguments
arg = get_arguments()

def pad_and_stack_image(path_train ):
    scaler = StandardScaler()
    
    # sorts the file in a list
    numbers = re.compile(r'(\d+)')
    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts 

    files_train = [f for f in os.listdir(path_train) if isfile(join(path_train, f))]
    if '.DS_Store' in files_train:
        files_train.remove('.DS_Store')
    files_train = sorted(files_train, key=numericalSort)  
      
    image_list_train =[]
    mlcs_images =[]
    
    window = 11 # we tried different window sizes, 7x7, 11x11, 21x21. according to our experiments 11x11 is the optimal one
    for i in range(len(files_train)):
        #reading the file
        image_bgr  = cv2.imread(join(path_train, files_train[i]),1)
        #image_yuv = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2YUV)
        height = image_bgr.shape[0]
        width = image_bgr.shape[1]
        if image_bgr.dtype == 'uint16':
            divider = 65535
        else:
            divider = 255
        
        image_bgr  = image_bgr / divider
        image_bgr = image_bgr.astype(np.float32) 
        ch_b, ch_g, ch_r = cv2.split(image_bgr)
        image_list_train.append(ch_b)
        image_list_train.append(ch_g)
        image_list_train.append(ch_r)
        
    # Applying PCA    
    if not (0 < arg.pca < len(files_train)):
        raise ValueError("PCA component ({0}) must be between 0 and number of training images ({1})"
                     .format(arg.pca, len(files_train)))
        
    else:
        image_stack_yTrain  = np.array(image_list_train).astype(np.float32)
        image_stack_yTrain  = image_stack_yTrain.flatten().reshape(image_stack_yTrain.shape[0],-1).T
        
        scaler.fit(image_stack_yTrain)  
        image_stack_yTrain  =  scaler.transform(image_stack_yTrain)  
        pca = PCA(n_components = arg.pca)
        image_stack_yTrain= pca.fit_transform(image_stack_yTrain)
        
        
        # padding each image for creating a patch of window size centered at each pixel 
        for i in range(0, image_stack_yTrain.shape[1]):
            img_pca = image_stack_yTrain[:,i].reshape(height, width, order= 'C')
            mlcs_images.append(np.pad(img_pca, window//2, mode = 'constant'))
        patch_stack_train = np.array(mlcs_images)
    
    return (patch_stack_train, window, height,width)   
