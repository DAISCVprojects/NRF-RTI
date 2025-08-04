import skimage 
from sklearn.decomposition import PCA
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import math
import torch
import cv2
from os.path import isfile, join
from sklearn.preprocessing import StandardScaler
import h5py
import os
import sys
from args import get_arguments
arg = get_arguments()

# default is HSH for embedding (expnding) light direction to high dimensional vector
def expand_light(light):
    light_list = []
    for i in range(len(light )):
        phi, theta = light[i]
        # harmonic base functions
        # order 1
        H1 = 1/np.sqrt(2*np.pi)
        # order 2
        H2 = np.sqrt(6/np.pi) * (np.cos(phi)*np.sqrt(np.cos(theta)- np.cos(theta)**2))
        H3 = np.sqrt(3/(2*np.pi)) * (-1 + 2 * np.cos(theta))
        H4 = np.sqrt(6/np.pi) * (np.sin(phi)*np.sqrt(np.cos(theta)-np.cos(theta)**2))
        # order 3
        H5 = np.sqrt(30/np.pi)*(np.cos(2*phi)*(-np.cos(theta) + np.cos(theta)**2))
        H6 = np.sqrt(30/np.pi)*(np.cos(phi)*(-1 + 2 * np.cos(theta)) * np.sqrt(np.cos(theta)-np.cos(theta)**2))
        H7 = np.sqrt(5/(2*np.pi))* (1-6*np.cos(theta) + 6 * np.cos(theta)**2)
        H8 = np.sqrt(30/np.pi)*(np.sin(phi)*(-1 + 2 * np.cos(theta)) * np.sqrt(np.cos(theta)-np.cos(theta)**2))
        H9 = np.sqrt(30/np.pi)*((-np.cos(theta) + np.cos(theta)**2)*np.sin(2*phi))

        light_list.append([H1, H2, H3, H4, H5, H6, H7, H8, H9])
    light= np.stack(light_list, axis = 0).astype(np.float32)
    
    return light

# Fourier encoding for pixel coordinates
def expand_coordinate(mgrid):
    np.random.seed(seed=42)
    mean = (0, 0)
    cov = [[0.3, 0], [0, 0.3]]
    sample = np.random.multivariate_normal(mean, cov, size = 10).astype(np.float32)
    
    coord_temp = []
    for i in range(0, mgrid.shape[0]):
        #light_temp.append
        dot_prod = np.matmul(sample,mgrid[i,:].reshape(2,1))* 2* np.pi
        _sin = np.sin(dot_prod )
        _cos = np.cos(dot_prod)
        coord_temp.append(np.append(_sin, _cos))
    mgrid = np.array(coord_temp).astype(np.float32)
    return mgrid


def csv_to_h5py(dataframe, file_name, data_save_dir):
    df_f = dataframe
    num_lines = len(df_f)
    num_features = len(df_f.columns)
 
    with h5py.File(data_save_dir + file_name, 'w') as h5f:
        # use num_features-1 if the csv file has a column header
        dset1 = h5f.create_dataset('features',
                               shape=(num_lines, num_features-3),
                               compression=None,
                               dtype='float32')
        dset2 = h5f.create_dataset('labels',
                               shape=(num_lines,3),
                               compression=None,
                               dtype='float32')

        features = df_f.values[:, 0:num_features-3]
        labels = df_f.iloc[:, -3:].values

        dset1[:, :] = features
        dset2[:,:] = labels
        


def processTrainData(path_train, path_train_light,  path_test, path_test_light, data_save_dir):
    scaler = StandardScaler()
    image_list_train =[]
    image_list_test =[]
    
    # sorts the file in a list
    numbers = re.compile(r'(\d+)')
    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts
    
    # reading and sorting training dataset file names
    files_train = [f for f in os.listdir(path_train) if isfile(join(path_train, f))]
    if '.DS_Store' in files_train:
        files_train.remove('.DS_Store')
    files_train = sorted(files_train, key=numericalSort)  
      
    # reading and sorting testing dataset file names
    files_test = [f for f in os.listdir(path_test) if isfile(join(path_test, f))]
    if '.DS_Store' in files_test:
        files_test.remove('.DS_Store')
    files_test  = sorted(files_test , key=numericalSort)
    
    def to_spherical(x, y, z):
        # Converts a cartesian coordinate (x, y, z) into a spherical one (radius, theta, phi).
        radius = np.linalg.norm(np.array([x, y, z]))
        # elevation/zenith
        theta = np.arctan2(z, np.sqrt(x * x + y * y))
        # azimuth
        phi = np.arctan2(y, x)
        return [phi, theta]
    
    # creating image coordinates
    im = skimage.io.imread(join(path_train, files_train[0]))
    h = im.shape[0]
    w = im.shape[1]
    dim = 2
    tensor1 = torch.linspace(-1, 1, steps=h)
    tensor2 = torch.linspace(-1, 1, steps=w)
    mgrid = torch.stack(torch.meshgrid(tensor1, tensor2), dim = -1)
    mgrid = mgrid.reshape(-1, dim)
    mgrid = mgrid.numpy().astype(np.float32)
    # comment this if you get model checkpoint weights dimension mismatch during teetsin, some objects were trained without coordinate expansion
    mgrid = expand_coordinate(mgrid)
    
    #  creating a grid of integer values for indexing purpose only during reshufling the data
    tensor1 = torch.linspace(0, h-1, steps=h)
    tensor2 = torch.linspace(0, w-1, steps=w)
    idx = torch.stack(torch.meshgrid(tensor1, tensor2), dim = -1)
    idx= idx.reshape(-1, dim)
    idx = idx.numpy().astype(np.float32) 
    
    # stacking index and coordinates
    mgrid = np.hstack((idx, mgrid))
   
    ch_b_list = []  
    ch_g_list = []
    ch_r_list = []
    # converting images fromRGB to YUV color space, train_set  
    for i in range(len(files_train)):
        #reading the file
        image_bgr = cv2.imread(join(path_train, files_train[i]),1)
        #image_yuv = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2YUV)
        if image_bgr.dtype == 'uint16':
            divider = 65535
        else:
            divider = 255
        
        image_bgr  = image_bgr / divider
        image_bgr = image_bgr.astype(np.float32) 
        ch_b, ch_g, ch_r = cv2.split(image_bgr)
        
        ch_b_list.append(ch_b.ravel())
        ch_g_list.append(ch_g.ravel())
        ch_r_list.append(ch_r.ravel()) 
        
    # stacking the u and v chennels of the images
    b = np.hstack(ch_b_list)
    g = np.hstack(ch_g_list)
    r = np.hstack(ch_r_list)

    image_list_train.append(b)
    image_list_train.append(g)
    image_list_train.append(r)
    del b,g,r
    
    image_list_train = np.stack(image_list_train, axis = -1)
    
    ch_b_list = []  
    ch_g_list = []
    ch_r_list = []
    for i in range(len(files_test)):
        image_bgr  = cv2.imread(join(path_test, files_test[i]),1)
        if image_bgr.dtype == 'uint16':
            divider = 65535
        else:
            divider = 255
        
        image_bgr  = image_bgr / divider
        image_bgr = image_bgr.astype(np.float32) 
        ch_b, ch_g, ch_r = cv2.split(image_bgr)
        
        ch_b_list.append(ch_b.ravel())
        ch_g_list.append(ch_g.ravel())
        ch_r_list.append(ch_r.ravel())
    b = np.hstack(ch_b_list)
    g = np.hstack(ch_g_list)
    r = np.hstack(ch_r_list)
    
    image_list_test.append(b)
    image_list_test.append(g)
    image_list_test.append(r)
    del b,g,r
    
    image_list_test = np.stack(image_list_test, axis = -1) 
    del   ch_b_list ,  ch_g_list, ch_r_list    
        
     # changing the light(training light) from cartesian coordinate to spherical cooridinates
    light_train = pd.read_csv(path_train_light, names = [0,1,2])
    light_file_train = light_train.to_numpy()
    n1 = len(light_file_train) 
    light_dir = []
    for i in range(n1):
        x, y, z = light_file_train[i]
        light_dir.append(to_spherical(x,y,z))
    light_file_train = np.stack(light_dir, axis = 0)
    
    light_file_train =  expand_light(light_file_train)
    
    
    # changing the light(Testing light) from cartesian coordinate to spherical cooridinates
    light_test = pd.read_csv(path_test_light, names = [0,1,2])
    light_file_test = light_test.to_numpy()
    n3 = len(light_file_test) 
    light_dir = []
    for i in range(n3):
        x, y, z = light_file_test[i]
        light_dir.append(to_spherical(x,y,z))
    light_file_test  = np.stack(light_dir, axis = 0)
    light_file_test =  expand_light(light_file_test)
    
    # replicate the pixel cordinate matrix as many as number of lights 
    replicated_array_train = np.vstack([mgrid]*n1)
    replicated_array_test = np.vstack([mgrid]*n3)
    
    # replicating each  light direction (256x256) time
    light_array_train = np.repeat(light_file_train, repeats = len(mgrid), axis =0).astype(np.float32) 
    light_array_test = np.repeat(light_file_test, repeats = len(mgrid), axis =0).astype(np.float32) 

    # stacking light_array matrix and coordinate matrix
    # training set
    train_list = []
    train_list.append(replicated_array_train)
    train_list.append(light_array_train)
    train_list.append(image_list_train)
    train = np.hstack(train_list)
    df_train = pd.DataFrame(train)

    # shuffling the dataframe
    df_train = df_train.sample(frac = 1, random_state=42)
    leng = int(df_train.shape[0]*(10/100))
    
    # randomly sampling for validation(10% of the training examples)
    df_val = df_train.sample(n = leng, random_state=52)
    df_train.drop(df_val.index, inplace=True, axis=0)
    
    # test set
    test_list = []
    test_list.append(replicated_array_test)
    test_list.append(light_array_test)
    test_list.append(image_list_test)
    test = np.hstack(test_list)
    df_test = pd.DataFrame(test)
  
    csv_to_h5py(df_train, 'train.h5', data_save_dir)
    csv_to_h5py(df_val, 'val.h5', data_save_dir)  
    csv_to_h5py(df_test, 'test.h5', data_save_dir)  


'''
Runing the program starts here, once all the paths are provided, Open CMD and run 'python data_preprocess.py'.
then the dataset will be proceed and stored in HDF5 file format
'''
path_train = arg.dataset_dir_raw + "train"
path_test = arg.dataset_dir_raw + "test"    
path_train_light = arg.dataset_dir_raw + "light_train" 
path_test_light = arg.dataset_dir_raw + "light_test" 

# data path for saving the proceesed data in hdf5 format
data_save_dir = arg.dataset_dir_raw
print('Data Pre_processing started...')
processTrainData(path_train, path_train_light,  path_test, path_test_light, data_save_dir)
print('Data Pre_processing completed...')