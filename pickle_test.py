# Import libraries
import os
import sys
import pandas as pd
import sklearn 
import numpy as np
import random
import pickle
from matplotlib import pyplot as plt
import my_func
import time
from eye_identifier import EyeCenterIdentifier, GridSearch
from image_preprocess import imanorm, histeq, imaderiv

# Set up global parameters
global SIZE
global HEIGHT 
global WIDTH 
global half_HEIGHT 
global half_WIDTH
global N_plots 
SIZE = 96
HEIGHT = 12
WIDTH = 20
half_HEIGHT = 6
half_WIDTH = 10


if __name__ == '__main__':
    datapath = '../pickles/datasets.pickle'
    modelpath = '../pickles/best_model.pickle'
    if os.path.exists(datapath):
        with open(datapath, 'rb') as datasets:
            train_X, train_y, train_images, train_pos, test_X, test_pos = pickle.load(datasets)
    else:
        print('No pickled data sets')
        sys.exit()
    if os.path.exists(modelpath):
        with open(modelpath, 'rb') as model:
            best_model = pickle.load(model)
    else:
        print('No model pickle.')
        sys.exit()
    
    data_pred = best_model.predict(test_X, has_prob = True)
    mse = best_model.get_mse(data_pred, test_pos) #1.63
    print('The MSE of the model is {:.2f}'.format(mse))
    