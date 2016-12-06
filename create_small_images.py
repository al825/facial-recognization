'''
This program resizes the facial images to be used as GUI buttons
'''
from PIL import Image
import pandas as pd
import my_func
from sklearn.model_selection import train_test_split
import numpy as np


def re_size(im, outfile):
    '''read in the image array, convert to image and save the resized image'''
    im = np.asarray(im, np.uint8)
    im.resize(96, 96)
    pil_im = Image.fromarray(im)
    resized_im = pil_im.resize((70, 70))
    resized_im.save(outfile)
    
if __name__ == '__main__':
    data_ori = pd.read_csv(r"..\data\training.csv")
    # use a subset of the data     
    data = data_ori.iloc[:200]
    images = data.Image.map(my_func.str_split) # Transfer Image into arrays
    data = data.drop('Image', 1)    
    data_pos = data[['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y']]   
    images_train, images_test, data_pos_train, data_pos_test = train_test_split(images, data_pos, test_size = 0.2, random_state = 312)
    for i in range(images_test.shape[0]):
        im = images_test.iloc[i]
        re_size(im, outfile=r"..\figures\small_images\image_{}.png".format(i))

