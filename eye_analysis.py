# Import libraries
import pandas as pd
import sklearn 
import numpy as np
import random
from matplotlib import pyplot as plt
import my_func
import time

# Import data
data_ori = pd.read_csv(r"..\data\training\training.csv")

# Transfer Image  into arrays
images = data_ori.Image.map(my_func.str_split)

data = data_ori.drop('Image', 1)

# use a subset of the data 
N = 1000 
images = images[:N]
data = data.iloc[:N]

# set up parameters
HEIGHT = 12
WIDTH = 20
half_HEIGHT = 6
half_WIDTH = 10

# Split the data into training set and testing set
from sklearn.model_selection import train_test_split
data_train, data_test, images_train, images_test = train_test_split(data, images, test_size = 0.25, random_state = 312)
print ("The training set has {} observations and the testing set has {} observations". format(data_train.shape[0], data_test.shape[0]))





# Get 12 subplots from each image
N_plots = 12

# Create the eye data set
random.seed(123)

col_names = ['pixel' + str(v) for v in range(0, HEIGHT * WIDTH)] + ['center_X', 'center_Y', 'is_eye']
data_eye = pd.DataFrame(columns = col_names)   

for i in range(0, data_train.shape[0]):
    t1=time.time()
    center_X = np.empty(0)
    center_Y = np.empty(0)
    is_eye = np.empty(0)
    for _eye in ['left_eye_center', 'right_eye_center']:
        _eye_x = _eye + '_x'
        _eye_y = _eye + '_y'
        if not (pd.isnull(data.loc[i, _eye_x]) and pd.isnull(data.loc[i, _eye_y])):
            _x = data.loc[i, _eye_x]
            _y = data.loc[i, _eye_y]
            center_X = np.append(center_X, np.array([_x, _x - half_WIDTH, _x + half_WIDTH, _x, _x]))
            center_Y = np.append(center_Y, np.array([_y, _y, _y, _y + half_HEIGHT, _y - half_HEIGHT]))
            is_eye = np.append(is_eye, [1, 0, 0, 0, 0])
    for r in [1,2]:
        while True:
            _x = random.uniform(0, 95)
            _y = random.uniform(0, 95)
            # do not want the random center to be too close to the eyes
            if not (abs(_x - data.loc[i, 'left_eye_center_x']) + abs(_y - data.loc[i, 'left_eye_center_y']) < HEIGHT + WIDTH or abs(_x - data.loc[i, 'right_eye_center_x']) + abs(_y - data.loc[i, 'right_eye_center_y']) < HEIGHT + WIDTH):
                break
        center_X = np.append(center_X, _x)
        center_Y = np.append(center_Y, _y)
        is_eye = np.append(is_eye, 0)

    for j in range (0,len(center_X)):
                    
        temp = my_func.cut_image(center_X[j], center_Y[j], half_WIDTH, half_HEIGHT)    
        ima = pd.Series(images_train.iloc[i][temp[1]])
        ima = ima.append(pd.Series([center_X[j], center_Y[j], is_eye[j]]))
        ima.index = col_names
        data_eye = data_eye.append(ima, ignore_index = True)           
    print("Iteration {} used {:.2f} seconds".format(i, time.time()-t1))

data_eye.shape
data_eye.head()

#remove rows with nan
nan_index = data_eye.index[data_eye.isnull().any(axis=1)]
data_eye = data_eye.drop(nan_index, axis=0)


# Get the train_X and train_y
train_X = data_eye.drop(['center_X', 'center_Y', 'is_eye'], axis = 1)
train_y = data_eye.is_eye


rate_eye = sum(data_eye.is_eye == 1)/data_eye.shape[0]

print ("{:.2} of the subplots are eye.".format(rate_eye))

# Logistic regression
from sklearn.linear_model import LogisticRegression
clf_log = LogisticRegression(random_state = 312)
clf_log.fit(train_X, train_y)
clf_log.score(train_X, train_y)

#try 
left_eye_x_mean = int(data.left_eye_center_x.mean())
left_eye_y_mean = int(data.left_eye_center_y.mean())

right_eye_x_mean = int(data.right_eye_center_x.mean())
right_eye_y_mean = int(data.right_eye_center_y.mean())


test0 = data_test.iloc[0]
ima_test0 = images_test[0]

temp = my_func.cut_image(test0.left_eye_center_x, test0.left_eye_center_y, half_WIDTH, half_HEIGHT) 

ima_try = ima_test0[temp[1]]
ima_try.reshape(1, -1)
ima_try.index = train_X.columns

clf_log.predict(ima_try)

# start from the mean of left eye center and right eye center
i=0 
step_x = 2
step_y = 2
N_x = 4
N_y = 2
x_range = range(left_eye_x_mean - step_x *  N_x, left_eye_x_mean + step_x *  N_x + 1, step_x)
y_range = range(left_eye_y_mean - step_y *  N_y, left_eye_y_mean + step_y *  N_y + 1, step_y)
centers = [(x,y) for x in x_range for y in y_range]

data_temp = pd.DataFrame(columns = col_names[:-3])

for c in centers:
    temp = my_func.cut_image(c[0], c[1], half_WIDTH, half_HEIGHT)    
    ima = pd.Series(images_test.iloc[i][temp[1]])
    ima.index = data_temp.columns
    data_temp = data_temp.append(ima, ignore_index = True)
clf_log.predict(data_temp)
