# Import libraries
import sys
import getopt
''' For a given image transformation type and a given classifier, first fit a model with the default parameter settings, then do grid search.
'''
import pandas as pd
import sklearn 
import numpy as np
import random
import my_func
import time
from eye_identifier import EyeCenterIdentifier, GridSearch
from image_preprocess import imanorm, histeq, imaderiv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.linear_model import SGDClassifier





if __name__ == '__main__':
    transform = sys.argv[1]
    model = sys.argv[2] 
    if transform not in ('none', 'histeq', 'derivative'):
        print('Invalid transformation')
        sys.quit()
    if model not in ('LogisticRegression', 'RandomForestClassifier', 'SVC', 'SGD'):
        print('Invalid model')
        sys.quit()

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

    # Import data
    data_ori = pd.read_csv(r"..\data\training.csv")
    
    # use a subset of the data 
    N_sub = 200
    data = data_ori.iloc[:N_sub]
    images = data.Image.map(my_func.str_split) # Transfer Image into arrays
    data = data.drop('Image', 1)    
    data_pos = data[['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y']]


    # Transform images
    if transform == 'histeq': 
        images = images.apply(histeq)
    elif transform =='derivative':
        images = images.apply(imaderiv)

    # Remove rows with nan positions
    nan_index = data_pos.index[data_pos.isnull().any(axis=1)]
    images = images.drop(nan_index, axis=0)
    data = data.drop(nan_index, axis=0)
    data_pos = data_pos.drop(nan_index, axis=0)

    # Split the data into training set and testing set
    images_train, images_test, data_pos_train, data_pos_test = train_test_split(images, data_pos, test_size = 0.2, random_state = 312)

    # Get 20 subplots from each image, 5 right eye, 5 left eye, 10 randomly selected subplots
    N_plots = 20

    # Create the eye training data set
    random.seed(123)
    col_names = ['pixel' + str(v) for v in range(0, HEIGHT * WIDTH)] + ['center_X', 'center_Y', 'is_eye']
    data_eye = pd.DataFrame(columns = col_names)   

    for i in range(0, images_train.shape[0]):
        center_X = np.empty(0)
        center_Y = np.empty(0)
        is_eye = np.empty(0)
        # Select the two eye subplots
        for _eye in ['left_eye_center', 'right_eye_center']:
            _eye_x = _eye + '_x'
            _eye_y = _eye + '_y'
            _x = data_pos_train.iloc[i][ _eye_x]
            _y = data_pos_train.iloc[i][ _eye_y]
            _x = np.array([_x-2, _x, _x+2, _x, _x])
            _y = np.array([_y, _y, _y, _y-1, _y+1])
            center_X = np.append(center_X, _x)
            center_Y = np.append(center_Y, _y)
            is_eye = np.append(is_eye, [1] * 5)
        # randomly select ten subplots
        for r in range(int(N_plots / 2)):
            while True:
                _x = random.uniform(0, SIZE)
                _y = random.uniform(0, SIZE)
                # do not want the random center to be too close to the eyes
                if not (abs(_x - data_pos_train.iloc[i][ 'left_eye_center_x']) + abs(_y - data_pos_train.iloc[i][ 'left_eye_center_y']) < HEIGHT + WIDTH or abs(_x - data_pos_train.iloc[i][ 'right_eye_center_x']) + abs(_y - data_pos_train.iloc[i][ 'right_eye_center_y']) < HEIGHT + WIDTH):
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

    # Get the train_X and train_y
    train_X = data_eye.drop(['center_X', 'center_Y', 'is_eye'], axis = 1)
    train_y = data_eye.is_eye
    train_images = images_train
    train_pos = data_pos_train
    test_X = images_test
    test_pos = data_pos_test       

    # A Benchmark
    # If use the mean center of the training set, what is the mse
    pred_data_mean = pd.DataFrame({'left_eye_x_mean': [train_pos.left_eye_center_x.mean()] * len(test_X),
                                   'left_eye_y_mean': [train_pos.left_eye_center_y.mean()] * len(test_X),
                                   'right_eye_x_mean': [train_pos.right_eye_center_x.mean()] * len(test_X),
                                   'right_eye_y_mean': [train_pos.right_eye_center_y.mean()] * len(test_X)})
    bene_eyeidentifier = EyeCenterIdentifier(None, None, None)                               
    mse_mean = bene_eyeidentifier.get_mse(pred_data_mean, test_pos) # bench mark: 2.96
    
    # Build predict models
    step_size = (2, 2)
    N_steps = (5, 2)
    
    if model == 'LogisticRegression':
        clf = LogisticRegression(random_state = 312)
        clf_g = LogisticRegression
        clf_para_dict = {'penalty': ['l1', 'l2']}
        has_prob = True
    elif model == 'RandomForestClassifier':
        clf = RandomForestClassifier(random_state = 123)
        clf_g = RandomForestClassifier
        clf_para_dict = {'n_estimators': [10, 50, 100], 'criterion': ['gini', 'entropy']}
        has_prob = True
    elif model == 'SVC':
        clf = SVC(random_state = 312)
        clf_g = SVC
        clf_para_dict = {'kernel': ['rbf', 'poly'], 'C': [1, 10], 'gamma': [1e-3, 1e-4]}
        has_prob = False
    elif model == 'SGD':
        clf = SGDClassifier(loss='log', random_state=312)
        clf_g = SGDClassifier
        clf_para_dict = {'loss': ['log', 'modified_huber'], 'penalty': ['l2', 'elasticnet'], 'alpha': [0.1, 0.0001]}
        has_prob = True

        
    eye_id = EyeCenterIdentifier(clf, step_size, N_steps, False)
    t1=time.time()
    clf = eye_id.fit(train_X, train_y, train_pos)
    print("Fitting a single model used {:.2f} seconds".format(time.time()-t1))
    t1=time.time()
    data_pred = eye_id.predict(test_X, has_prob=has_prob)
    print("Predicting a single model used {:.2f} seconds".format(time.time()-t1))
    mse = eye_id.get_mse(data_pred, test_pos)
    print ('The mse of the {} model is: {}'.format(model, mse))
 
    # Grid Search
    para_dict = {'step_size': [(2, 2), (1, 1)], 'N_steps': [(5, 2), (8, 4)]}
    len_para = [len(v1) for v1 in clf_para_dict.values()] + [len(v2) for v2 in para_dict.values()]
    len_para = np.prod(len_para)
    clf_GridSearch = GridSearch(clf_g, para_dict, clf_para_dict, 3, random_state=312, verbose=False)
    t1=time.time()
    grid_result = clf_GridSearch.fit(train_X, train_y, train_images, train_pos, has_prob=has_prob)
    t_pass = time.time()-t1
    print("Grid Search used {:.2f} seconds. On average, 1 search used {:.2f} seconds".format(t_pass, t_pass/len_para))
    print('Grid Search result:')
    print(grid_result)
    clf = grid_result[0]['model']
    step_size = grid_result[0]['parameters']['step_size']
    N_steps = grid_result[0]['parameters']['N_steps']
    eye_id = EyeCenterIdentifier(clf, step_size, N_steps, False)
    clf = eye_id.fit(train_X, train_y, train_pos)
    data_pred = eye_id.predict(test_X, has_prob=has_prob)
    mse = eye_id.get_mse(data_pred, test_pos) 
    print ('The mse of the model selected by grid search is {}'.format(mse))

    


    