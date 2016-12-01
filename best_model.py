# Import libraries
import pandas as pd
import sklearn 
import numpy as np
import random
from matplotlib import pyplot as plt
import my_func
import time
from eye_identifier import EyeCenterIdentifier, GridSearch
from image_preprocess import imanorm, histeq, imaderiv
from sklearn.ensemble import RandomForestClassifier


class BestModel():
    SIZE = 96
    HEIGHT = 12
    WIDTH = 20
    half_HEIGHT = 6
    half_WIDTH = 10
    N_sub = 200
    N_plots = 20
    
    def __init__(self):
        
        
    def build_model(self):
    # Import data
    data_ori = pd.read_csv(r"..\data\training.csv")

    # use a subset of the data 
    
    data = data_ori.iloc[:BestModel.N_sub]
    images = data.Image.map(my_func.str_split) # Transfer Image into arrays
    data = data.drop('Image', 1)    
    data_pos = data[['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y']]


    # Remove rows with nan positions
    nan_index = data_pos.index[data_pos.isnull().any(axis=1)]
    images = images.drop(nan_index, axis=0)
    data = data.drop(nan_index, axis=0)
    data_pos = data_pos.drop(nan_index, axis=0)

    # histeq transform
    images = images.apply(histeq)
    #images = images.apply(imaderiv)

    # Split the data into training set and testing set
    from sklearn.model_selection import train_test_split
    images_train, images_test, data_pos_train, data_pos_test = train_test_split(images, data_pos, test_size = 0.2, random_state = 312)

    # Get 20 subplots from each image, 1 right eye, 1 left eye, 2 randomly selected subplots
    # Create the eye training data set
    random.seed(123)
    col_names = ['pixel' + str(v) for v in range(0, HEIGHT * WIDTH)] + ['center_X', 'center_Y', 'is_eye']
    data_eye = pd.DataFrame(columns = col_names)   

    for i in range(0, images_train.shape[0]):
        t1=time.time()
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
            is_eye = np.append(is_eye, [1] * int(N_plots / 4))
        # randomly select two subplots
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
        
    def make_prediction(self):


# Set up global parameters
global SIZE
global HEIGHT 
global WIDTH 
global half_HEIGHT 
global half_WIDTH
global N_plots 


       

# Get the train_X and train_y
train_X = data_eye.drop(['center_X', 'center_Y', 'is_eye'], axis = 1)
train_y = data_eye.is_eye
train_images = images_train
train_pos = data_pos_train
test_X = images_test
test_pos = data_pos_test

rate_eye = sum(data_eye.is_eye == 1)/data_eye.shape[0]

print ("{:.2} of the subplots are eye.".format(rate_eye))


# A Benchmark
# If use the mean center of the training set, what is the mse
pred_data_mean = pd.DataFrame({'left_eye_x_mean': [train_pos.left_eye_center_x.mean()] * len(test_X),
                               'left_eye_y_mean': [train_pos.left_eye_center_y.mean()] * len(test_X),
                               'right_eye_x_mean': [train_pos.right_eye_center_x.mean()] * len(test_X),
                               'right_eye_y_mean': [train_pos.right_eye_center_y.mean()] * len(test_X)})
eye_bench = EyeCenterIdentifier(None, None, None)
mse_mean = eye_bench.get_mse(pred_data_mean, test_pos) # bench mark: 2.96


# Build predict models
step_size = (1, 1)
N_steps = (8, 4)



clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=50, n_jobs=1, oob_score=False, random_state=312,
            verbose=0, warm_start=False)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                         intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                         penalty='l1', random_state=312, solver='liblinear', tol=0.0001,
                         verbose=0, warm_start=False)

from sklearn.linear_model import SGDClassifier      
clf = SGDClassifier(alpha=0.1, average=False, class_weight=None, epsilon=0.1,
              eta0=0.0, fit_intercept=True, l1_ratio=0.15,
              learning_rate='optimal', loss='log', n_iter=5, n_jobs=1,
              penalty='l2', power_t=0.5, random_state=312, shuffle=True,
              verbose=0, warm_start=False)
       
eye_id = EyeCenterIdentifier(clf, step_size, N_steps)
t1 = time.time()
clf = eye_id.fit(train_X, train_y, train_pos)
print("Time to fit the model: {:.2f} seconds".format(time.time()-t1))
t1 = time.time()
data_pred = eye_id.predict(test_X, has_prob = True)
print("Time to make the prediction: {:.2f} seconds".format(time.time()-t1))
mse = eye_id.get_mse(data_pred, test_pos) #1.63

# Draw plots
# Draw the subplots
for i in range(40):
    fig = plt.subplot(5, 8, (i+1))
    fig = eye_id.draw_face(test_X.iloc[i])
plt.show()

# Draw the predicted positions
for i in range(40): 
    plt.subplot(5, 8, (i+1))
    my_func.draw_results(test_X.iloc[i], test_pos.iloc[i], data_pred.iloc[i], pred_data_mean.iloc[0], draw_mean=True)
plt.show()

# draw the most accurate and the most wrong
data_pred_copy = data_pred.copy()
data_pred_copy.columns = test_pos.columns
data_pred_copy.index = test_pos.index        
row_se = (data_pred_copy - test_pos).apply(np.square).sum(axis=1)
i_min = row_se.argmin() #0.34
i_max = row_se.argmax() #32.49

plt.subplot(2,2,1)
eye_id.draw_face(test_X.loc[i_min])
plt.subplot(2,2,2)
my_func.draw_results(test_X.loc[i_min], test_pos.loc[i_min], data_pred_copy.loc[i_min], pred_data_mean.iloc[0], draw_mean=True)
plt.subplot(2,2,3)
eye_id.draw_face(test_X.loc[i_max])
plt.subplot(2,2,4)
my_func.draw_results(test_X.loc[i_max], test_pos.loc[i_max], data_pred_copy.loc[i_max], pred_data_mean.iloc[0], draw_mean=True)
fig.show()

eye_id.draw_subplots(test_X.loc[i_min], 'left')
eye_id.draw_subplots(test_X.loc[i_min], 'right')

eye_id.draw_subplots(test_X.loc[i_max], 'left')
eye_id.draw_subplots(test_X.loc[i_max], 'right')

# Which ones has outlier eyes (not working)
# max left eye x
max_index = data_ori.left_eye_center_x.argmax()
image_max = my_func.str_split(data_ori.iloc[max_index].Image)
data_pred_max = eye_id.predict(image_max, True)
pos_max = data_ori.loc[max_index, ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y']]
mse_max = eye_id.get_mse(data_pred_max, pos_max)
# 1409 random forest

min_index = data_ori.right_eye_center_x.argmin()
image_min = my_func.str_split(data_ori.iloc[min_index].Image)
data_pred_min = eye_id.predict(image_min, True)
pos_min = data_ori.loc[min_index, ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y']]
mse_min = eye_id.get_mse(data_pred_min, pos_min) #601 random forest



eye_id.draw_subplots(image_max, 'left')
eye_id.draw_subplots(image_max, 'right')

eye_id.draw_subplots(image_min, 'left')
eye_id.draw_subplots(image_min, 'right')

plt.subplot(2,2,1)
eye_id.draw_face(image_max)
plt.subplot(2,2,2)
my_func.draw_results(image_max, pos_max, data_pred_max, pred_data_mean.iloc[0], draw_mean=True)
plt.subplot(2,2,3)
eye_id.draw_face(image_min)
plt.subplot(2,2,4)
my_func.draw_results(image_min, pos_min, data_pred_min, pred_data_mean.iloc[0], draw_mean=True)
plt.show()

