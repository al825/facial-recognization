# Import libraries
import pandas as pd
import sklearn 
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import my_func
import time
from eye_identifier import EyeCenterIdentifier, GridSearch
from image_preprocess import imanorm, histeq, imaderiv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class BestModel():
    SIZE = 96
    HEIGHT = 12
    WIDTH = 20
    half_HEIGHT = 6
    half_WIDTH = 10
    N_sub = 200
    N_plots = 20
        
    def __init__(self, clf, step_size = (1, 1), N_steps = (8, 4)):
        self.step_size = step_size
        self.N_steps = N_steps
        self.clf = clf
        self.data_pred = None
        self.mse = None
               
    def process_data(self, location = r"..\data\training.csv"):
        # Import data
        data_ori = pd.read_csv(location)

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
        
        images_train, images_test, data_pos_train, data_pos_test = train_test_split(images, data_pos, test_size = 0.2, random_state = 312)

        # Get 20 subplots from each image, 1 right eye, 1 left eye, 2 randomly selected subplots
        # Create the eye training data set
        random.seed(123)
        col_names = ['pixel' + str(v) for v in range(0, BestModel.HEIGHT * BestModel.WIDTH)] + ['center_X', 'center_Y', 'is_eye']
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
                is_eye = np.append(is_eye, [1] * int(BestModel.N_plots / 4))
            # randomly select two subplots
            for r in range(int(BestModel.N_plots / 2)):
                while True:
                    _x = random.uniform(0, BestModel.SIZE)
                    _y = random.uniform(0, BestModel.SIZE)
                    # do not want the random center to be too close to the eyes
                    if not (abs(_x - data_pos_train.iloc[i][ 'left_eye_center_x']) + abs(_y - data_pos_train.iloc[i][ 'left_eye_center_y']) < BestModel.HEIGHT + BestModel.WIDTH or abs(_x - data_pos_train.iloc[i][ 'right_eye_center_x']) + abs(_y - data_pos_train.iloc[i][ 'right_eye_center_y']) < BestModel.HEIGHT + BestModel.WIDTH):
                        break
                center_X = np.append(center_X, _x)
                center_Y = np.append(center_Y, _y)
                is_eye = np.append(is_eye, 0)

            for j in range (0,len(center_X)):            
                temp = my_func.cut_image(center_X[j], center_Y[j], BestModel.half_WIDTH, BestModel.half_HEIGHT)    
                ima = pd.Series(images_train.iloc[i][temp[1]])
                ima = ima.append(pd.Series([center_X[j], center_Y[j], is_eye[j]]))
                ima.index = col_names
                data_eye = data_eye.append(ima, ignore_index = True)     
            
        # Get the train_X and train_y
        BestModel.train_X = data_eye.drop(['center_X', 'center_Y', 'is_eye'], axis = 1)
        BestModel.train_y = data_eye.is_eye
        BestModel.train_images = images_train
        BestModel.train_pos = data_pos_train
        BestModel.test_X = images_test
        BestModel.test_pos = data_pos_test

        # A Benchmark
        # If use the mean center of the training set, what is the mse
        BestModel.mean_pos = {'left_eye_center_x': BestModel.train_pos.left_eye_center_x.mean(),
                              'left_eye_center_y': BestModel.train_pos.left_eye_center_y.mean(),
                              'right_eye_center_x': BestModel.train_pos.right_eye_center_x.mean(),
                              'right_eye_center_y': BestModel.train_pos.right_eye_center_y.mean()}
            
        self.data_pred = pd.DataFrame(columns = ('id', 'left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y'))
       
    
    def build_model(self):       
        self.eye_id = EyeCenterIdentifier(self.clf, self.step_size, self.N_steps)
        self.clf = self.eye_id.fit(BestModel.train_X, BestModel.train_y, BestModel.train_pos)
    
    def make_prediction(self, index):
        data_pred = self.eye_id.predict(BestModel.test_X.iloc[index], has_prob=True)
        mse = self.eye_id.get_mse(data_pred, BestModel.test_pos.iloc[index]) 
        data_pred['id'] = index
        self.data_pred = self.data_pred.append(data_pred)
        return mse
        
    def draw_face(self, index, size):
        image=BestModel.test_X.iloc[index]
        f = Figure(figsize=(5,5), dpi=100)
        a = f.add_subplot(111)
        a.imshow(image.reshape((size, size)), cmap=plt.cm.gray)
        a.set_xlim(0, size)
        a.set_ylim(size, 0)
        return f, a
        
        
        
        
    def draw_results(self, index, size, draw_true=False, draw_mean=False):
        image=BestModel.test_X.iloc[index]
        pred_values = self.data_pred
        #true_values = BestModel.test_pos.iloc[index]
        #mean_values = BestModel.mean_pos        
        plt.imshow(image.reshape((size, size)), cmap=plt.cm.gray)
        #pred_pos, = plt.plot(pred_values.left_eye_center_x, pred_values.left_eye_center_y, 'r.', label='Predicted Position')
        #plt.plot(pred_values.right_eye_center_x, pred_values.right_eye_center_y, 'r.')
        #if draw_true: 
         #   true_pos, = plt.plot(true_values.left_eye_center_x, true_values.left_eye_center_y, 'g.', label='True Position')
          #  plt.plot(true_values.right_eye_center_x, true_values.right_eye_center_y, 'g.')
    
        #if draw_mean:
         #   mean_pos, = plt.plot(mean_values.left_eye_x_mean, mean_values.left_eye_y_mean, 'b.', label='Average Position')
          #  plt.plot(mean_values.right_eye_x_mean, mean_values.right_eye_y_mean, 'b.')
        plt.xlim([0,size])
        plt.ylim([size,0])
        return plt
    
if __name__ == '__main__':
    step_size = (1, 1)
    N_steps = (8, 4)
    clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=50, n_jobs=1, oob_score=False, random_state=312,
            verbose=0, warm_start=False)
    best_model= BestModel(clf, step_size, N_steps)
    best_model.process_data()
    best_model.build_model()
    mse = best_model.make_prediction(1)
    print (mse)
    #fig = best_model.draw_face(index=1, size=96)


    
    


        

 