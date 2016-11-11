import pandas as pd
import numpy as np
import random
from sklearn.cluster import AgglomerativeClustering
from itertools import product
from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.inf)

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
N_plots = 20

class EyeCenterIdentifier:
    ''' Train the classifier model and predict the result'''
    def __init__(self, clf, step_size, N_steps, verbose=False):
        self.clf = clf  
        self.step_size = step_size
        self.N_steps = N_steps
        self.verbose = verbose
        
    def fit(self, train_X, train_y, train_pos):
        ''' Train the classifier'''
        self.train_X = train_X
        self.train_y = train_y
        self.train_pos = train_pos       
        self.left_eye_x_mean = self.train_pos.left_eye_center_x.mean()
        self.left_eye_y_mean = self.train_pos.left_eye_center_y.mean()
        self.right_eye_x_mean = self.train_pos.right_eye_center_x.mean()
        self.right_eye_y_mean = self.train_pos.right_eye_center_y.mean()        
        self.clf.fit(train_X, train_y)
        return self.clf
        
    def _cut_image(self, center_x, center_y, half_width, half_height):
        ''' Return the indices of pixels to form the sub-image and the center of the sub-image. 
        For a given certer (X, Y), get the indices of pixels so that the pixels can generate a subplot
        of size width * height with the center as center. If the distance between the center and the border is less than half_width or half_height, generate the subplot from the border, recalculate the center. 
        '''
        temp_func1 = lambda r: r if r > 0 else 0
        temp_func2 = lambda r: r if r < SIZE else SIZE
        start_x = int(center_x - half_width)
        start_y = int(center_y - half_height)
        end_x = start_x + half_width*2
        end_y = start_y + half_height*2
        if start_x < 0  or start_y < 0 :
            start_x = temp_func1(start_x)
            start_y = temp_func1(start_y)
            end_x = start_x + half_width*2
            end_y = start_y + half_height*2
            center_x = start_x + half_width
            center_y = start_y + half_height
        if end_x > SIZE  or end_y > SIZE:
            end_x = temp_func2(end_x)
            end_y = temp_func2(end_y)
            start_x = end_x - half_width*2
            start_y = end_y - half_height*2
            center_x = start_x + half_width
            center_y = start_y + half_height
        index_start = start_y * SIZE + start_x  
        index = np.empty(0)
        for i in range(0, int(half_height*2)):
            index = np.append(index, np.arange(index_start, index_start + half_width * 2))
            index_start += SIZE
        index = index.astype(int)
        return ((center_x, center_y), index)
      
        
    def _gen_test_data(self, data_image, start_pos, step_size, N_steps):
        ''' Generate sub-images within the searching region
            Searching image: start_x - step_x*N_x <= X <= start_x + step_x*N_x
                             start_y - step_y*N_y <= Y <= start_y - step_y*N_y
        '''
        start_x, start_y = (int(start_pos[0]), int(start_pos[1]))
        step_x, step_y = step_size
        N_x, N_y = N_steps
        x_range = range(start_x - step_x*N_x, start_x + step_x*N_x + 1, step_x)
        y_range = range(start_y - step_y*N_y, start_y + step_y*N_y + 1, step_y)
        centers = np.array([(x,y) for y in y_range for x in x_range ])
        data_temp = pd.DataFrame(columns = range(HEIGHT * WIDTH))
        centers_true = []
        for c in centers:
            temp_index = self._cut_image(c[0], c[1], half_WIDTH, half_HEIGHT)  
            ima = pd.Series(data_image[temp_index[1]])
            centers_true.append(temp_index[0])
            ima.index = data_temp.columns
            data_temp = data_temp.append(ima, ignore_index = True)
        centers_true = np.array(centers_true)
        return (centers_true, data_temp)
    
    def _serach_eye_center(self, data_image, start_pos, step_size, N_steps, eye='left', has_prob=False):
        '''Small-region search near the eye.
        '''
        gen_test_data = self._gen_test_data(data_image, start_pos, step_size, N_steps)
        centers = gen_test_data[0]
        data_temp = gen_test_data[1]
        pred = self.clf.predict(data_temp) 
        if self.verbose: 
            print ("Search near the {} eye centers: ".format(eye))
            print (pred)
        if 1 in pred:
            if has_prob: 
                prob = self.clf.predict_proba(data_temp)[pred==1]
                prob = np.array([p[1] for p in prob])
            else: 
                prob = np.array([1] * len(pred[pred == 1]))
            x_mean = np.average([c[0] for c in centers[pred == 1]], weights = prob)
            y_mean = np.average([c[1] for c in centers[pred == 1]], weights = prob)
            return x_mean, y_mean
        else: 
            return None, None
               
    def _serach_eye_center_overall(self, data_image, eye='left', has_prob=False):
        '''Overall search. 
           step size is set to be eye 
           eye can take 3 values ['left', 'right', 'both']
        '''
        gen_test_data = self._gen_test_data(data_image, (SIZE/2, SIZE/2), (2,2), (19, 21))
        centers = gen_test_data[0]
        data_temp = gen_test_data[1]
        pred = self.clf.predict(data_temp) 
        if self.verbose: 
            print("Overall search for the {} eye: ".format(eye))
            print (pred)
        if pred[pred==1].sum()>=2: 
            # do clusterings to get left eye center and right eye center 
            centers_p = centers[pred == 1]
            clt = AgglomerativeClustering(n_clusters=2)
            centers_c = clt.fit_predict(centers_p)
            if has_prob: 
                prob = self.clf.predict_proba(data_temp)[pred==1]
                prob = np.array([p[1] for p in prob])
            else: 
                prob = np.array([1] * len(pred[pred == 1]))
            eye_x = np.empty(0)
            eye_y = np.empty(0)
            for cl in set(centers_c):
                centers_temp = centers_p[centers_c == cl]
                prob_temp = prob[centers_c == cl]
                x_temp = np.average([c[0] for c in centers_temp], weights = prob_temp)
                y_temp = np.average([c[1] for c in centers_temp], weights = prob_temp)
                # use the median instead of the mean to avoid the effects from the extreme cases
                #x_temp = np.meidan([c[0] for c in centers_temp])
                #y_temp = np.median([c[1] for c in centers_temp])
                
                eye_x = np.append(eye_x, x_temp)
                eye_y = np.append(eye_y, y_temp) 
            left_eye_x = eye_x[0] if eye_x[0] >= eye_x[1] else eye_x[1]
            left_eye_y = eye_y[0] if eye_x[0] >= eye_x[1] else eye_y[1]
            right_eye_x = eye_x[0] if eye_x[0] < eye_x[1] else eye_x[1]
            right_eye_y = eye_y[0] if eye_x[0] < eye_x[1] else eye_y[1]
        else: 
            left_eye_x = self.left_eye_x_mean
            left_eye_y = self.left_eye_y_mean
            right_eye_x = self.right_eye_x_mean
            right_eye_y = self.right_eye_y_mean
        if eye == 'left': return left_eye_x, left_eye_y
        elif eye == 'right': return right_eye_x, right_eye_y
        else: return left_eye_x, left_eye_y, right_eye_x, right_eye_y

    
        
    def predict(self, test_X, has_prob=False):
        ''' Predict the eye centers.
        '''
        # Set the predict arrays
        left_eye_x_p = np.empty(0)
        left_eye_y_p = np.empty(0)
        right_eye_x_p = np.empty(0)
        right_eye_y_p = np.empty(0)        
        data_pred = pd.DataFrame(columns=['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y'])
        for i, data_image in enumerate(test_X):
            if not data_image.shape:
		# data_image is a single line
                if i == 0:
                    data_image = test_X
                else:
                    break
            if self.verbose:
                print ("Predict the {}th sample".format(i))
            left_eye_x_pred, left_eye_y_pred = self._serach_eye_center(data_image, (self.left_eye_x_mean, self.left_eye_y_mean), self.step_size, self.N_steps, 'left', has_prob) 
            right_eye_x_pred, right_eye_y_pred = self._serach_eye_center(data_image, (self.right_eye_x_mean, self.right_eye_y_mean), self.step_size, self.N_steps, 'right', has_prob)
            if (left_eye_x_pred is None and left_eye_y_pred is None) and not (right_eye_x_pred is None and right_eye_y_pred is None):
                left_eye_x_pred, left_eye_y_pred = self._serach_eye_center_overall(data_image, 'left', has_prob)
            elif not (left_eye_x_pred is None and left_eye_y_pred is None) and (right_eye_x_pred is None and right_eye_y_pred is None):
            	right_eye_x_pred, right_eye_y_pred = self._serach_eye_center_overall(data_image, 'right', has_prob)
            elif left_eye_x_pred is None and left_eye_y_pred is None and right_eye_x_pred is None and right_eye_y_pred is None:
                left_eye_x_pred, left_eye_y_pred, right_eye_x_pred, right_eye_y_pred = self._serach_eye_center_overall(data_image, 'both', has_prob)      
            left_eye_x_p = np.append(left_eye_x_p, left_eye_x_pred)
            left_eye_y_p = np.append(left_eye_y_p, left_eye_y_pred)
            right_eye_x_p = np.append(right_eye_x_p, right_eye_x_pred)
            right_eye_y_p = np.append(right_eye_y_p, right_eye_y_pred)        
        data_pred.left_eye_center_x = left_eye_x_p
        data_pred.left_eye_center_y = left_eye_y_p
        data_pred.right_eye_center_x = right_eye_x_p
        data_pred.right_eye_center_y = right_eye_y_p
        return data_pred
        
    def get_mse(self, data_pred, test_pos):
        '''return the mse.
        '''
        if isinstance(test_pos, pd.Series):
            test_pos = pd.DataFrame(test_pos).transpose()
        data_pred_copy = data_pred.copy()
        data_pred_copy.columns = test_pos.columns
        data_pred_copy.index = test_pos.index        
        mse = (data_pred_copy - test_pos).apply(np.square).sum().sum()/(data_pred_copy.shape[0] * data_pred_copy.shape[1])
        return mse

    def draw_face(self, data_image):
        '''Draw the facial image with the positively predicted sub-images shown.
        '''
        plt.imshow(data_image.reshape((SIZE, SIZE)), cmap=plt.cm.gray)
        for _eye in ['left', 'right']:
            if _eye == 'left': 
                start_pos = (self.left_eye_x_mean, self.left_eye_y_mean)
            else:
                start_pos = (self.right_eye_x_mean, self.right_eye_y_mean)     
            gen_test_data = self._gen_test_data(data_image, start_pos, self.step_size, self.N_steps)
            centers = gen_test_data[0]
            data_temp = gen_test_data[1]
            pred = self.clf.predict(data_temp)
            if not 1 in pred:
                gen_test_data = self._gen_test_data(data_image, (SIZE/2, SIZE/2), (2,2), (19, 21))
                centers = gen_test_data[0]
                data_temp = gen_test_data[1]
                pred = self.clf.predict(data_temp)                      
            for i, c in enumerate(centers):
                c_x, c_y = c
                s_x = c_x - half_WIDTH
                s_y = c_y - half_HEIGHT
                e_x = c_x + half_WIDTH
                e_y = c_y + half_HEIGHT
                if pred[i] == 1: 
                    color = 'green' 
                #else: 
                 #   color = 'red'
            
                    plt.plot([s_x, e_x], [s_y, s_y], color=color, linestyle='--')
                    plt.plot([s_x, s_x], [s_y, e_y], color=color, linestyle='--')
                    plt.plot([s_x, e_x], [e_y, e_y], color=color, linestyle='--')
                    plt.plot([e_x, e_x], [s_y, e_y], color=color, linestyle='--')

                
        plt.xlim([0,SIZE])
        plt.ylim([SIZE ,0])
        return plt

    def draw_subplots(self, data_image, eye):
        ''' Show the predicted results of all the sub images.
        '''
        if eye == 'left': 
            start_pos = (self.left_eye_x_mean, self.left_eye_y_mean)
        else:
            start_pos = (self.right_eye_x_mean, self.right_eye_y_mean)  
        gen_test_data = self._gen_test_data(data_image, start_pos, self.step_size, self.N_steps)
        centers = gen_test_data[0]
        data_temp = gen_test_data[1]
        pred = self.clf.predict(data_temp)
        for i, d in data_temp.iterrows():
            plt.subplot(self.N_steps[1]*2+1, self.N_steps[0]*2+1, i+1)
            plt.imshow(d.reshape((HEIGHT, WIDTH)), cmap=plt.cm.gray)
            if pred[i] == 1:
                plt.text(s='1', x=18, y=3, color='green', fontsize=20)
            else:
                plt.text(s='0', x=18, y=3, color='red', fontsize=20)
            plt.axis('off')
        plt.show()
        return plt
        
        
	

class GridSearch():
    ''' Grid search with cross validation   
    '''

    def __init__(self, clf, para_dict, clf_para_dict, folds=3, random_state=1, verbose=False):
        self.clf = clf
        self.para_dict = para_dict  
        self.clf_para_dict = clf_para_dict
        self.folds = folds
        self.random_state = random_state     
        self.verbose = verbose        
      
    def _gen_cv(self, train_pos):
        ''' Generate indices for each training set and validate set for the cross validation.
            Because the number of rows of the training set = the number of rows of the training position set times N_plots, we have 2 sets of indeces: one for train_X and one for train_pos
        '''
        _x = train_pos.shape[0]
        split_i = int(_x / self.folds)
        random.seed(self.random_state)
        t_index_xy = np.empty(0)
        cv_index_xy = np.empty(0)
        t_index_pos = np.empty(0)
        cv_index_pos = np.empty(0)  
        shuffled_index=np.arange(_x)               
        for k in range(self.folds): 
            random.shuffle(shuffled_index)
            index_cv = shuffled_index[:split_i]
            index_t = shuffled_index[split_i:]
            t_index_pos = np.append(t_index_pos, np.array(index_t))
            cv_index_pos = np.append(cv_index_pos, np.array(index_cv))
            t_index_xy = np.append(t_index_xy, np.array([np.arange(i * N_plots, (i+1) * N_plots) for i in index_t]))
            cv_index_xy = np.append(cv_index_xy, np.array([np.arange(i * N_plots, (i+1) * N_plots) for i in index_cv]))
        t_index_pos = t_index_pos.reshape(self.folds, (_x-split_i))
        cv_index_pos = cv_index_pos.reshape(self.folds, split_i)
        t_index_xy = t_index_xy.reshape(self.folds, N_plots * (_x-split_i))
        cv_index_xy = cv_index_xy.reshape(self.folds, N_plots * split_i)
        return t_index_xy, cv_index_xy, t_index_pos, cv_index_pos
                
    def _gen_dict(self, para_dict):
        ''' Get product of the parameter dictionary
            {'a':[1,2], 'b':[3]} -> {'a': 1, 'b': 3}, {'a': 2, 'b': 3}
        '''
        dict_array = np.empty(0)
        dict_keys = list(para_dict.keys())
        v_tuples = product(*para_dict.values())
        for vt in v_tuples:
            dict_temp ={}
            for i, v in enumerate(vt):
                dict_temp[dict_keys[i]] = v
            dict_array = np.append(dict_array, dict_temp)
        return dict_array         
     
    def _fit_single(self, train_X, train_y, train_pos, validate_X, validate_pos, has_prob):
        ''' Fit the EyeCenterIdentifier with a single set of parameters. 
        '''
        score = np.empty(0)
        params = np.empty(0)
        for dict_p in self._gen_dict(self.para_dict):
            for dict_c in self._gen_dict(self.clf_para_dict):
                if self.verbose: 
                    print ('The parameters are {}, {}'.format(dict_p, dict_c))
                clf = self.clf(**dict_c, random_state=self.random_state)
                eye_clf = EyeCenterIdentifier(clf, **dict_p)
                eye_clf.fit(train_X, train_y, train_pos)
                data_pred = eye_clf.predict(validate_X, has_prob)
                mse = eye_clf.get_mse(data_pred, validate_pos)
                score = np.append(score, mse)
                params = np.append(params, {'model': clf, 'parameters': dict_p})
        return score, params
        
    def fit(self, train_X, train_y, train_images, train_pos, has_prob=False):
        '''Grid search among all possible parameter combinations.
           Return the model with minimal MSE
        '''
        t_index_xy, cv_index_xy, t_index_pos, cv_index_pos = self._gen_cv(train_pos)
        score = np.empty(0)
        params = np.empty(0)
        for i in range(self.folds):
            if self.verbose:
                print('Fitting fold {} in the total {} folds'.format(i, self.folds))
            train_X_temp = train_X.iloc[t_index_xy[i]]
            train_y_temp = train_y.iloc[t_index_xy[i]]
            cv_X_temp = train_images.iloc[cv_index_pos[i]]
            train_pos_temp = train_pos.iloc[t_index_pos[i]]
            cv_pos_temp = train_pos.iloc[cv_index_pos[i]]    
            score_temp, params_temp = self._fit_single(train_X_temp, train_y_temp, train_pos_temp, cv_X_temp, cv_pos_temp, has_prob)
            score = np.append(score, score_temp)
        score = score.reshape(self.folds, len(params_temp))
        score_mean = np.mean(score, axis=0)
        print("Scores for each parameter combination:")
        for i in range(len(score_mean)):
            print ("{}: {}".format(params_temp[i], score_mean[i]))
        min_index = np.argmin(score_mean)
        return params_temp[min_index], score_mean[min_index]

         
        

    