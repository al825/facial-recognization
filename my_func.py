#read in data
import pandas as pd
#import sklearn 
import numpy as np
from matplotlib import pyplot as plt

global SIZE
SIZE = 96



#data_ori = pd.read_csv(r".\data\training\training.csv")
#images = data_ori.Image.map(str_split)

def str_split(str):
    return pd.Series(map(int,str.split()))
    
    
def draw_face(data_ori, images, size, i=0): 
    '''draw a face and mark the key points '''
    plt.imshow(images[i].reshape((size, size)), cmap=plt.cm.gray)
    plt.plot(data_ori.left_eye_center_x[i], data_ori.left_eye_center_y[i], 'r.')
    plt.plot(data_ori.right_eye_center_x[i], data_ori.right_eye_center_y[i], 'r.')
    plt.plot(data_ori.left_eye_inner_corner_x[i], data_ori.left_eye_inner_corner_y[i], 'r.')
    plt.plot(data_ori.left_eye_outer_corner_x[i], data_ori.left_eye_outer_corner_y[i], 'r.')
    plt.plot(data_ori.right_eye_inner_corner_x[i], data_ori.right_eye_inner_corner_y[i], 'r.')
    plt.plot(data_ori.right_eye_outer_corner_x[i], data_ori.right_eye_outer_corner_y[i], 'r.')
    plt.plot(data_ori.left_eyebrow_inner_end_x[i], data_ori.left_eyebrow_inner_end_y[i], 'r.')
    plt.plot(data_ori.left_eyebrow_outer_end_x[i], data_ori.left_eyebrow_outer_end_y[i], 'r.')
    plt.plot(data_ori.right_eyebrow_inner_end_x[i], data_ori.right_eyebrow_inner_end_y[i], 'r.')
    plt.plot(data_ori.right_eyebrow_outer_end_x[i], data_ori.right_eyebrow_outer_end_y[i], 'r.')
    plt.plot(data_ori.nose_tip_x[i], data_ori.nose_tip_y[i], 'r.')
    plt.plot(data_ori.mouth_left_corner_x[i], data_ori.mouth_left_corner_y[i], 'r.')
    plt.plot(data_ori.mouth_right_corner_x[i], data_ori.mouth_right_corner_y[i], 'r.')
    plt.plot(data_ori.mouth_center_top_lip_x[i], data_ori.mouth_center_top_lip_y[i], 'r.')
    plt.plot(data_ori.mouth_center_bottom_lip_x[i], data_ori.mouth_center_bottom_lip_y[i], 'r.')
    plt.xlim([0,size])
    plt.ylim([size,0])
    #fig.show()
    return plt
    
def draw_face2 (data_pos, images, size, parts, i=0): 
    '''Draw a face and mark the key points of the whole data set'''
    fig = plt.figure(figsize = (size, size))
    plt.subplot(1,1,1)
    plt.imshow(images[i].reshape((size, size)), cmap=plt.cm.gray)
    col = plt.cm.rainbow(np.linspace(0, 1, len(parts)))
    for index, p in enumerate(parts):        
        var_x = p + '_x'
        var_y = p + '_y'
        
        for r in range(data_pos.shape[0]):
            if r != i:
                plt.plot(data_pos[var_x][r], data_pos[var_y][r], marker='.', color=col[index])
        plt.plot(data_pos[var_x][i], data_pos[var_y][i], marker='.', color=col[index], ms=2)
    plt.xlim([0, size])
    plt.ylim([size, 0])
    plt.show()
    
#draw_face2(0, ['left_eye_center'])
#draw_face2(0, ['left_eye_center', 'right_eye_center'])

def cut_image(center_x, center_y, half_width, half_height):
    '''For a given certer (X, Y), get the indices of pixels so that the pixels can generate a subplot
    of size width * height with the center as center. If the distance between the center and the border is less than half_width or half_height, generate the subplot from the border, recalculate the center.
    Return the index and the cneter'''
    temp_func1 = lambda r: r if r > 0 else 0
    temp_func2 = lambda r: r if r < 96 else 96
    start_x = int(center_x - half_width)
    start_y = int(center_y - half_height)
    end_x = start_x + half_width * 2
    end_y = start_y + half_height * 2
    if start_x < 0  or start_y < 0 :
        start_x = temp_func1(start_x)
        start_y = temp_func1(start_y)
        end_x = start_x + half_width * 2
        end_y = start_y + half_height * 2
        center_x = start_x + half_width
        center_y = start_y + half_height
    if end_x > 96  or end_y > 96:
        end_x = temp_func2(end_x)
        end_y = temp_func2(end_y)
        start_x = end_x - half_width * 2
        start_y = end_y - half_height * 2
        center_x = start_x + half_width
        center_y = start_y + half_height
    index_start = start_y * 96 + start_x  
    index = np.empty(0)
    for i in range(0, int(half_height * 2)):
        index = np.append(index, np.arange(index_start, index_start + half_width * 2))
        index_start += 96
    index = index.astype(int)
    return ((center_x, center_y), index)
    
def padding_image(data, height, width):
    data_height = data.shape[0]
    data_width = data.shape[1]
    up_n = round((height - data_height)/2)
    down_n = height - data_height - up_n
    left_n = round((width - data_width)/2)
    right_n = width - data_width - left_n
    gen_data = np.empty(0)
    for i in range(0, data_height):
        temp = np.lib.pad(data.iloc[i], (left_n, right_n), 'constant', constant_values=(0, 0))
        gen_data = np.append(gen_data, temp)
    start = np.zeros(up_n * width)
    end = np.zeros(down_n * width)
    gen_data = np.append(start, gen_data)
    gen_data = np.append(gen_data, end)
    gen_data.reshpae(height, width)
    return gen_data

def draw_eye(data, size, predict = None):
    plt.imshow(data.reshape(size), cmap=plt.cm.gray)
    if predict: 
        plt.text(s=predict, x=18, y=3, color='red', fontsize=20)
    return plt

def draw_results(image, true_values, pred_values, mean_values, draw_mean=True):
    plt.imshow(image.reshape((SIZE, SIZE)), cmap=plt.cm.gray)
    true_pos, = plt.plot(true_values.left_eye_center_x, true_values.left_eye_center_y, 'g.', label='True Position')
    plt.plot(true_values.right_eye_center_x, true_values.right_eye_center_y, 'g.')
    pred_pos, = plt.plot(pred_values.left_eye_center_x, pred_values.left_eye_center_y, 'r.', label='Predicted Position')
    plt.plot(pred_values.right_eye_center_x, pred_values.right_eye_center_y, 'r.')
    if draw_mean:
        mean_pos, = plt.plot(mean_values.left_eye_x_mean, mean_values.left_eye_y_mean, 'b.', label='Average Position')
        plt.plot(mean_values.right_eye_x_mean, mean_values.right_eye_y_mean, 'b.')
    plt.xlim([0,SIZE])
    plt.ylim([SIZE,0])
    #if draw_mean:
        #plt.legend(handles = [true_pos, pred_pos, mean_pos])
    #else:
        #plt.legend(handles = [true_pos, pred_pos])
    return plt

    

    
      
        
        
        
    
