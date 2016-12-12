# Eye Center Identification
Image recognition has become a popular and challenging topic in the field of machine learning.This project is a symplified version of Kaggle's competition--[Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection). 

This project applies machine learning classification techniques to make predictions of the eye center locations on facial images. A GUI application has been made to show the predicted results. 

## Data

1. Original data source: https://www.kaggle.com/c/facial-keypoints-detection/data training.zip
2. Data used in this project: the first 200 samples from the original data set

## Language
The programs in this project are written in Python3.5.

## Programs and Usage
1. my_func.py
    * description: contains functions for data processing and results visualization
    * usage: called by BuildModel_entire.py, best_model.py

2. image_preprocess.py
    * description: contains functions for image transformation
    * usage: called by BuildModel_entire.py, best_model.py

3. eye_identifier.py
     * description: contains EyeCenterIdentifier class and GridSearch class
     * usage: called by BuildModel_entire.py and best_model.py

4. BuildModel_entire.py
    * description: process the data, build EyeCenterIdentifier and do GridSearch 
    * call using command line:   
      ipython3 BuildModel_entire.py [transformation] [clf]
      transformation can take values: none, histeq, derivative
      clf can take values: LogisticRegression, RandomForestClassifier, SVC, SGD
      e.g ipython3 BuildModel_entire.py none LogisticRegression

## Example
Here is an example of the predictions of eye locations: 
![Example](https://github.com/al825/facial-recognization/blob/master/image_for_readme.png)  

The red dots represent the predicted eye center locations and the blue dots represent the results from the benchmark model. 



