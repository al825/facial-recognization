import tkinter as tk
from build_model_gui import BestModel
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
from sklearn.model_selection import train_test_split


class EyeCenterApp(tk.Tk):
    def __init__(self, height, width, best_model):
        tk.Tk.__init__(self)
        self.geometry('{}x{}'.format(height, width))
        self.wm_title('Eye Center Prediction')
        self.resizable(width=False, height=False)
        self.container = tk.Frame(self)
        self.container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)  
        self.best_model = best_model
        self.frames = {}
        for f in [StartPage, PageOne, PageTwo]:
            self.frames[f] = f(self.container, self)
            self.frames[f].config(height=height)
            self.frames[f].config(width=width)
            self.frames[f].grid(row=0, column=0, sticky="nsew")
        self.show_frame(StartPage)
        
    def show_frame(self, page):
        self.frames[page].tkraise()
        
    def process_data(self):
        self.best_model.process_data()
        
    def build_model(self):
        self.best_model.build_model()
        
        
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.controller = controller
        self.create_widgets()
    
    def create_widgets(self): 
        self.instruction = tk.Label(self, text='Eye Center Recognization')
        self.instruction.config(font=("Courier", 20))
        self.instruction.place(relx=0.5, rely=0.3, anchor=tk.CENTER)
        self.button = tk.Button(self, text = 'START', command=self.click_start)
        self.button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
     
    def click_start(self):
        #self.pack_forget()
        self.controller.show_frame(PageOne)
        #self.controller.show_frame(PageOne)
        #print('start')
        self.controller.after(1000, self.controller.process_data)
        #print('finish')
        self.controller.after(1000, self.controller.show_frame, PageTwo)
        #print('page2')
        #self.controller.frames[PageOne].pd()
        #self.controller.frames[PageOne].visible = True
        #time.sleep(5)
        #
        #print(self.controller.frames[PageOne].visible)
        #if self.controller.frames[PageOne].visible:
         #   time.sleep(20)
        # process the data and build the model
        #self.controller.process_data()
        #self.controller.build_model()
        
        #self.controller.show_frame(PageTwo)
        #print("Shown the PageTwo")
        #time.sleep(5)
        
                    
        
class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.controller = controller
        self.create_widgets()
    
    def create_widgets(self): 
        self.label = tk.Label(self,text='Creating Model')
        self.label.config(font=("Courier", 20))
        self.label.place(relx=0.4, rely=0.4, anchor=tk.CENTER)
        

     
        
class PageTwo(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.controller = controller
        self.create_widgets()
        
    def create_widgets(self):
        self.label = tk.Label(self, text='Predict')
        self.label.config(font=("Courier", 30))
        self.label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    
    
        
        
        

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
    
    root = EyeCenterApp(height=500, width=500, best_model=best_model)
    root.mainloop()    

    


