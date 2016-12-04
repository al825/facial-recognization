import tkinter as tk
from build_model_gui import BestModel
import pandas as pd
import sklearn 
import numpy as np
import random

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure


import my_func
import time
from eye_identifier import EyeCenterIdentifier, GridSearch
from image_preprocess import imanorm, histeq, imaderiv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class EyeCenterApp(tk.Tk):
    def __init__(self, width, height, best_model):
        tk.Tk.__init__(self)
        self.geometry('{}x{}'.format(width, height))
        self.wm_title('Eye Center Prediction')
        self.resizable(width=False, height=False)
        self.container = tk.Frame(self)
        self.container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)  
        self.best_model = best_model
        self.frames = {}
        self.init_page(StartPage)
        self.show_frame(StartPage)
        
    def init_page(self, page):
        self.frames[page] = page(self.container, self)
        #self.frames[page].config(height=height)
        #self.frames[page].config(width=width)
        self.frames[page].grid(row=0, column=0, sticky="nsew")
        
        
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
        self.controller.init_page(PageOne)
        self.controller.show_frame(PageOne)
        self.controller.after(1000, self.controller.process_data)
        self.controller.after(1000, self.controller.init_page, PageTwo)
        #self.controller.init_page(PageTwo)
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
        self.label.config(font=("Courier", 20))
        self.label.grid(row=0, column=0, columnspan=5)
        for i in range(10):
            img = tk.PhotoImage(file="dan4.gif")    
            ima_button = tk.Button(self, image=img, command=lambda: self.click_button(i=i))
            ima_button.image = img # keep a reference!
            ima_button.grid(row=int(i/5)+1, column=i-int(i/5)*5+1, padx=5, pady=5)
            
    def click_button(self, i):
        self.controller.init_page(PageThree)
        self.controller.show_frame(PageThree)
        
        

    
class PageThree(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.controller = controller
        self.create_widgets()
        
    def create_widgets(self):
        self.label = tk.Label(self, text='Predict')
        self.label.config(font=("Courier", 10))
        self.label.place(relx=0.1, rely=0.1, anchor=tk.CENTER)
        
        figure = self.controller.best_model.draw_face(1, 96)
        canvas = FigureCanvasTkAgg(figure, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)  
        #f = Figure(figsize=(5,5), dpi=100)
        #a = f.add_subplot(111)
        #a.plot([1,2,3,4,5,6,7,8],[5,6,1,3,8,9,3,5])
        #canvas = FigureCanvasTkAgg(f, self)
        #canvas.show()
        #canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        

        
    
    
        
        
        

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
    
    root = EyeCenterApp(width=500, height=800, best_model=best_model)
    root.mainloop()    
    
    
#Note: for image, 1. use paint to adjust the format; 2. do not forget to keep a reference

    


