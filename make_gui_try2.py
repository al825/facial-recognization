import tkinter as tk
from build_model_gui import BestModel
import pandas as pd
import sklearn 
import numpy as np
import random
import threading


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
    '''This is the controller of the eye center app.'''
    def __init__(self, width, height, best_model, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.geometry('{}x{}'.format(width, height)) 
        self.wm_title('Eye Center Prediction')
        self.resizable(width=False, height=False)
        self.width = width
        self.height = height
        # make a frame as containner
        self.container = tk.Frame(self)
        self.container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)  
        self.best_model = best_model
        # all the pages(frames) will be added to this container
        self.frames = {}
        self.init_page(StartPage)
        self.show_frame(StartPage)
        
    def init_page(self, page, *args, **kwargs):
        '''Initialize the pages'''
        self.frames[page] = page(self.container, self, *args, **kwargs)
        #self.frames[page].config(height=height)
        #self.frames[page].config(width=width)
        self.frames[page].grid(row=0, column=0, sticky="nsew")
        
        
    def show_frame(self, page):
        '''Show the pages'''
        self.frames[page].tkraise()
        
    def build_model(self):
        '''Process the data and use the training data to build the prediction model'''
        self.best_model.process_data()
        self.best_model.build_model()
        
        
class StartPage(tk.Frame):
    '''Start page consists of a title, a start button and a smiling face animation'''
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.controller = controller
        self.create_widgets()
    
    def create_widgets(self): 
        self.instruction = tk.Label(self, text='Eye Center \n Recognization')
        self.instruction.config(font=("Courier", 20))
        self.instruction.place(relx=0.5, rely=0.3, anchor=tk.CENTER)
        self.button = tk.Button(self, text = 'START', command=self.click_start)
        self.button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        # make a find eye animation
        self.canvas = tk.Canvas(self)
        self.canvas.place(relx=0.9, rely=0.9, anchor=tk.CENTER)
        self.canvas.create_oval(150, 10, 250, 110, outline="black", width=2)
        self.canvas.create_oval(180, 45, 190, 55, outline="black")
        self.canvas.create_oval(210, 45, 220, 55, outline="black")
        self.canvas.create_arc(180, 70, 220, 90, start=0, extent=-180, outline="red", style=tk.ARC, width=2)
        self.animate()
    
    def draw_eye_centers(self):
        self.eye_center1 = self.canvas.create_oval(185, 50, 185, 50, outline='blue', fill='blue', width=2)
        self.eye_center2 = self.canvas.create_oval(215, 50, 215, 50, outline='blue', fill='green', width=2)
        
    def remove_eye_centers(self):
        self.canvas.delete(self.eye_center1)
        self.canvas.delete(self.eye_center2)
        
    def animate(self):
        '''Draw the eye centers, after 1 second, remove the ete centers. Repeat the animation every 1 second'''
        self.draw_eye_centers()
        self.after(1000, self.remove_eye_centers)
        self.after(2000, self.animate)
     
    def click_start(self):
        '''When click the start button, initialize and show PageOne first.
           The running man starts to run while the model is being built.
           When finish building model, show PageTwo       
        '''
        self.controller.init_page(PageOne)
        self.controller.show_frame(PageOne)
        
        # create a thread to move the running man
        t1 = threading.Thread(target=self.controller.frames[PageOne].moveit)
        t1.start()
        # create another thread to build the model at the same time
        self.t2 = threading.Thread(target=self.controller.build_model)
        self.t2.start()
        # Everyone 25ms, check if the model has been built
        self.check_model(self.t2)
        
        # if do not have running man, use the following codes to show PageOne while building the model    
        #self.controller.after(500, self.controller.process_data)        
        #self.controller.after(500, self.controller.build_model)
        #self.controller.after(500, self.controller.init_page, PageTwo)
        #self.controller.after(500, self.controller.show_frame, PageTwo)
        
    def check_model(self, threading_name):
        while True:
            if not threading_name.isAlive():
                self.controller.init_page(PageTwo)
                self.controller.show_frame(PageTwo)
                break
            else:
                self.after(25, self.check_model, self.t2)
                break # the break is necessary because the check_model function will be run again in 25 ms, but if not break, the while Ture will always loop

       

                    
        
class PageOne(tk.Frame):
    '''PageOne consists a label and a running man animation.
       The running man will continue to run while the model is being built.
       When the model building is done, PageTwo would shown.
    '''
    
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.controller = controller
        self.create_widgets()
    
    def create_widgets(self): 
        self.label = tk.Label(self,text='Creating Model')
        self.label.config(font=("Courier", 18))
        self.label.place(relx=0.4, rely=0.4, anchor=tk.CENTER)        
        self.canvas = tk.Canvas(self, width=self.controller.width)
        self.img1 = tk.PhotoImage(file=r"..\figures\small_images\running_man.png")
        self.img2 = tk.PhotoImage(file=r"..\figures\small_images\running_man_reverse.png")
        self.image_size = 60
        self.image_on_canvas = self.canvas.create_image(0, 0, image=self.img1, anchor='nw') #(0, 0) is the coordicates of the canvas, not of the tkinter window
        self.canvas.place(relx=0, rely=0.5, anchor='nw')


    def moveit(self):
        '''Move the running man 5 pixels horizontally each 0.1 second.
           The running man will flip over when bump into the boundary.
        '''
        direction = 1
        def _moveit(direction):
            # sub function does not need self
            if direction == 1:
                if self.canvas.coords(self.image_on_canvas)[0] < self.controller.width - self.image_size:
                    self.canvas.move(self.image_on_canvas, 5, 0)
                else:
                    # change the image 
                    self.canvas.itemconfig(self.image_on_canvas, image = self.img2)
                    self.canvas.move(self.image_on_canvas, -5, 0)
                    direction = -1                    
            else:
                if self.canvas.coords(self.image_on_canvas)[0] >= 0:
                    self.canvas.move(self.image_on_canvas, -5, 0)
                else:
                    # change the image 
                    self.canvas.itemconfig(self.image_on_canvas, image = self.img1)
                    self.canvas.move(self.image_on_canvas, 5, 0)
                    direction = 1
            self.after(100, _moveit, direction)    
        _moveit(direction)
     
        
class PageTwo(tk.Frame):
    '''PageTwo consists of a label and 40 image buttons'''
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.controller = controller
        self.buttons = {}
        self.create_widgets()
        
    def create_widgets(self):
        self.label = tk.Label(self, text='Choose a facial image')
        self.label.config(font=("Courier", 15))
        self.label.grid(row=0, column=0, columnspan=7, padx=5, pady=15)
        for r in range(11):
            self.rowconfigure(r, weight=1)    
        for c in range(8):
            self.columnconfigure(c, weight=1)
        
        for i in range(40):
            img = tk.PhotoImage(file=r"..\figures\small_images\image_{}.png".format(i))    
            ima_button = ImageButton(self, index = i, image=img)
            self.buttons['button_'.format(i)] = ima_button
            ima_button.config(command=self.buttons['button_'.format(i)].click_button)
            ima_button.image = img # keep a reference!
            ima_button.grid(row=int(i/5)+1, column=i-int(i/5)*5+1, padx=3, pady=3, stick='W')

    
class PageThree(tk.Frame):
    ''' PageThree consists of a 2 labels, 3 checkboxes and a facial images
        The first label is the title; the second label shows the MSE of the predicted results
        The three check boxes represent the eye center locations from the model, the true locations and the benchmark model respectively
        The facial image is drawn by matplotlib.    
    '''
    def __init__(self, parent, controller, index):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.controller = controller
        self.index = index
        self.mse = self.controller.best_model.make_prediction(index=index)
        self.create_widgets()
        self.p1 = None
        
    def create_widgets(self):
        label1 = tk.Label(self, text='Predict Results')
        label1.config(font=("Courier", 15))
        label1.pack()
        
        figure, ax = self.controller.best_model.draw_face(self.index, 96) # figure is the Figure object from matplotlib
        canvas = FigureCanvasTkAgg(figure, self)
        self.ax = ax
        self.canvas = canvas
        
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        button_back = tk.Button(self, text='Make another prediction', command=lambda: self.controller.show_frame(PageTwo))
        button_back.pack(side=tk.BOTTOM)       
        label2 = tk.Label(self, text="MSE: {:.2f}".format(self.mse))
        label2.pack()
        
        check_box1 = CheckBox(self, "Predicted Eye Centers")
        check_box2 = CheckBox(self, "True Eye Centers")
        check_box3 = CheckBox(self, "Benchmark Model")
        
                  
class CheckBox(tk.Checkbutton):
    '''This is the CheckBox class'''
    def __init__(self, parent, text):
        self.cv = tk.IntVar() # var to record the check status of the check box
        self.parent = parent
        self.text=text
        tk.Checkbutton.__init__(self, parent, text=text, variable=self.cv, onvalue=1, offvalue=0, command=self.check_box)
        self.pack()
        self.p = None
        
        
    def check_box(self):
        '''When click the checkbox, draw the corresponding eye centers'''
        if self.text == 'Predicted Eye Centers':
            values = self.parent.controller.best_model.data_pred.loc[self.parent.controller.best_model.data_pred['id'] == self.parent.index]
            color='blue'
        elif self.text == 'True Eye Centers':
            values = self.parent.controller.best_model.test_pos.iloc[self.parent.index]
            color='green'
        else:
            values = self.parent.controller.best_model.mean_pos
            color='red'
        #when the box not checked, draw the dots, when the box is checked, remove the dots   
        if self.cv.get() == 1:
            self.p, = self.parent.ax.plot((values['left_eye_center_x'], values['right_eye_center_x']), (values['left_eye_center_y'], values['right_eye_center_y']), marker='.', color=color, linestyle="None", markersize=10) # assign the 'layer' to a variable so that it can be removed later
            self.parent.canvas.draw() # call the canvas.draw() to show the updated figure
        else:
            if self.p:
                self.p.remove()
                self.parent.canvas.draw()
            
        
class ImageButton(tk.Button):
    '''This is the image button class. 
       When click the button, initialize the corresponding PageThree and show PageThree
    '''
    def __init__(self, parent, index, *args, **kwargs):
        tk.Button.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.index = index
     
    def click_button(self):        
        self.parent.controller.init_page(PageThree, index=self.index)
        self.parent.controller.show_frame(PageThree)

        

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
    
    root = EyeCenterApp(width=450, height=750, best_model=best_model)
    root.mainloop()    
    
    
#Note: for image, 1. use paint to adjust the format; 2. do not forget to keep a reference

    


