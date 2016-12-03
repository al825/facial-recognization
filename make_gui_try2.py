import tkinter as tk

from build_model_gui import BestModel

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
        for f in [StartPage, PageOne]:
            self.frames[f] = f(self.container, self)
            self.frames[f].config(height=height)
            self.frames[f].config(width=width)
            self.frames[f].grid(row=0, column=0, sticky="nsew")
        self.show_frame(StartPage)
        
    def show_frame(self, page):
        self.frames[page].tkraise()
        
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.controller = controller
        self.create_widgets()
    
    def create_widgets(self): 
        self.instruction = tk.Label(self, text='Facial Recognization')
        self.instruction.config(font=("Courier", 20))
        self.instruction.place(relx=0.5, rely=0.3, anchor=tk.CENTER)
        self.button = tk.Button(self, text = 'START', command=self.click_start)
        self.button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
     
    def click_start(self):
        self.controller.show_frame(PageOne)
        # model
        best_model = BestModel()
        #self.controlller.show_frame(PageTwo)
                    
        
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
    root = EyeCenterApp(height=500, width=500)
    root.mainloop()


