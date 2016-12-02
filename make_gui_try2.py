from tkinter import *


class EyeCenterApp(Tk):
    def __init__(self, height, width):
        Tk.__init__(self)
        self.geometry('{}x{}'.format(height, width))
        self.wm_title('Eye Center Prediction')
        self.resizable(width=False, height=False)
        self.container = Frame(self)
        self.container.pack(side = TOP, fill=BOTH, expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)        
        self.frames = {}
        for f in [StartPage, PageOne]:
            self.frames[f] = f(self.container, self)
            self.frames[f].config(height=height)
            self.frames[f].config(width=width)
            self.frames[f].grid(row=0, column=0, sticky="nsew")
        self.show_frame(StartPage)
        
    def show_frame(self, page):
        self.frames[page].tkraise()
        
class StartPage(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self.parent = parent
        self.controller = controller
        self.create_widgets()
    
    def create_widgets(self): 
        self.instruction = Label(self, text='Facial Recognization')
        self.instruction.config(font=("Courier", 20))
        self.instruction.place(relx=0.5, rely=0.3, anchor=CENTER)
        self.button = Button(self, text = 'START', command=self.click_start)
        self.button.place(relx=0.5, rely=0.5, anchor=CENTER)
     
    def click_start(self):
        self.controller.show_frame(PageOne)
        # model
        #self.controlller.show_frame(PageTwo)
                    
        
class PageOne(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self.parent = parent
        self.controller = controller
        self.create_widgets()
    
    def create_widgets(self): 
        self.label = Label(self,text='Creating Model')
        self.label.config(font=("Courier", 30))
        self.label.place(relx=0.5, rely=0.5, anchor=CENTER)
     
        
class PageTwo(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self.parent = parent
        self.controller = controller
        self.create_widgets()
        
    def create_widgets(self):
        self.label = Label(self, text='Predict')
        self.label.config(font=("Courier", 30))
        self.label.place(relx=0.5, rely=0.5, anchor=CENTER)
    
    
        
        
        

if __name__ == '__main__':        
    root = EyeCenterApp(height=500, width=500)
    root.mainloop()


