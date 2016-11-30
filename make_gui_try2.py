from tkinter import *
        
class FrameOne(Frame):
    def __init__(self, master):
        Frame.__init__(self,master)
        self['width'] = 500
        self['height'] = 500
        self.pack(fill='both')
        self.create_widgets()
    
    def create_widgets(self): 
        self.instruction = Label(self, text='Facial Recognization')
        self.instruction.config(font=("Courier", 20))
        self.instruction.place(relx=0.5, rely=0.3, anchor=CENTER)
        self.button = Button(self, text = 'START', command=self.click_start)
        self.button.place(relx=0.5, rely=0.5, anchor=CENTER)
     
    def click_start(self):
        self.pack_forget()
        self.next_frame=FrameTwo(self.master)
        
class FrameTwo(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.config(width=500, height=500)
        self.pack(fill='both')
        self.create_widges()
        
    def create_widges(self):
        self.label = Label(text='Creating Model')
        self.label.config(font=("Courier", 30))
        self.label.place(relx=0.5, rely=0.5, anchor=CENTER)
        

if __name__ == '__main__':        
    root = Tk()
    root.geometry("500x500")
    root.resizable(width=False, height=False)
    frame1 = FrameOne(root)
    root.mainloop()

'''
# create the root window
root = Tk()
# optionally give it a title
root.title("My Title")
# set the root window's height, width and x,y position
# x and y are the coordinates of the upper left corner
w = 300
h = 200
x = 50
y = 100
# use width x height + x_offset + y_offset (no spaces!)
root.geometry("%dx%d+%d+%d" % (w, h, x, y))
# use a colorful frame
frame = Frame(root, bg='green')
frame.pack(fill='both', expand='yes')
# position a label on the frame using place(x, y)
# place(x=0, y=0) would be the upper left frame corner
label = Label(frame, text="Hello Python Programmer!")
label.place(x=20, y=30)
# put the button below the label, change y coordinate
button = Button(frame, text="Press me", bg='yellow')
button.place(x=20, y=60)
root.mainloop()
'''  
