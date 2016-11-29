from tkinter import *
from PIL import Image, ImageTk


class BuckyButtons:
    def __init__(self, master):
        self.master = master
        self.frame = Frame(master)
        self.frame.pack()
        
        self.button1 = Button(self.frame, text='Print', command=self.shownext)
        self.button1.pack(side=LEFT)
        

        
    def shownext(self):
        self.frame.pack_forget()
        self.frame2 = Frame(self.master)
        self.frame2.pack()
        image = Image.open(r"dan4.jpg")
        photo = ImageTk.PhotoImage(image)
        self.label1 = Label(self.frame2, image=photo) # photo needs to be on a label
        self.label1.pack()


if __name__ == '__main__':        
    root = Tk()
    bb = BuckyButtons(root)
    root.mainloop()
