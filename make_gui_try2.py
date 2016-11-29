from tkinter import *
from PIL import Image, ImageTk



        
def shownext():
    frame.pack_forget()
    frame2 = Frame(root, bg='red')    
    frame2.pack()
    button2 = Button(frame2, text='yes')
    button2.pack()
    #image = Image.open(r"dan4.jpg")
    #photo = ImageTk.PhotoImage(image)
    #label1 = Label(frame2, image=photo) # photo needs to be on a label
    #label1.pack()


if __name__ == '__main__':        
    root = Tk()
    frame = Frame(root, bg='blue')
    frame.pack(fill=X)
    button1 = Button(frame, text='print', command=shownext)
    button1.pack()

    root.mainloop()
