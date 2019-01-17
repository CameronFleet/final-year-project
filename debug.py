from tkinter import *
from lunarlander import (LunarLander, LunarLanderContinuous)
from main import demo_heuristic_lander, heuristic
import os
class Application(Frame):

    def createWidgets(self):
        self.start = Button(self)
        self.start["text"] = "Start instance"
        self.start["command"] = lambda : os.system("python main.py nottest")

        self.test = Button(self)
        self.test["text"] = "Test instance"
        self.test["command"] = lambda : os.system("python main.py test")

        self.start.pack({"side": "left"})
        self.test.pack({"side": "left"})


    def __init__(self, master=None):
        Frame.__init__(self, master, width=300, height=700)
        self.pack_propagate(0)
        self.pack()
        self.env = LunarLander()
        self.createWidgets()

root = Tk()
app = Application(master=root)
app.mainloop()
root.destroy()