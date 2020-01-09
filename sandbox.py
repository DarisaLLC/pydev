
def lazy_property(fn):
    '''Decorator that makes a property lazy-evaluated.
    '''
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


class Person:
    def __init__(self, name, occupation):
        self.name = name
        self.occupation = occupation

    @lazy_property
    def relatives(self):
        # Get all relatives
        relatives = 'relatives'
        return relatives


if __name__ == '__main__':
    p = Person ('foo', 'bar')
    print(p.relatives)
from tkinter import *
from tkinter import filedialog
import tkinter
import os

class Chooser:
    def __init__(self, master):
        self.resolution = StringVar()
        self.resolution.set("360p")
        self.filename = StringVar()

        label = Label(master, text="Select Target Resolution")
        label.pack(fill=X, padx=10, pady=10)

        option_res = OptionMenu(master, self.resolution, "240p", "360p", "480p", "720p", "1080p", command=self.selected)
        option_res.pack(padx=10, pady=10)

        button_choose = Button(master, text='Choose File', command=self.choose_file)
        button_choose.pack(fill=X, padx=10, pady=40)

        button_start = Button(master, text='Start', command=self.start)
        button_start.pack(fill=X, padx=5)

    def selected(self, res):
        print ("Resolution " + self.resolution.get() + " is chosen.")

    def choose_file(self):
        self.filename = filedialog.askopenfilename(initialdir = "/home",  filetypes=[('All', '*'), ('mp4','*.mp4'), ('mpeg', '*mpeg')])
        print ("File " + self.filename + " is chosen.")

    def start(self):
        print ("Resizer Converting your video...")
        #print "../core/cmake-build-debug/VideoResAdjuster " + self.filename + " " + self.resolution.get()
        os.system("../core/cmake-build-debug/VideoResAdjuster " + self.filename + " " + self.resolution.get())
        print ("Finished.")


master = Tk()
master.minsize(width=320, height=240)
master.title("Video Resolution Adjuster")
chooser = Chooser(master)

mainloop()