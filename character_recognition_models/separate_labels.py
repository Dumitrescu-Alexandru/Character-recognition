import numpy as np
import matplotlib
from sources.data import Data
from tkinter import *
letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler
root = Tk()

set = Data()
file = open("letsarrange.txt","r")
start = file.read()
start = int(start)
file.close()
i = start - 60000
j = 0
def write_i():
    global i
    global j
    file = open("letsarrange.txt","w")
    file.write(str(j+start))
    file.close()
started = False
def upper_case(*args):
    global started
    fisier = open("aranjate.txt", "a")
    global i
    global j
    print("upper case")
    fisier.write(" 1")
    fisier.close()
    print(i)
    i = i+1
    j = j+1
    if started == True:
        from matplotlib import pyplot
        pyplot.close()
    else:
        started = True
    write_i()
    show_img(set, i)
def lower_thick(*args):
    global started
    fisier = open("aranjate.txt", "a")
    global i
    global j
    print("upper case")
    fisier.write(" 3")
    fisier.close()
    print(i)
    i = i + 1
    j = j + 1
    if started == True:
        from matplotlib import pyplot
        pyplot.close()
    else:
        started = True
    write_i()
    show_img(set, i)

def lower_case(*args):
    global started
    fisier = open("aranjate.txt", "a")
    global i
    global j
    print("lower case")
    fisier.write(" 0")
    print(i)
    fisier.close()
    i = i + 1
    j = j + 1
    write_i()
    if started == True:
        from matplotlib import pyplot
        pyplot.close()
    else:
        started = True
    show_img(set, i)
def lower_semi_thick(*args):
    global started
    fisier = open("aranjate.txt", "a")
    global i
    global j
    print("upper case")
    fisier.write(" 2")
    fisier.close()
    print(i)
    i = i + 1
    j = j + 1
    if started == True:
        from matplotlib import pyplot
        pyplot.close()
    else:
        started = True
    write_i()
    show_img(set, i)
def show_img(date,img_nr):

    print(letters[int(date.letter_labels[i]-1)])
    image = date.letter_images[img_nr]
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()

    ax = fig.add_subplot(1, 1, 1)
    imgplot = ax.imshow(image, cmap=mpl.cm.gray)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.show()
    canvas.close_event(guiEvent=upper_case)
    canvas._tkcanvas.grid(row=0,columnspan=2)



button1 = Button(text='Mare',command=upper_case)
button1.grid(row=1,column=0)


button2 = Button(text="Mic",command=lower_case)
button2.grid(row=1,column=1)

button3 = Button(text='write i',command=write_i)
button3.grid(row=1,column=2)
root.bind("a",func=upper_case)
root.bind("s",func=lower_case)
root.bind("d",func=lower_semi_thick)
root.bind("f",func=lower_thick)
show_img(set,i)

print('asd')
root.mainloop()
root.close()
file = open("letsarrange.txt","w")
file.write(start+i)
# s e small, a e big