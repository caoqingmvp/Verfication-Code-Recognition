import tkinter
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2

top = Tk()
top.title = 'Verfication_Code'
top.geometry('600x250')
canvas = Canvas(top, width=160, height=160, bd=0, bg='white')
canvas.grid(row=1, column=0)


def showImg():

    File = askopenfilename(title='Open Image')
    e.set(File)

    load = Image.open(e.get())

    imgfile = ImageTk.PhotoImage(load)

    canvas.image = imgfile
    canvas.create_image(2, 2, anchor='nw', image=imgfile)


e = StringVar()

submit_button = Button(top, text='Open', command=showImg)
submit_button.grid(row=0, column=0)


def Predict():

    input_ = np.float32(cv2.imread(e.get(), 0))
    input_ = input_.flatten() / 255
    dropout_ = 0.8

    saver = tf.train.import_meta_graph('/Users/caoqingmvp/Pycharm/Verfication_Code/output.model-86000.meta')
    graph = tf.get_default_graph()
    inputs = graph.get_tensor_by_name('input:0')
    dropout = graph.get_tensor_by_name('dropout:0')
    predict_max_idx = graph.get_tensor_by_name('predict_max_idx:0')
    with tf.Session() as sess:
        saver.restore(sess, '/Users/caoqingmvp/Pycharm/Verfication_Code/output.model-86000')
        predict = sess.run(predict_max_idx, feed_dict={inputs: [input_], dropout: [dropout_]})

    predict_vec = predict[0]

    predict_text = []

    for i in predict_vec:
        predict_text.append(i)

    textvar = "The code is : %s" % (predict_text)
    t1.delete(0.0, tkinter.END)
    t1.insert('insert', textvar + '\n')
    t1.update()


submit_button = Button(top, text='Predict', command=Predict)
submit_button.grid(row=0, column=1)

l1 = Label(top, text='Please <Open> an image, then press <Predict> ')
l1.grid(row=2)

t1 = Text(top, bd=0, width=20, height=10, font='Fixdsys -14')
t1.grid(row=1, column=1)
top.mainloop()
