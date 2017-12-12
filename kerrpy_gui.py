#!/usr/bin/env/python
import sys
from PyQt5 import QtGui, QtWidgets, QtCore

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import random
import numpy as np
import time
import sys

from kerrpy_cython.raytracer import call_kernel
from kerrpy_cython.initial_setup import generate_initial_conditions

class Window(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        self.ax = self.figure.add_subplot(111)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.progress = QtWidgets.QProgressBar(self)

        # set the layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.progress)
        self.setLayout(layout)

        self.cosa = np.random.rand(500,500)

        import threading
        dimension = 1000
        camera = {"r": 40, "theta": 1.511, "phi": 0, "focal_lenght": 20, "rows": dimension, "cols": dimension,
                   "pixel_heigh": 16 / dimension, "pixel_width": 16 / dimension, "yaw": 0, "roll": 0, "pitch": 0}
        self.x,self.y = generate_initial_conditions(camera, 0.5,True)
        self.status = np.full(dimension*dimension,-1,dtype=np.int32)
        self.event = threading.Event()

        self.p = threading.Thread(target=self.plot,args=(self.event,))
        self.p.daemon = False
        self.p.start()


        self.o = threading.Thread(target=self.change,args=(self.event,))
        self.o.start()

    def plot(self,event):
        while not event.isSet():
            ''' plot some random stuff '''
            # plot data
            dimension = 1000
            status = self.status.reshape(dimension,dimension)
            x,y = self.x,self.y
            r = np.array(x[::5]).reshape(dimension,dimension)
            theta = np.array(x[1::5]).reshape(dimension,dimension)
            phi = np.array(x[2::5]).reshape(dimension,dimension)
            pr = np.array(x[3::5]).reshape(dimension,dimension)
            ptheta = np.array(x[4::5]).reshape(dimension,dimension)

            percentage = (np.count_nonzero(np.array(status) != -1)/status.size) * 100
            self.progress.setValue(percentage)

            # p1 = np.floor(np.fmod(theta+np.pi, np.pi) * 20.0 / (np.pi)).astype(np.int)
            # p2 = np.floor(np.fmod(phi+2*np.pi, 2*np.pi) * 20.0 / (2*np.pi)).astype(np.int)
            # result = (p1 ^ p2) & 1
            # image = np.full((dimension,dimension),0)
            # image[(status == 0) & (result == 1)] = 1
            # p1 = np.floor(np.fmod(theta+np.pi, np.pi) * 8.0 / (np.pi)).astype(np.int)
            # p2 = np.floor(np.fmod(phi+2*np.pi, 2*np.pi) * 10.0 / (2*np.pi)).astype(np.int)
            # result = (p1 ^ p2) & 1
            # image[(status == 2) & (result == 1)] = 0
            # r_n = (r - 0) / (10.0 - 0)
            # p1 = (r_n * 4).astype(np.int)
            # p2 = np.floor(np.fmod(phi+2*np.pi, 2*np.pi) * 26.0 / (2*np.pi)).astype(np.int)
            # result = (p1 ^ p2) & 1
            # image[(status == 1) & (result == 1)] = 2

            self.ax.imshow(theta)
            # refresh canvas
            self.canvas.draw()

    def change(self,event):
        dimension = 1000
        universe = {"inner_disk_radius":0,"outer_disk_radius":10,"a":0.5}
        camera = {"r": 40, "theta": 1.511, "phi": 0, "focal_lenght": 20, "rows": dimension, "cols": dimension,
                   "pixel_heigh": 16 / dimension, "pixel_width": 16 / dimension, "yaw": 0, "roll": 0, "pitch": 0}
        t0 = time.time()
        call_kernel(self.x,self.y,self.status,camera,universe,True,False)
        print(time.time()-t0)
        time.sleep(.5)
        event.set()


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    main = Window()
    main.show()
    sys.exit(app.exec_())
