from kerrpy_cython.raytracer import call_kernel
from kerrpy_cython.initial_setup import generate_initial_conditions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import time

def show_image(array):
    from PIL import Image
    cm_hot = mpl.cm.get_cmap('viridis')
    x_scaled = np.uint8(255 * (array - array.min()) / (array.max() - array.min()))
    im = cm_hot(x_scaled)
    im = np.uint8(im * 255)
    im = Image.fromarray(im)
    im.save("kerr_texure.png")

dimension = 1000
camera = {"r": 40, "theta": 1.511, "phi": 0, "focal_lenght": 20, "rows": dimension, "cols": dimension,
           "pixel_heigh": 16 / dimension, "pixel_width": 16 / dimension, "yaw": 0, "roll": 0, "pitch": 0}
universe = {"inner_disk_radius":0,"outer_disk_radius":10,"a":0.5}
x,y = generate_initial_conditions(camera, 0.5,True)

status = np.zeros(dimension*dimension,dtype=np.int32)
t0 = time.perf_counter()
call_kernel(x,y,status,camera,universe,True,False)
print(time.perf_counter()-t0)
status = status.reshape(dimension,dimension)

r = np.array(x[::5]).reshape(dimension,dimension)
theta = np.array(x[1::5]).reshape(dimension,dimension)
phi = np.array(x[2::5]).reshape(dimension,dimension)
pr = np.array(x[3::5]).reshape(dimension,dimension)
ptheta = np.array(x[4::5]).reshape(dimension,dimension)


p1 = np.floor(np.fmod(theta+np.pi, np.pi) * 20.0 / (np.pi)).astype(np.int)
p2 = np.floor(np.fmod(phi+2*np.pi, 2*np.pi) * 20.0 / (2*np.pi)).astype(np.int)
result = (p1 ^ p2) & 1
image = np.full((dimension,dimension),0)
image[(status == 0) & (result == 1)] = 1
p1 = np.floor(np.fmod(theta+np.pi, np.pi) * 8.0 / (np.pi)).astype(np.int)
p2 = np.floor(np.fmod(phi+2*np.pi, 2*np.pi) * 10.0 / (2*np.pi)).astype(np.int)
result = (p1 ^ p2) & 1
image[(status == 2) & (result == 1)] = 4
r_n = (r - 0) / (10.0 - 0)
p1 = (r_n * 4).astype(np.int)
p2 = np.floor(np.fmod(phi+2*np.pi, 2*np.pi) * 26.0 / (2*np.pi)).astype(np.int)
result = (p1 ^ p2) & 1
image[(status == 1) & (result == 1)] = 2
show_image(image)



