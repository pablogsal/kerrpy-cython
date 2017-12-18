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
universe = {"inner_disk_radius":0,"outer_disk_radius":10,"a":0}
x,y = generate_initial_conditions(camera, 0,True)

status = np.zeros(dimension*dimension,dtype=np.int32)
t0 = time.perf_counter()
iterations = call_kernel(x,y,status,camera,universe,True,False)
print(time.perf_counter()-t0)
status = np.array(iterations).reshape(dimension,dimension)
status[status > 300] = 0
plt.imshow(status)
plt.show()

