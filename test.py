from kerrpy_cython.raytracer import call_kernel
from kerrpy_cython.initial_setup import get_initial_conditions
import numpy as np
import matplotlib.pyplot as plt

import time

dimension = 1000
camera = {"r": 40, "theta": 1.511, "phi": 0, "focal_lenght": 20, "rows": dimension, "cols": dimension,
           "pixel_heigh": 16 / dimension, "pixel_width": 16 / dimension, "yaw": 0, "roll": 0, "pitch": 0}
x,y = get_initial_conditions(camera, 0.9,True)

status = np.zeros(dimension*dimension,dtype=np.int32)
t0 = time.perf_counter()
call_kernel(x,y,status,camera,0.9,True,False)
print(time.perf_counter()-t0)
plt.imshow(status.reshape(dimension,dimension))
plt.show()



