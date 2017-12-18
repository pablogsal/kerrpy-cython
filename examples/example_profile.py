from kerrpy_cython.raytracer import call_kernel
from kerrpy_cython.initial_setup import generate_initial_conditions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import time
import pstats, cProfile

dimension = 100
camera = {"r": 40, "theta": 1.511, "phi": 0, "focal_lenght": 20, "rows": dimension, "cols": dimension,
           "pixel_heigh": 16 / dimension, "pixel_width": 16 / dimension, "yaw": 0, "roll": 0, "pitch": 0}
universe = {"inner_disk_radius":0,"outer_disk_radius":10,"a":0.5}
x,y = generate_initial_conditions(camera, 0.5,True)

status = np.zeros(dimension*dimension,dtype=np.int32)

cProfile.runctx("call_kernel(x,y,status,camera,universe,False,False)", globals(), locals(), "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats("pyx:")
# s.strip_dirs().sort_stats("time").print_stats("kerr_equations.pyx")
