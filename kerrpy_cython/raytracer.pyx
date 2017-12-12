from libc.math cimport *
from cython.parallel cimport prange

import numpy as np

from kerrpy_cython.common.camera cimport Camera, compute_camera_values
from kerrpy_cython.common.metric_utils cimport MetricValues, compute_metric_values
from kerrpy_cython.common.universe cimport Universe
from kerrpy_cython.geodesic_integrator cimport solver_rk45

import tqdm

from libc.stdio cimport printf
from libc.stdlib cimport malloc, calloc

################################################
##            GLOBAL DEFINITIONS              ##
################################################

cdef int SPHERE = 0
cdef int DISK = 1
cdef int HORIZON = 2

DEF SYSTEM_SIZE = 5
DEF DATA_SIZE = 4

################################################
##         PYTHON-ACCESIBLE FUNCTIONS         ##
################################################

# Notice that these functions HAVE overhead because they interface with python code and
# python objects will be constructed and unpacked each time the function is summoned.

def call_kernel(double[:] init_conditions, double[:] data, int[:] status,
                dict camera_values, dict universe_values,
                parallel=False, progress=False):

    cdef MetricValues metric = compute_metric_values(
        universe_values["a"], camera_values["r"], camera_values["theta"])

    cdef Camera camera = compute_camera_values(camera_values, &metric)

    cdef Universe universe
    universe.inner_disk_radius = universe_values["inner_disk_radius"]
    universe.outer_disk_radius = universe_values["outer_disk_radius"]

    cdef int image_rows = camera_values["rows"]
    cdef int image_cols = camera_values["cols"]

    cdef unsigned int[:] iterations = np.zeros(camera.rows * camera.cols,dtype=np.uint32)
    if parallel:
        if progress:
            kernel_parallel_progress(0, -150, init_conditions, -0.001, -150, data, status, &camera, &universe, iterations)
        else:
            kernel_parallel(0, -150, init_conditions, -0.001, -150, data, status, &camera, &universe, iterations)
    else:
        kernel_serial(0, -150, init_conditions, -0.001, -150, data, status, &camera, &universe, iterations)

    return iterations

################################################
##                C FUNCTIONS                 ##
################################################

# This functios will be (hopefully) compliled in pure C and they will not have any Python overhead
# of any kind. This means that all the loops and math operations are free of the python-lag.

# ¡IMPORTANT!
# Please, check using cython -a {this_file_name}.pyx that these functions do not have python-related code,
# which is indicated by yellow lines in the html output.

cdef void kernel_serial(double x0, double xend, double[:] initial_conditions, double h,
                       double hmax, double[:] data, int[:] status,
                       Camera* camera, Universe* universe, unsigned int[:] iterations):
    cdef int row, col, pixel
    for row in tqdm.trange(camera.rows):
        for col in tqdm.trange(camera.cols):
            pixel =  col+camera.cols*row
            kernel(x0, xend, &initial_conditions[pixel * SYSTEM_SIZE],h, hmax,
                   &data[pixel * DATA_SIZE], &status[pixel], camera, universe,
                   &iterations[pixel])


cdef void kernel_parallel_progress(double x0, double xend, double[:] initial_conditions, double h,
                       double hmax, double[:] data, int[:] status,
                       Camera* camera, Universe* universe, unsigned int[:] iterations):
    cdef int row, col, pixel
    for row in tqdm.trange(camera.rows):
        for col in prange(camera.cols, nogil=True, schedule="guided"):
            pixel =  col+camera.cols*row
            kernel(x0, xend, &initial_conditions[pixel * SYSTEM_SIZE],h, hmax,
                   &data[pixel * DATA_SIZE], &status[pixel], camera, universe,
                   &iterations[pixel])



cdef void kernel_parallel(double x0, double xend, double[:] initial_conditions, double h,
                       double hmax, double[:] data, int[:] status,
                       Camera* camera, Universe* universe, unsigned int[:] iterations):
    cdef int row, col, pixel
    for pixel in prange(camera.rows * camera.cols,nogil=True, schedule="guided"):
        kernel(x0, xend, &initial_conditions[pixel * SYSTEM_SIZE],h, hmax,
               &data[pixel * DATA_SIZE], &status[pixel], camera, universe,
               &iterations[pixel])

cdef void kernel(double x0, double xend, double* pixel_init_conditions, double h,
                       double hmax, double* pixel_data, int* pixel_status,
                       Camera* camera, Universe* universe, unsigned int* iterations) nogil:
    # Compute pixel's row and col of this thread
    cdef int local_status
    cdef double x
    cdef double facold = 1.0e-4 #TODO Allow the user to set this

    # Status flag: at the output, the (x,y)-th element will be
    # set to SPHERE, HORIZON or disk, showing the final state of the ray.
    local_status = pixel_status[0]

    # Integrate the ray only if it's still in the sphere. If it has
    # collided either with the disk or within the horizon, it is not
    # necessary to integrate it anymore.
    if local_status == SPHERE or local_status == -1:
        # Current time
        x = x0

        # MAIN ROUTINE. Integrate the ray from x to xend, checking disk
        # collisions on the go with the following algorithm:
        #   -> 0. Check that the ray has not collided with the disk or
        #   with the horizon and that the current time has not exceeded
        #   the final time.
        #   -> 1. Advance the ray a step, calling the main RK45 solver.
        #   -> 2. Test whether the ray has collided with the horizon.
        #          2.1 If the answer to the 2. test is negative: test
        #          whether the current theta has crossed theta = pi/2,
        #          and call bisect in case it did, updating its status
        #          accordingly (set it to DISK if the ray collided with
        #          the horizon).
        #          2.2. If the answer to the 2. test is positive: update
        #          the status of the ray to HORIZON.
        local_status = solver_rk45(pixel_init_conditions, &x, xend, &h, xend - x,
                                   pixel_data, &facold, universe, iterations)

        # Update the global status variable with the new computed status
        pixel_status[0] = local_status

