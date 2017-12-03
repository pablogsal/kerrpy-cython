#cython: language_level=3, boundscheck=False,cdivision=True

import numpy as np

from cython.parallel cimport prange
cimport numpy as np
from libc.math cimport M_PI as pi
from libc.math cimport *

from kerrpy_cython._common.camera cimport Camera, _compute_camera_values
from kerrpy_cython._common.metric_utils cimport MetricValues, _compute_metric_values

################################################
##            GLOBAL DEFINITIONS              ##
################################################


################################################
##         PYTHON-ACCESIBLE FUNCTIONS         ##
################################################

# Notice that these functions HAVE overhead because they interface with python code and
# python objects will be constructed and unpacked each time the function is summoned.

cpdef get_initial_conditions(dict camera_values, double a, parallel=False):
    cdef MetricValues metric = _compute_metric_values(a, camera_values["r"], camera_values["theta"])
    cdef Camera camera = _compute_camera_values(camera_values, &metric)

    cdef int image_rows = camera_values["rows"]
    cdef int image_cols = camera_values["cols"]
    cdef double[:] initial_conditions = np.zeros(image_rows*image_cols*5)
    cdef double[:] constants = np.zeros(image_rows*image_cols*4)

    if parallel:
        _get_initial_conditions_parallel(initial_conditions, constants, &camera, &metric)
    else:
        _get_initial_conditions(initial_conditions, constants, &camera, &metric)
    return initial_conditions,constants


################################################
##                C FUNCTIONS                 ##
################################################

# This functios will be (hopefully) compliled in pure C and they will not have any Python overhead
# of any kind. This means that all the loops and math operations are free of the python-lag.

# ¡IMPORTANT!
# Please, check using cython -a this_file_name.pyx that these functions do not have python-related code,
# which is indicated by yellow lines in the html output.

cdef void _get_initial_conditions(double[:] initial_conditions, double[:] constants,
                                  Camera* camera, MetricValues* metric):
    cdef int row, col, pixel
    for row in range(camera.rows):
        for col in range(camera.cols):
            pixel =  col+camera.cols*row
            setInitialConditions(&initial_conditions[pixel * 5], &constants[pixel * 4], row, col, camera, metric)

cdef void _get_initial_conditions_parallel(double[:] initial_conditions, double[:] constants,
                                      Camera* camera, MetricValues* metric):
    cdef int row, col, pixel
    for row in prange(camera.rows, nogil=True, schedule="guided"):
        for col in range(camera.cols):
            pixel =  col+camera.cols*row
            setInitialConditions(&initial_conditions[pixel * 5], &constants[pixel * 4], row, col, camera, metric)

cdef void setInitialConditions(double* initial_conditions, double* constants,
                           int row, int col, Camera* camera, MetricValues* metric) nogil:
    cdef double pR, pTheta, pPhi, b, q

    # Compute pixel position in the physical space
    cdef double x = - (col + 0.5 - camera.cols/2) * camera.pixel_width
    cdef double y = (row + 0.5 - camera.rows/2) * camera.pixel_heigh

    # Compute direction of the incoming ray in the camera's reference
    # frame
    cdef double rayPhi = pi + atan(x / camera.focal_lenght)
    cdef double rayTheta = pi/2 + atan(y / sqrt(camera.focal_lenght*camera.focal_lenght + x*x))

    # Compute canonical momenta of the ray and the conserved quantites
    # b and q
    getCanonicalMomenta(rayTheta, rayPhi, &pR, &pTheta, &pPhi, metric, camera)

    getConservedQuantities(pTheta, pPhi, camera.theta, metric.a, &b, &q)

    # Save ray's initial conditions
    initial_conditions[0] = camera.r
    initial_conditions[1] = camera.theta
    initial_conditions[2] = camera.phi
    initial_conditions[3] = pR
    initial_conditions[4] = pTheta

    # Save ray's constants
    constants[0] = b
    constants[1] = q
    constants[2] = metric.a
    constants[3] = 1.0

cdef void getCanonicalMomenta(double  rayTheta, double  rayPhi,
                              double *pR, double *pTheta, double *pPhi,
                              MetricValues* metric, Camera* camera) nogil:
    # **************************** SET NORMAL **************************** #
    # Cartesian components of the unit vector N pointing in the direction of
    # the incoming ray
    cdef double  Nx = sin(rayTheta) * cos(rayPhi)
    cdef double  Ny = sin(rayTheta) * sin(rayPhi)
    cdef double  Nz = cos(rayTheta)

    # ********************** SET DIRECTION OF MOTION ********************** #
    # Compute denominator, common to all the cartesian components
    cdef double  den = 1. - camera.beta * Ny

    # Compute factor common to nx and nz
    cdef double  fac = -sqrt(1. - camera.beta * camera.beta)

    # Compute cartesian coordinates of the direction of motion. See(A.9)
    cdef double  nY = (-Ny + camera.beta) / den
    cdef double  nX = fac * Nx / den
    cdef double  nZ = fac * Nz / den

    # Convert the direction of motion to the FIDO's spherical orthonormal
    # basis. See (A.10)
    cdef double  nR = nX
    cdef double  nTheta = -nZ
    cdef double  nPhi = nY

    # *********************** SET CANONICAL MOMENTA *********************** #
    # Compute energy as measured by the FIDO. See (A.11)
    cdef double  E = 1. / (metric.alpha + metric.omega * metric.pomega * nPhi)

    # Set conserved energy to unity. See (A.11)
    # cdef double  pt = -1

    # Compute the canonical momenta. See (A.11)
    pR[0] = E * metric.rho * nR / sqrt(metric.delta)
    pTheta[0] = E * metric.rho * nTheta
    pPhi[0] = E * metric.pomega * nPhi

cdef void getConservedQuantities(double  pTheta, double  pPhi,
                                 double  theta, double a, double*b, double*q) nogil:
    # ********************* GET CONSERVED QUANTITIES ********************* #
    # Get conserved quantities. See (A.12)
    b[0] = pPhi
    cdef double a2 = a * a

    cdef double  sinT = sin(theta)
    cdef double  sinT2 = sinT * sinT

    cdef double  cosT = cos(theta)
    cdef double  cosT2 = cosT * cosT

    cdef double  pTheta2 = pTheta * pTheta
    cdef double  b2 = pPhi * pPhi

    q[0] = pTheta2 + cosT2 * ((b2 / sinT2) - a2)
