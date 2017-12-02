#cython: language_level=3, boundscheck=False,cdivision=True

from libc.math cimport exp, sin, cos, sqrt, atan

from kerrpy_cython._common.metric_utils cimport MetricValues
from kerrpy_cython._common.camera cimport Camera

cdef Camera _compute_camera_values(dict camera_values, MetricValues* metric):

    cdef Camera camera

    camera.r = camera_values["r"]
    camera.theta = camera_values["theta"]
    camera.phi = camera_values["phi"]
    camera.roll = camera_values["roll"]
    camera.pitch = camera_values["pitch"]
    camera.yaw = camera_values["yaw"]
    camera.pixel_heigh = camera_values["pixel_heigh"]
    camera.pixel_width = camera_values["pixel_width"]
    camera.rows = camera_values["rows"]
    camera.cols = camera_values["cols"]
    camera.focal_lenght = camera_values["focal_lenght"]

    cdef double Omega

    # Define speed with equation (A.7)
    Omega = 1. / (metric.a + pow(camera.r,(3. / 2.)))
    camera.beta = metric.pomega * (Omega - metric.omega) / metric.alpha

    # FIXME: This is being forced to zero only for testing purposes.
    # Remove this line if you want some real fancy images.
    camera.beta = 0

    return camera