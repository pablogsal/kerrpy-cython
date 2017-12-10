#cython: language_level=3, boundscheck=False,cdivision=True

from kerrpy_cython.common.metric_utils cimport MetricValues

cdef struct Camera:
    double r
    double theta
    double phi
    double roll
    double pitch
    double yaw
    double pixel_heigh
    double pixel_width
    long rows
    long cols
    double focal_lenght
    double beta

cdef Camera compute_camera_values(dict camera_values, MetricValues*metric)
