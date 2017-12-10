#cython: language_level=3, boundscheck=False,cdivision=True

from kerrpy_cython.common.metric_utils cimport MetricValues

cdef struct Universe:
    double inner_disk_radius
    double outer_disk_radius
