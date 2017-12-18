#cython: language_level=3, boundscheck=False,cdivision=True

from libc.math cimport sin, cos, sqrt

cdef struct MetricValues:
    double a
    double a2
    double r
    double r2
    double theta
    double rho
    double delta
    double sigma
    double alpha
    double omega
    double pomega

cdef MetricValues compute_metric_values(double a, double r, double theta)
