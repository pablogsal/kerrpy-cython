#cython: language_level=3, boundscheck=False,cdivision=True

from libc.math cimport sin, cos, sqrt
from kerrpy_cython._common.metric_utils cimport MetricValues

cdef MetricValues _compute_metric_values(double a, double r, double theta):
    cdef MetricValues metric_values
    # Shorten long named variables to ease the notation
    a2 = a * a
    r2 = r * r

    # Compute the constants described between (A.1) and (A.2)
    rho = sqrt(r2 + a2 * cos(theta) * cos(theta))
    delta = r2 + a2
    sigma = sqrt((r2 + a2) ** 2 - a2 * delta * sin(theta) ** 2)
    alpha = rho * sqrt(delta) / sigma
    omega = 2 * a * r / (sigma ** 2)

    # Wut? pomega? See https://en.wikipedia.org/wiki/Pi_(letter)#Variant_pi
    pomega = sigma * sin(theta) / rho

    metric_values.a = a
    metric_values.a2 = a2
    metric_values.r = r
    metric_values.r2 = r2
    metric_values.theta = theta
    metric_values.rho = rho
    metric_values.delta = delta
    metric_values.sigma = sigma
    metric_values.alpha = alpha
    metric_values.omega = omega
    metric_values.pomega = pomega

    return metric_values
