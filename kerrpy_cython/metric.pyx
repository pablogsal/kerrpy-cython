#cython: language_level=3, boundscheck=False,cdivision=True

from libc.math cimport *
import numpy as np
cimport numpy as np

################################################
##         PYTHON-ACCESIBLE FUNCTIONS         ##
################################################

# Notice that these functions HAVE overhead because they interface with python code and
# python objects will be constructed and unpacked each time the function is summoned.

cpdef np.ndarray[np.float64_t, ndim=2] metric(double r, double theta, double a):

    # Create output matrix

    cdef np.ndarray[np.float64_t, ndim=2] kerr_metric = np.zeros((4,4))

    # Calculate values

    calculate_metric(r, theta, a , kerr_metric)

    return kerr_metric

cpdef np.ndarray[np.float64_t, ndim=2] inverse_metric(double r, double theta, double a):

    # Create output matrix

    cdef np.ndarray[np.float64_t, ndim=2] kerr_metric = np.zeros((4,4))

    # Calculate values

    calculate_inverse_metric(r, theta, a , kerr_metric)

    return kerr_metric

cpdef double kretschmann(double r, double theta, double a):
    return calculate_Kretschmann(r, theta, a)

################################################
##                C FUNCTIONS                 ##
################################################

# This functios will be (hopefully) compliled in pure C and they will not have any Python overhead
# of any kind. This means that all the loops and math operations are free of the python-lag.

# ¡IMPORTANT!
# Please, check using cython -a {this_file_name}.pyx that these functions do not have python-related code,
# which is indicated by yellow lines in the html output.


cdef void calculate_inverse_metric(double r, double theta,
        double a, double [:,:] kerr_metric):

    cdef double r2 = r*r
    cdef double a2 = a*a
    cdef double sintheta2 = sin(theta) * sin(theta)

    cdef double ro = sqrt(r2 + a2 * cos(theta)**2)
    cdef double delta = r2 - 2*r + a2
    cdef double sigma = sqrt( (r2 + a2) * (r2 + a2) - a2 * delta * sintheta2)
    cdef double alpha = ro * sqrt(delta) / sigma
    cdef double omega = 2.0 * a * r / (sigma * sigma)
    cdef double pomega = sigma * sin(theta) / ro

    # Compute auxiliar products

    cdef double ro2 = ro * ro
    cdef double ro2inv = 1.0 / ro2
    cdef double deltainv = 1.0 / delta
    cdef double pomega2 = pomega*pomega
    cdef double omega2 = omega * omega
    # Construct the nonzero components of the metric

    kerr_metric[0,0] = - deltainv * ( r2 + a2 + a * pomega2 * omega)
    kerr_metric[1,1] = delta * ro2inv
    kerr_metric[2,2] = ro2inv
    kerr_metric[3,3] = ( delta / sintheta2 - a2  ) * ro2inv * deltainv
    kerr_metric[0,3] = - 2.0 * a * r * ro2inv * deltainv
    kerr_metric[3,0] = kerr_metric[0,3]


cdef void calculate_metric(double r, double theta,
        double a, double [:,:] kerr_metric):

    # Calculate metric quantities 

    cdef double r2 = r*r
    cdef double a2 = a*a

    cdef double ro = sqrt(r2 + a2 * cos(theta)**2)
    cdef double delta = r2 - 2*r + a2
    cdef double sigma = sqrt( (r2 + a2) * (r2 + a2) - a2 * delta * sin(theta)*sin(theta))
    cdef double alpha = ro * sqrt(delta) / sigma
    cdef double omega = 2.0 * a * r / (sigma * sigma)
    cdef double pomega = sigma * sin(theta) / ro

    # Compute auxiliar products

    cdef double ro2 = ro * ro
    cdef double pomega2 = pomega*pomega
    cdef double omega2 = omega * omega
    # Construct the nonzero components of the metric

    kerr_metric[0,0] = - alpha * alpha + pomega2 * omega2
    kerr_metric[1,1] = ro2 / delta
    kerr_metric[2,2] = ro2
    kerr_metric[3,3] = pomega2
    kerr_metric[0,3] = - pomega2 * omega
    kerr_metric[3,0] = kerr_metric[0,3]

cdef double calculate_Kretschmann( double r, double theta, double a):

    cdef double acostheta2 = ( a * cos(theta) ) * ( a * cos(theta) )
    cdef double acostheta4 = acostheta2 * acostheta2
    cdef double acostheta6 = acostheta4 * acostheta2
    cdef double r2 = r*r
    cdef double r4 = r2 * r2
    cdef double r6 = r4 * r2
    
    cdef double numerator =  48.0 * ( 15.0 * acostheta4 - acostheta6 - 15.0 * r4 * acostheta2 + r6)
    cdef double denom     =  acostheta2 + r2
    denom          =  denom * denom * denom * denom * denom * denom
    return numerator / denom

cpdef double calculate_temporal_component(double [:] three_momenta,
                                  double [:] three_position,
                                  double a, int causality):
    """
    Calculate the temporal component of the momenta given the value of the three-momenta and
    its position in the spacetime.

    :param three_position: np.array[:]
        A numpy array representing the initial position of the geodesic in a time slice of
        the spacetime.

        It must be given in the form

        three_position = [r, theta, phi]

        where {r,theta,phi} are the Boyer-Lindquist coordinates.

    :param thee_momenta: np.array[:]
        A numpy array representing the initial momenta ( covariant tangent vector) of the
        geodesic in a time slice of the spacetime.

        It must be given in the form

        three_momenta = [pr, ptheta, pphi]

        where {pr, ptheta, pphi} are the covariant components of the tangent three-vector in
        Boyer-Lindquist coordinates.

    :param causality: int
        The causal character of the geodesic tangent vector.

            - (-1) for timelike geodesics.
            -   0  for spacelike geodesics.
    """

    cdef double r = three_position[0]
    cdef double theta = three_position[1]
    cdef double phi = three_position[2]
    cdef double pr = three_momenta[0]
    cdef double ptheta = three_momenta[1]
    cdef double pphi = three_momenta[2]
    # Calculate common terms

    cdef double r2 = r*r
    cdef double a2 = a*a

    cdef double ro = sqrt(r2 + a2 * cos(theta)**2)
    cdef double delta = r2 - 2*r + a2
    cdef double sigma = sqrt((r2 + a2)**2 - a2 * delta * sin(theta)**2)
    cdef double pomega = sigma * sin(theta) / ro
    cdef double omega = 2 * a * r / ( sigma * sigma )
    cdef double alpha = ro *  sqrt(delta) / sigma

    cdef double result = 0
    if causality == 0:
        result =  - sqrt( pphi * pphi * ro * ro +
                      ( ptheta * ptheta + pr * pr * delta )
                       * pomega * pomega)
        result *= alpha
        result /= ro * pomega
        result -=  pphi * omega

        return result
    else:
        result = - sqrt( pphi * pphi * ro * ro +
                      ( ptheta * ptheta + pr * pr * delta + ro * ro )
                       * pomega * pomega)
        result *= alpha
        result /= ro * pomega
        result -=  pphi * omega

        return result

