import cython
from libc.math cimport cos,sin

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # tuern off zerodivisioncheck
cdef void KerrGeodesicEquations(double* y, double* f,double* data) nogil:
    """
    This function computes the right hand side of the Kerr geodesic equations described
    in http://arxiv.org/abs/1502.03808.

    All the computations are highly optimized to avoid calculating the same term twice.

    :param y: pointer to double
        The current coordinate values to compute the equations. ( Will not be modified)
        The array must follow the following convention for the variables:
            0 -> r
            1 -> theta
            2 -> phi
            3 -> p_r
            4 -> p_theta

    :param f: pointer to double
        The place where the values of the equations will be stored.
        The array will follow this convention:
            0 -> d(r)/dt
            1 -> d(theta)/dt
            2 -> d(phi)/dt
            3 -> d(p_r)/dt
            4 -> d(p_theta)/dt

            Where t is the independent variable (propper time).

    :param data: pointer to double
        Aditional data needed for the computation. Explicitly:

            0 -> b ( Angular momentum)
            1 -> q ( Carter's constant)
            2 -> a ( Black Hole spin)
            3 -> e ( Energy )
    """
    # Variables to hold the position of the ray, its momenta and related
    # operations between them and the constant a, which is the spin of the
    # black hole.
    cdef double r, r2, twor, theta, pR, pR2, pTheta, pTheta2, b, twob, b2, q, bMinusA, bMinusAE,a, a2
    # Variables to hold the sine and cosine of theta, along with some
    # operations with them
    cdef double sinT, cosT, sinT2, sinT2Inv, cosT2

    # Variables to hold the value of the functions P, R, Delta (which is
    # called D), Theta (which is called Z) and rho, along with some operations
    # involving these values.
    cdef double P, R, D, Dinv, Z, DZplusR, rho2Inv, twoRho2Inv, rho4Inv

    # Variable for the energy

    cdef double energy
    cdef double energy2

    # Retrieval of the input data (position of the ray, momenta and
    # constants).
    r = y[0]
    theta = y[1]
    pR = y[3]
    pTheta = y[4]

    # Computation of the square of r, widely used in the computations.
    r2 = r*r

    # Sine and cosine of theta, as well as their squares and inverses.
    sinT = sin(theta)
    cosT = cos(theta)
    sinT2 = sinT*sinT
    sinT2Inv = 1/sinT2
    cosT2 = cosT*cosT

    # Retrieval of the constants data: b and q, along with the computation of
    # the square of b and the number b - a, repeateadly used throughout the
    # computation
    b = data[0]
    q = data[1]
    a = data[2]
    energy = data[3]
    energy2 = energy * energy

    a2 = a*a
    b2 = b*b
    bMinusA = b - a
    bMinusAE = b - a * energy

    # Commonly used variables: R, D, Theta (that is called Z) and
    # rho (and its square and cube).
    D = r2 - 2*r + a2
    Dinv = 1/D

    P = ( a2 + r2 ) * energy - a * b
    R = P*P - D*(bMinusAE*bMinusAE + q )
    Z = q - cosT2*(b2*sinT2Inv - energy2 *  a2)

    rho2Inv = 1/(r2 + a2*cosT2)
    twoRho2Inv = rho2Inv/2
    rho4Inv = rho2Inv*rho2Inv

    # Squares of the momenta components
    pR2 = pR*pR
    pTheta2 = pTheta*pTheta

    # Double b and double r, that's it! :)
    twob = 2*b
    twor = 2*r

    # Declaration of variables used in the actual computation: dR, dZ, dRho
    # and dD will store the derivatives of the corresponding functions (with
    # respect to the corresponding variable in each thread). The sumX values
    # are used as intermediate steps in the final computations, in order to
    # ease notation.
    cdef double dR, dZ, dRhoTimesRho, dD, sum1, sum2, sum3, sum4, sum5, sum6

    # *********************** EQUATION 1 *********************** //
    f[0] = D * pR * rho2Inv

    # *********************** EQUATION 2 *********************** //
    f[1] = pTheta * rho2Inv

    # *********************** EQUATION 3 *********************** //
    # Derivatives with respect to b
    dR = -2.0 * D * bMinusAE + (-2.0) * a * P
    dZ = - twob * cosT2 * sinT2Inv

    f[2] = - (dR + D*dZ)*Dinv*twoRho2Inv

    # *********************** EQUATION 4 *********************** //
    # Derivatives with respect to r
    dD = twor - 2
    dR = 4.0 * r * energy * P  - ( q + bMinusAE * bMinusAE ) * ( twor - 2.0 )
    DZplusR = D*Z + R

    sum1 = + pTheta2
    sum2 = + D*pR2
    sum3 = - (DZplusR * Dinv)
    sum4 = - (dD*pR2)
    sum5 = + (dD*Z + dR) * Dinv
    sum6 = - (dD*DZplusR * Dinv * Dinv)

    f[3] = r*(sum1 + sum2 + sum3)*rho4Inv + (sum4 + sum5 + sum6)*twoRho2Inv
    # *********************** EQUATION 5 *********************** //
    # Derivatives with respect to theta (called z here)
    dRhoTimesRho = - a2*cosT*sinT

    cdef double cosT3 = cosT2*cosT
    cdef double sinT3 = sinT2*sinT

    dZ = - 2 * ( Z - q ) / cosT * sinT + (2*b2*cosT3)/(sinT3)

    sum1 = + pTheta2
    sum2 = + D*pR2
    sum3 = - DZplusR * Dinv
    sum4 = + dZ * twoRho2Inv

    f[4] = dRhoTimesRho*(sum1 + sum2 + sum3)*rho4Inv + sum4