import cython
import numpy as np
cimport numpy as np
from libc.math cimport *
from libc.string cimport memcpy

from kerrpy_cython.kerr_equations cimport KerrGeodesicEquations
from kerrpy_cython.metric cimport calculate_temporal_component
from kerrpy_cython.common.universe cimport Universe

cdef extern from "stdlib.h" nogil:
    double fabs (double number)

################################################
##            GLOBAL DEFINITIONS              ##
################################################

####  Butcher's tableau coefficients   ####

# These coefficients are needed for the RK45 multistep_solver to work.
# When calculating the different samples for the derivative, each
# sample is weighted differently according to some coefficients: These
# numbers the weighting coefficients. Each Runge-Kutta method has its
# coefficients, which in turns define the method completely and these
# are the coefficients of Dopri5, an adaptative RK45 schema.

# For more information on this matter:

# https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods

cdef int SPHERE = 0
cdef int DISK = 1
cdef int HORIZON = 2


cdef double A21 = (1./5.)

cdef double A31 = (3./40.)
cdef double A32 = (9./40.)

cdef double A41 = (44./45.)
cdef double A42 = (- 56./15.)
cdef double A43 = (32./9.)

cdef double A51 = (19372./6561.)
cdef double A52 = (- 25360./2187.)
cdef double A53 = (64448./6561.)
cdef double A54 = (- 212./729.)

cdef double A61 = (9017./3168.)
cdef double A62 = (- 355./33.)
cdef double A63 = (46732./5247.)
cdef double A64 = (49./176.)
cdef double A65 = (- 5103./18656.)

cdef double A71 = (35./384.)
cdef double A72 = 0
cdef double A73 = (500./1113.)
cdef double A74 = (125./192.)
cdef double A75 = (- 2187./6784.)
cdef double A76 = (11./84.)

cdef double C2 = (1./5.)
cdef double C3 = (3./10.)
cdef double C4 = (4./5.)
cdef double C5 = (8./9.)
cdef double C6 = 1
cdef double C7 = 1

cdef double E1 = (71./57600.)
cdef double E2 = 0
cdef double E3 = (- 71./16695.)
cdef double E4 = (71./1920.)
cdef double E5 = (- 17253./339200.)
cdef double E6 = (22./525.)
cdef double E7 = (- 1./40.)


#### System Size ####

# We are not constructing a general-pourpose integrator, instead we know that we
# want to integrate Kerr's geodesics. The system of differential equations for
# our version of the geodesic equations ( our version <-> the Hamiltonian we are
# using ) has 5 equations: 3 for the coordinates and 2 for the momenta because
# the third momenta equation vanishes explicitly. 

DEF SYSTEM_SIZE = 5


################################################
##         PYTHON-ACCESIBLE FUNCTIONS         ##
################################################

# Notice that these functions HAVE overhead because they interface with python code and
# python objects will be constructed and unpacked each time the function is summoned.

### Integrate ray ###
cpdef np.ndarray[np.float64_t, ndim=2] integrate_ray(double [:] three_position,
                                                     double [:] three_momenta  ,
                                                     int   causality, double a,
                                                     double x0, double xend, int n_steps):
    """
    Integrate a geodesic in Kerr spacetime provided the initial data.

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
        
        Its only effect is the calculus of the energy (the temporal component of the momenta).

    :param a: double
        The spin of the Kerr Black hole as it appear in the metric.

    :param x0: double
        The initial value of the proper time.
    :param xend: double
        The ending value of the proper time
    :param n_steps: int
        The number of points to sample along the geodesic.

        Note: This does NOT include the initial point.
    """
    cdef double r = three_position[0]
    cdef double theta = three_position[1]
    cdef double phi = three_position[2]
    cdef double pR = three_momenta[0]
    cdef double pTheta = three_momenta[1]
    cdef double pPhi = three_momenta[2]


    # Calculate the temporal component of the momenta ( the energy )

    cdef double energy = calculate_temporal_component(three_momenta, three_position, a, causality)
    # Set conserved quantities. See (A.12)

    cdef double b = pPhi
    cdef double q = pTheta**2 + cos(theta)**2*(b**2 / sin(theta)**2 - a * a)
    # Store the initial conditions in all the pixels of the systemState array.

    cdef np.ndarray[np.float64_t, ndim=2] result = np.zeros((n_steps+1,5))
    cdef np.ndarray[np.float64_t, ndim=1] init = np.array([r, theta, phi, pR, pTheta])
    cdef np.ndarray[np.float64_t, ndim=1] data = np.array([b,q,a,energy]) 
   
    multistep_solver(x0, xend, n_steps, init, data, result, NULL) #TODO: Change null for universe

    return result


### Integrate Camera ray ###

# This functions constructs the initial conditions for a null geodesic and integrate
# this light ray. Its here mainly for testing the integrator against Mathematica's NDSOLVE.

cpdef np.ndarray[np.float64_t, ndim=2] test_integrate_camera_ray(double r, double cam_theta,
                                                     double cam_phi, double theta_cs,
                                                     double phi_cs, double a, double causality, int n_steps):

    # Simplify notation
    cdef double theta = cam_theta
    cdef double a2 = a*a
    cdef double r2 = r*r

    # Calculate initial vector direction

    cdef double Nx = sin(theta_cs) * cos(phi_cs)
    cdef double Ny = sin(theta_cs) * sin(phi_cs)
    cdef double Nz = cos(theta_cs)

    # Convert the direction of motion to the FIDO's spherical orthonormal
    # basis. See (A.10)

    #TODO: Fix this mess.
    # IMPORTANT: This is not computed as in (A.10) because the MATHEMATICA DATA
    # has been generated without the aberration computation. Sorry for that!

    cdef double nR = Nx
    cdef double nTheta = Nz
    cdef double nPhi = Ny

    # Get canonical momenta

    cdef double ro = sqrt(r2 + a2 * cos(theta)**2)
    cdef double delta = r2 - 2*r + a2
    cdef double sigma = sqrt((r2 + a2)**2 - a2 * delta * sin(theta)**2)
    cdef double pomega = sigma * sin(theta) / ro

    # Compute energy as measured by the FIDO. See (A.11)

    # TODO: Fix this mess
    # IMPORTANT: This is not computed as in (A.11) because the MATHEMATICA DATA
    # has been generated with this quantity as 1. Sorry for that!
    cdef double E = 1.0

    # Compute the canonical momenta. See (A.11)
    cdef double pR = E * ro * nR / sqrt(delta)
    cdef double pTheta = E * ro * nTheta
    cdef double pPhi = E * pomega * nPhi

    # Calculate the conserved quantities b and q.

    # Set conserved quantities. See (A.12)
    cdef double b = pPhi
    cdef double q = pTheta**2 + cos(theta)**2*(b**2 / sin(theta)**2 - a2)

    # Store the initial conditions in all the pixels of the systemState array.


    cdef double x0 = 0.0
    cdef double xend = -30.0
    cdef np.ndarray[np.float64_t, ndim=2] result = np.zeros((n_steps+1,5))
    cdef np.ndarray[np.float64_t, ndim=1] init = np.array([r, cam_theta, cam_phi, pR, pTheta])
    cdef np.ndarray[np.float64_t, ndim=1] data = np.array([b,q,a,E]) 
    multistep_solver(x0, xend, n_steps, init, data, result, NULL)

    return result



################################################
##                C FUNCTIONS                 ##
################################################

# This functios will be (hopefully) compliled in pure C and they will not have any Python overhead
# of any kind. This means that all the loops and math operations are free of the python-lag.

# ¡IMPORTANT!
# Please, check using cython -a {this_file_name}.pyx that these functions do not have python-related code,
# which is indicated by yellow lines in the html output.

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # tuern off zerodivisioncheck
cdef void multistep_solver(double x, double xend, int n_steps,
                 double [:] initCond,double[:] data,
                 double [:,:] result, Universe* universe):
    """
    This function acts as an interface with the RK45 multistep_solver. Its pourpose is avoid the
    overhead of calling the Runge-Kutta multistep_solver multiple times from python, which creates
    and unpacks a lot of python objects. The caller must pass a result buffer ( a python
    memoryview) and this function will populate the buffer with the result of successive
    calls to the Runge-Kutta multistep_solver.

    :param x: double
        The initial point of the independent variable for the integration
    :param xend: double
        The ending point of the independent variable for the integration.
    :param n_steps: int
        The number of steps to make when performing the integration.
        ¡IMPORTANT! This number has to be larger than the dimension of the `result` buffer.
    :param initCond: memoryview
        The initial conditions at the point x.
        ¡IMPORTANT! The content of this memoryview will be overwritten by the multistep_solver.
    :param data: memoryview
        Aditional data needed for computing the right hand side of the equations to integrate.
    :param result: memoryview (2D)
        A buffer to store the result of the integration in each RK45 step.
        ¡IMPORTANT! The first dimension of the buffer must be larger or equal than n_steps.
    """
    cdef double hOrig = 0.01
    cdef double globalFacold = 1.0e-4
    cdef double step = (xend - x) / n_steps
    cdef int current_step, i
    
    # data to them. This may seem a bit odd but is here because the function KerrGeodesicEquations
    # is called with the initial data to compute K1 and with the calculated data to compute K2,...,Kn.
    # As the type of the initCond variable is a memoryview we would need to use another memoryview to
    # match the signature of the function but to do this we would need to allocate memory in the heap
    # and we do NOT want to do this. So we use arrays of known size ( 5 ) and therefore we cannot use the
    # memoryview that this function recieves. So we create ONCE two arrays and copy the data into them.
    # Notice that this is done ONCE in contrast to the option of doing whatever alternative in the RK45Solver
    # that will be called n_steps times, so the overhead is minimum with this option.

    cdef double initial_conditions[5]
    cdef double aditional_data[4]

    for i in range(5): # TODO: SYSTEM_SIZE
        initial_conditions[i] = initCond[i]

    for i in range(4):
        aditional_data[i] = data[i]

    # Store initial step conditions

    result[0,0] = initial_conditions[0]
    result[0,1] = initial_conditions[1]
    result[0,2] = initial_conditions[2]
    result[0,3] = initial_conditions[3]
    result[0,4] = initial_conditions[4]

    cdef unsigned int iterations;
    for current_step in range(n_steps):
        solver_rk45(initial_conditions, &x, x + step, &hOrig, 0.1,
                    aditional_data, &globalFacold, universe, &iterations)

        # Store the result of the integration in the buffer

        result[current_step + 1,0] = initial_conditions[0]
        result[current_step + 1,1] = initial_conditions[1]
        result[current_step + 1,2] = initial_conditions[2]
        result[current_step + 1,3] = initial_conditions[3]
        result[current_step + 1,4] = initial_conditions[4]

cdef int solver_rk45(double* init_cond, double* initial_x0, double xend,
                     double* h_orig, double hmax, double* data,
                     double* initial_fac_old,
                     Universe* universe,
                     unsigned int* iterations,
                     double rtoli   = 1e-06,
                     double atoli   = 1e-12,
                     double safe    = 0.9,
                     double beta    = 0.04,
                     double uround  = 2.3e-16,
                     double fac1    = 0.2,
                     double fac2    = 10.0) nogil:
    ################################
    ##### Configuration vars #######
    ################################
    cdef double safeInv = 1.0 / safe
    cdef double fac1_inverse = 1.0 / fac1
    cdef double fac2_inverse = 1.0 / fac2

    cdef double inner_disk_radius = 0
    cdef double outer_disk_radius = 0

    if universe != NULL:
        inner_disk_radius = universe.inner_disk_radius
        outer_disk_radius = universe.outer_disk_radius

    
    #################################
    #####  Variable definitions #####
    #################################

    # Declare a counter for the loops
    cdef int i
    
    # Loop variable to manage the automatic step size detection.
    cdef double hnew
    
    # Retrieve the value of h and the value of x0
    cdef double h = h_orig[0]
    cdef double x0 = initial_x0[0]

    # Check the direction of the integration: to the future or to the past
    # and get the absolute value of the maximum step size.
    cdef double integration_direction = +1. if xend - x0 > 0. else -1.
    hmax = fabs(hmax)

    # Auxiliar array to store the intermediate calls to the
    # KerrGeodesicEquations function

    cdef double y1[SYSTEM_SIZE]

    # Auxiliary variables used to compute the errors at each step.
    
    cdef float sqr                 # Scaled differences in each eq.
    cdef float err = 0             # Global error of the step
    cdef float sk                  # Scale based on the tolerances

    # Initial values for the step size automatic prediction variables.
    # They are basically factors to maintain the new step size in known
    # bounds, but you can see the corresponding chunk of code far below to
    # know more about the puropose of each of these variables.

    cdef float fac_old = initial_fac_old[0]
    cdef float expo1 = 0.2 - beta * 0.75
    cdef float fac11, fac

    # Loop variables initialisation. The current step is repeated when
    # `reject` is set to true, event that happens when the global error
    # estimation exceeds 1.

    cdef int reject = False  # TODO: Avisar a alejandro de que esto esta como double

    # Variables to keep track of the current r and the previous and
    # current theta
    cdef double currentR;
    cdef int prev_theta_sign, current_theta_sign;

    # Initialize previous theta to the initial conditions
    prev_theta_sign = sign(init_cond[1] - M_PI / 2);

    # Local variable to know how many iterations spent the bisect in the
    # current step.
    cdef int bisect_iter = 0;

    # Initial status of the ray: SPHERE
    cdef int status = SPHERE

    cdef float horizon_radius = 2.0
    cdef int last = False


    while x0 > xend:

        # Check that the step size is not too small and that the horizon is
        # not too near. Although the last condition belongs to the raytracer
        # logic, it HAS to be checked here.
        
        if 0.1 * fabs(h) <= fabs(x0) * uround and not last: #TODO Set the check for the horizon
            h_orig[0] = h
            return HORIZON

        # PHASE 0. Check if the current time x_0 plus the current step
        # (multiplied by a safety factor to prevent steps too small)
        # exceeds the end time x_{end}.

        if (x0 + 1.01 * h - xend) * integration_direction > 0.0:
            h = xend - x0
            last = True
        
        err = advance_step(init_cond, y1, data, h, atoli, rtoli)

        # For full information about the step size computation, please see
        # equation (4.13) and its surroundings in [1] and the notes in
        # Section IV.2 in [2].
        # Mainly, the new step size is computed from the previous one and
        # the current error in order to assure a high probability of
        # having an acceptable error in the next step. Furthermore, safe
        # factors and minimum/maximum factors are taken into account.
        # The stabilization of the step size behaviour is done with the
        # variable beta (expo1 depends only of beta), taking into account
        # the previous accepted error

        # Stabilization computations:
        fac11 = pow(err, expo1)
        fac = fac11 / pow(fac_old, beta)
        # We need the multiplying factor (always taking into account the
        # safe factor) to be between fac1 and fac2 i.e., we require
        # fac1 <= h_new/h <= fac2:
        fac = fmax(fac2_inverse, fmin(fac1_inverse, fac * safeInv))
        # New step final (but temporary) computation
        h_new = h / fac

        # PHASE 3. Check whether the current step has to be repeated,
        # depending on its estimated error:

        # Check whether the normalized error, err, is below or over 1.:
        # PHASE 3.1: REJECT STEP if err > 1
        if err > 1.0:

            # Stabilization technique with the minimum and safe factors
            #  when the step is rejected.
            h_new = h / fmin(fac1_inverse, fac11 * safeInv)
            reject = True

        # PHASE 3.2: ACCEPT STEP if err <= 1.
        else:

            # Update old factor to new current error (upper bounded to 1e-4)
            fac_old = fmax(err, 1.0e-4)

            # Advance current time!
            x0 += h

            # Assure the new step size does not exceeds the provided
            # bounds.

            if fabs(h_new) > hmax:
                h_new = integration_direction * hmax

            # If the previous step was rejected, take the minimum of the
            # old and new step sizes

            if reject:
                h_new = integration_direction * fmin(fabs(h_new), fabs(h))

            # Necessary update for next steps: the local initCond variable holds
            # the current initial condition (now the computed solution)

            memcpy(init_cond, y1, sizeof(double) * SYSTEM_SIZE)

            # This step was accepted, so it was not rejected, so reject is
            # false. SCIENCE.

            reject = False

            current_theta_sign = sign(y1[1] - M_PI / 2);

            if prev_theta_sign != current_theta_sign:
                bisect_iter += bisect(y1, data, h, x0, atoli, rtoli)

                # Retrieve the current r
                current_r = y1[0]

                # Finally, check whether the current r is inside the disk,
                # updating the status and copying back the data in the
                # case it is.
                if inner_disk_radius< current_r < outer_disk_radius:
                    memcpy(init_cond, y1, sizeof(double) * SYSTEM_SIZE)
                    status = DISK
                    break

            #Update the previous variable for the next step computation
            prev_theta_sign = current_theta_sign;

        # Final step size update!

        h = h_new
        iterations[0] += 1

    # END WHILE LOOP

    # Update the user's h, fac_old and x0
    
    h_orig[0] = h
    initial_fac_old[0] = fac_old
    initial_x0[0] = x0
    
    return status


cdef double advance_step(double* initCond, double* y1, double* data,
                         double h, double atoli, double rtoli) nogil:
    # Local error of each eq.
    cdef float errors[SYSTEM_SIZE]
    # Auxiliar arrays to store the intermediate K1, ..., K7 computations
    cdef double k1[SYSTEM_SIZE]
    cdef double k2[SYSTEM_SIZE]
    cdef double k3[SYSTEM_SIZE]
    cdef double k4[SYSTEM_SIZE]
    cdef double k5[SYSTEM_SIZE]
    cdef double k6[SYSTEM_SIZE]
    cdef double k7[SYSTEM_SIZE]
    # K1 computation
    KerrGeodesicEquations(initCond, k1, data)
    # K2 computation
    for i in range(SYSTEM_SIZE):
        y1[i] = initCond[i] + h * A21 * k1[i]
    KerrGeodesicEquations(y1, k2, data)
    # K3 computation
    for i in range(SYSTEM_SIZE):
        y1[i] = initCond[i] + h * (A31 * k1[i] + A32 * k2[i])
    KerrGeodesicEquations(y1, k3, data)
    # K4 computation
    for i in range(SYSTEM_SIZE):
        y1[i] = initCond[i] + h * (A41 * k1[i] +
                                   A42 * k2[i] +
                                   A43 * k3[i])
    KerrGeodesicEquations(y1, k4, data)
    # K5 computation
    for i in range(SYSTEM_SIZE):
        y1[i] = initCond[i] + h * (A51 * k1[i] +
                                   A52 * k2[i] +
                                   A53 * k3[i] +
                                   A54 * k4[i])
    KerrGeodesicEquations(y1, k5, data)
    # K6 computation
    for i in range(SYSTEM_SIZE):
        y1[i] = initCond[i] + h * (A61 * k1[i] +
                                   A62 * k2[i] +
                                   A63 * k3[i] +
                                   A64 * k4[i] +
                                   A65 * k5[i])
    KerrGeodesicEquations(y1, k6, data)
    # K7 computation
    for i in range(SYSTEM_SIZE):
        y1[i] = initCond[i] + h * (A71 * k1[i] +
                                   A73 * k3[i] +
                                   A74 * k4[i] +
                                   A75 * k5[i] +
                                   A76 * k6[i])
    KerrGeodesicEquations(y1, k7, data)
    # The Butcher's table (Table 5.2, [1]), shows that the estimated
    # solution has exactly the same coefficients as the ones used to
    # compute K7. Then, the solution is the last computed y1!
    # The local error of each equation is computed as the difference
    # between the solution y and the higher order solution \hat{y}, as
    # specified in the last two rows of the Butcher's table (Table
    # 5.2, [1]). Instead of computing \hat{y} and then substract it
    # from y, the differences between the coefficientes of each
    # solution have been computed and the error is directly obtained
    # using them:
    for i in range(SYSTEM_SIZE):
        errors[i] = h * (E1 * k1[i] +
                         E3 * k3[i] +
                         E4 * k4[i] +
                         E5 * k5[i] +
                         E6 * k6[i] +
                         E7 * k7[i])
    cdef float err = 0
    for i in range(SYSTEM_SIZE):
        # The local estimated error has to satisfy the following
        # condition: |err[i]| < Atol[i] + Rtol*max(|y_0[i]|, |y_j[i]|)
        # (see equation (4.10), [1]). The variable sk stores the right
        # hand size of this inequality to use it as a scale in the local
        # error computation this way we "normalize" the error and we can
        # compare it against 1.
        sk = atoli + rtoli * fmax(fabs(initCond[i]), fabs(y1[i]))

        # Compute the square of the local estimated error (scaled with the
        # previous factor), as the global error will be computed as in
        # equation 4.11 ([1]): the square root of the mean of the squared
        # local scaled errors.
        sqr = (errors[i]) / sk
        errors[i] = sqr * sqr
        err += errors[i]
    # The sum of the local squared errors in now in errors[0], but the
    # global error is the square root of the mean of those local
    # errors: we finish here the computation and store it in err.
    err = sqrt(err / SYSTEM_SIZE)  # TODO: SYSTEM_SIZE
    return err


cdef int bisect(double* yOriginal, double* data, double first_step,
                double x, double atoli, double rtoli) nogil:

    cdef double BISECT_TOL = 0.000001 #TODO Make this configurable
    cdef int BISECT_MAX_ITER = 100 #TODO Make this configurable

    # It is necessary to maintain the previous theta to know the direction
    # change; we'll store it centered in zero, and not in pi/2 in order to
    # removes some useless substractions in the main loop.
    cdef double prevThetaCentered, currentThetaCentered
    prevThetaCentered = yOriginal[1] - M_PI / 2

    # The first step shall be to the other side and half of its length.
    cdef double step = - first_step * 0.5

    # Loop variables, to control that the iterations does not exceed a maximum
    # number
    cdef int iterations = 0

    # Array used by advanceStep() routine, which expects a pointer where the
    # computed new state should be stored
    cdef double yNew[5]

    # This loop implements the main behaviour of the algorithm basically,
    # this is how it works:
    #    1. It advance the point one single step with the RK45 algorithm.
    #    2. If theta has crossed pi/2, it changes the direction of the
    #    new step. The magnitude of the new step is always half of the
    #    magnitude of the previous one.
    #    3. It repeats 1 and 2 until the current theta is very near of Pi/2
    #    ("very near" is defined by BISECT_TOL) or until the number of
    #    iterations exceeds a maximum number previously defined.
    while fabs(prevThetaCentered) > BISECT_TOL and iterations < BISECT_MAX_ITER:
        # 1. Advance the ray one step.
        advance_step(yOriginal,yNew, data,step, atoli, rtoli)
        memcpy(yOriginal, yNew, sizeof(double)*SYSTEM_SIZE)
        x += step

        # Compute the current theta, centered in zero
        currentThetaCentered = yOriginal[1] - M_PI/2

        # 2. Change the step direction whenever theta crosses the target,
        # pi/2, and make it half of the previous one.
        step = step * sign(currentThetaCentered)*sign(prevThetaCentered) * 0.5

        # Update the previous theta, centered in zero, with the current one
        prevThetaCentered = currentThetaCentered

        iterations+=1

    # Return the number of iterations spent in the loop
    return iterations

################################################
##                C FUNCTIONS                 ##
################################################
"""
/**
 * Returns the sign of `x`; i.e., it returns +1 if x >= 0 and -1 otherwise.
 * @param  x The number whose sign has to be returned
 * @return   Sign of `x`, considering 0 as positive.
 */
"""
cdef inline int sign(double x) nogil:
    return -1 if x < 0 else +1
