#cython: language_level=3, boundscheck=False,cdivision=True
from kerrpy_cython.common.universe cimport Universe

################################################
##                C FUNCTIONS                 ##
################################################

# This functios will be (hopefully) compliled in pure C and they will not have any Python overhead
# of any kind. This means that all the loops and math operations are free of the python-lag.

# ¡IMPORTANT!
# Please, check using cython -a {this_file_name}.pyx that these functions do not have python-related code,
# which is indicated by yellow lines in the html output.
"""
 /**
  * Applies the DOPRI5 algorithm over the system defined in the KerrGeodesicEquations
  * function, using the initial conditions specified in InitCond,
  * and returning the solution found at xend.
  * @param[in,out]  Real*  globalX0     Start of the integration interval
  *                        [x_0, x_{end}]. At the output, this variable is set
  *                        to the final time the multistep_solver reached.
  * @param[in]      Real   xend         End of the integration interval
  *                        [x_0, x_{end}].
  * @param[in,out]  Real*  initCond     Device pointer to a serialized matrix of
  *                        initial conditions; i.e., given a 2D matrix of R rows
  *                        and C columns, where every entry is an n-tuple of
  *                        initial conditions (y_0[0], y_0[1], ..., y_0[n-1]),
  *                        the vector pointed by devInitCond contains R*C*n
  *                        serialized entries, starting with the first row from
  *                        left to right, then the second one in the same order
  *                        and so on.
  *                        The elements of vector pointed by initCond are
  *                        replaced with the new computed values at the end of
  *                        the algorithm; please, make sure you will not need
  *                        them after calling this procedure.
  * @param[in,out]  Real*  hOrig        Step size. This code controls
  *                        automatically the step size, but this value is taken
  *                        as a test for the first try; furthermore, the method
  *                        returns the last computed value of h to let the user
  *                        know the final state of the multistep_solver.
  * @param[in]      Real   hmax         Value of the maximum step size allowed,
  *                        usually defined as x_{end} - x_0, as we do not to
  *                        exceed x_{end} in one iteration.
  * @param[in]      Real*  data         Device pointer to a serialized matrix of
  *                        additional data to be passed to computeComonent;
  *                        currently, this is used to pass the constants b and q
  *                        of each ray to the KerrGeodesicEquations method.
  * @param[out]     int*   iterations   Output variable to know how many
  *                        iterations were spent in the computation
  * @param[in,out]  float* globalFacold Input and output variable, used as a
  *                        first value for facold and to let the caller know the
  *                        final value of facold.
  */
"""
cdef int solver_rk45( double* initCond, double* globalX0, double xend,
                     double* hOrig   , double hmax, double* data,
                     double* globalFacold,
                     Universe* universe,
                     double rtoli   = *,
                     double atoli   = *,
                     double safe    = *,
                     double beta    = *,
                     double uround  = *,
                     double fac1    = *,
                     double fac2    = *) nogil
