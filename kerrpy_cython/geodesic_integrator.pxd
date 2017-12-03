#cython: language_level=3, boundscheck=False,cdivision=True


################################################
##                C FUNCTIONS                 ##
################################################

# This functios will be (hopefully) compliled in pure C and they will not have any Python overhead
# of any kind. This means that all the loops and math operations are free of the python-lag.

# ¡IMPORTANT!
# Please, check using cython -a {this_file_name}.pyx that these functions do not have python-related code,
# which is indicated by yellow lines in the html output.
cdef int SolverRK45( double* initCond, double* globalX0, double xend,
                     double* hOrig   , double hmax, double* data,
                     double* globalFacold,
                     double rtoli   = *,
                     double atoli   = *,
                     double safe    = *,
                     double beta    = *,
                     double uround  = *,
                     double fac1    = *,
                     double fac2    = *) nogil
