import numpy as np
cimport numpy as np

np.import_array()

# -- HEADERS --
cdef extern from 'c_utils.h':
    int c_utils_geterror(int * esize)

cdef extern from 'c_uh.h':
    int c_uh_getnuhmaxlength()

cdef extern from 'c_uh.h':
    double c_uh_getuheps()

cdef extern from 'c_uh.h':
    int c_uh_getuh(int nuhlengthmax,
            int uhid, 
            double lag,
            int * nuh, 
            double * uh)

def __cinit__(self):
    pass

def uh_getnuhmaxlength():
    return c_uh_getnuhmaxlength()

def uh_getuheps():
    return c_uh_getuheps()

def uh_getuh(int nuhlengthmax, int uhid, double lag,
        np.ndarray[int, ndim=1, mode='c'] nuh not None,
        np.ndarray[double, ndim=1, mode='c'] uh not None):

    cdef int ierr

    ierr = c_uh_getuh(nuhlengthmax,
            uhid, 
            lag,
            <int*> np.PyArray_DATA(nuh),
            <double*> np.PyArray_DATA(uh))

    return ierr


