import numpy as np
cimport numpy as np

np.import_array()

# -- HEADERS --
cdef extern from 'c_utils.h':
    int c_utils_daysinmonth(int year, int month)

cdef extern from 'c_utils.h':
    int c_utils_dayofyear(int month, int day)

cdef extern from 'c_utils.h':
    int c_utils_add1day(int * date)

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


def daysinmonth(int year, int month):
    return c_utils_daysinmonth(year, month)


def dayofyear(int month, int day):
    return c_utils_dayofyear(month, day)


def add1day(np.ndarray[int, ndim=1, mode='c'] date not None):

    cdef int ierr

    ierr = c_utils_add1day(<int*> np.PyArray_DATA(date))

    return ierr



