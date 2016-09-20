import numpy as np
cimport numpy as np

np.import_array()

# -- HEADERS --
cdef extern from 'c_utils.h':
    int c_utils_daysinmonth(int year, int month)

    int c_utils_dayofyear(int month, int day)

    int c_utils_add1day(int * date)

    int c_utils_add1month(int * date)

    int c_utils_accumulate(int nval, double start,
            int year_monthstart,
            double * inputs, double * outputs)

    int c_utils_root_square_test(int ntest,
        int *niter, int *status, double eps,
        double * roots,
        int nargs, double * args)


cdef extern from 'c_uh.h':
    int c_uh_getnuhmaxlength()

    double c_uh_getuheps()

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


def add1month(np.ndarray[int, ndim=1, mode='c'] date not None):

    cdef int ierr

    ierr = c_utils_add1month(<int*> np.PyArray_DATA(date))

    return ierr


def accumulate(double start, int year_monthstart,
        np.ndarray[double, ndim=1, mode='c'] inputs not None,
        np.ndarray[double, ndim=2, mode='c'] outputs not None):

    cdef int ierr

    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError('inputs.shape[0] != outputs.shape[0]')

    if outputs.shape[1] != 3:
        raise ValueError('outputs.shape[1] != 3')

    ierr = c_utils_accumulate(inputs.shape[0], start,
        year_monthstart,
        <double*> np.PyArray_DATA(inputs),
        <double*> np.PyArray_DATA(outputs))

    return ierr


def root_square_test(int ntest, double eps,
        np.ndarray[int, ndim=1, mode='c'] niter not None,
        np.ndarray[int, ndim=1, mode='c'] status not None,
        np.ndarray[double, ndim=1, mode='c'] roots not None,
        np.ndarray[double, ndim=1, mode='c'] args not None):

    cdef int ierr

    if niter.shape[0] != 1:
        raise ValueError('niter.shape[0] != 1')

    if status.shape[0] != 1:
        raise ValueError('status.shape[0] != 1')

    if roots.shape[0] != 3:
        raise ValueError('roots.shape[0] != 3')

    ierr = c_utils_root_square_test(ntest,
        <int*> np.PyArray_DATA(niter),
        <int*> np.PyArray_DATA(status),
        eps,
        <double*> np.PyArray_DATA(roots),
        args.shape[0],
        <double*> np.PyArray_DATA(args))

    return ierr

