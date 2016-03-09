import numpy as np
cimport numpy as np

np.import_array()

# -- HEADERS --

cdef extern from 'c_knndaily.h':
    int c_knndaily_run(int nconfig, int nval, int nvar, int nrand,
        int seed,
        int start, int end,
        double * config,
        double * var,
        double * states,
        int * knndaily_idx)

def __cinit__(self):
    pass

def knndaily_run(int seed, int start, int end,
        np.ndarray[double, ndim=1, mode='c'] config not None,
        np.ndarray[double, ndim=2, mode='c'] var not None,
        np.ndarray[double, ndim=1, mode='c'] states not None,
        np.ndarray[int, ndim=1, mode='c'] knndaily_idx not None):

    cdef int ierr

    # check dimensions
    if states.shape[0] != var.shape[1]+1:
        raise ValueError('states.shape[0] != var.shape[1]+1')

    ierr = c_knndaily_run(config.shape[0], \
            var.shape[0], \
            var.shape[1], \
            knndaily_idx.shape[0], \
            seed, start, end,
            <double*> np.PyArray_DATA(config), \
            <double*> np.PyArray_DATA(var), \
            <double*> np.PyArray_DATA(states), \
            <int*> np.PyArray_DATA(knndaily_idx))

    return ierr


