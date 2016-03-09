import numpy as np
cimport numpy as np

np.import_array()

# -- HEADERS --

cdef extern from 'c_knndaily.h':
    int c_knndaily_run(int nparams, int nval, int nvar, int nrand,
        int start, int end,
        double * params,
        double * rand,
        double * var,
        double * states,
        int * knn_idx)

def __cinit__(self):
    pass

def knndaily_run(int start, int end,
        np.ndarray[double, ndim=1, mode='c'] config not None,
        np.ndarray[double, ndim=2, mode='c'] rand not None,
        np.ndarray[double, ndim=2, mode='c'] var not None,
        np.ndarray[double, ndim=1, mode='c'] states not None,
        np.ndarray[int, ndim=1, mode='c'] knn_idx not None):

    cdef int ierr

    # check dimensions
    if rand.shape[0] != var.shape[0]:
        raise ValueError('states.shape[0] != var.shape[1]+1')

    if rand.shape[1] != 1:
        raise ValueError('rand.shape[1] != 1')

    if states.shape[0] != var.shape[1]+1:
        raise ValueError('states.shape[0] != var.shape[1]+1')

    ierr = c_knndaily_run(config.shape[0], \
            var.shape[0], \
            var.shape[1], \
            knn_idx.shape[0], \
            start, end,
            <double*> np.PyArray_DATA(config), \
            <double*> np.PyArray_DATA(rand), \
            <double*> np.PyArray_DATA(var), \
            <double*> np.PyArray_DATA(states), \
            <int*> np.PyArray_DATA(knn_idx))

    return ierr


