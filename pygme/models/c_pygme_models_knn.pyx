import numpy as np
cimport numpy as np

np.import_array()

# -- HEADERS --

cdef extern from 'c_knn.h':
    int c_knn_run(int nparams, int nval, int nvar, int nrand,
        int idx_select,
        double * params,
        double * weights,
        double * var,
        double * rand,
        int * knn_idx)

def __cinit__(self):
    pass

def knn_run(int idx_select,
        np.ndarray[double, ndim=1, mode='c'] params not None,
        np.ndarray[double, ndim=1, mode='c'] weights not None,
        np.ndarray[double, ndim=2, mode='c'] var not None,
        np.ndarray[double, ndim=1, mode='c'] rand not None,
        np.ndarray[int, ndim=1, mode='c'] knn_idx not None):

    cdef int ierr

    # check dimensions
    if params.shape[0] != 3:
        raise ValueError('params.shape[0] != 3')

    if weights.shape[0] != var.shape[0]:
        raise ValueError('weights.shape[0] != var.shape[0]')

    if rand.shape[0] != knn_idx.shape[0]:
        raise ValueError('rand.shape[0] != knn_idx.shape[0]')

    ierr = c_knn_run(params.shape[0], \
            var.shape[0], \
            var.shape[1], \
            rand.shape[0], \
            idx_select,
            <double*> np.PyArray_DATA(params), \
            <double*> np.PyArray_DATA(weights), \
            <double*> np.PyArray_DATA(var), \
            <double*> np.PyArray_DATA(rand), \
            <int*> np.PyArray_DATA(knn_idx))

    return ierr


