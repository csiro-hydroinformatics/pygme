import numpy as np
cimport numpy as np

np.import_array()

# -- HEADERS --

cdef extern from 'c_basics.h':
    int c_monthlypattern_run(int nval,
        int nconfig,
        int nstates,
        int noutputs,
        int start, int end,
    	double * config,
    	double * statesini,
        double * outputs)


cdef extern from 'c_basics.h':
    int c_sinuspattern_run(int nval,
        int nconfig,
        int nparams,
        int nstates,
        int noutputs,
        int start, int end,
        double * config,
        double * params,
        double * statesini,
        double * outputs)


def __cinit__(self):
    pass

def monthlypattern_run(int start, int end,
        np.ndarray[double, ndim=1, mode='c'] config not None,
        np.ndarray[double, ndim=1, mode='c'] states not None,
        np.ndarray[double, ndim=2, mode='c'] outputs not None):

    cdef int ierr

    # check dimensions
    if states.shape[0] < 1:
        raise ValueError('states.shape[0] < 1')

    if config.shape[0] != 12:
        raise ValueError('config.shape[0] != 12')

    # Run model
    ierr = c_monthlypattern_run(outputs.shape[0],
            config.shape[0], \
            states.shape[0], \
            outputs.shape[1], \
            start, end,
            <double*> np.PyArray_DATA(config), \
            <double*> np.PyArray_DATA(states), \
            <double*> np.PyArray_DATA(outputs))

    return ierr


def sinuspattern_run(int start, int end,
        np.ndarray[double, ndim=1, mode='c'] config not None,
        np.ndarray[double, ndim=1, mode='c'] params not None,
        np.ndarray[double, ndim=1, mode='c'] states not None,
        np.ndarray[double, ndim=2, mode='c'] outputs not None):

    cdef int ierr

    # check dimensions
    if config.shape[0] < 1:
        raise ValueError('config.shape[0] < 2')

    if states.shape[0] < 2:
        raise ValueError('states.shape[0] < 2')

    if params.shape[0] != 4:
        raise ValueError('params.shape[0] != 4')

    # Run model
    ierr = c_sinuspattern_run(outputs.shape[0],
            config.shape[0],
            params.shape[0],
            states.shape[0], 
            outputs.shape[1], 
            start, end,
            <double*> np.PyArray_DATA(config), 
            <double*> np.PyArray_DATA(params), 
            <double*> np.PyArray_DATA(states), 
            <double*> np.PyArray_DATA(outputs))

    return ierr


