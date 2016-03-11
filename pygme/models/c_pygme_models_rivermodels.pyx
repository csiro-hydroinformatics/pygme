import numpy as np
cimport numpy as np

np.import_array()

# -- HEADERS --

cdef extern from 'c_demand.h':
    int c_demand_run(int nval,
        int nconfig,
        int ninputs,
        int nstates,
        int noutputs,
        int start, int end,
    	double * config,
    	double * inputs,
    	double * statesini,
        double * outputs)

def __cinit__(self):
    pass

def demand_run(int start, int end,
        np.ndarray[double, ndim=1, mode='c'] config not None,
        np.ndarray[double, ndim=2, mode='c'] inputs not None,
        np.ndarray[double, ndim=1, mode='c'] states not None,
        np.ndarray[double, ndim=2, mode='c'] outputs not None):

    cdef int ierr

    # check dimensions
    if states.shape[0] < 2:
        raise ValueError('states.shape[0] < 2')

    if config.shape[0] != 12:
        raise ValueError('config.shape[0] != 12')

    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError('inputs.shape[0] != outputs.shape[0]')

    if inputs.shape[1] != 2:
        raise ValueError('inputs.shape[1] != 2')

    # Run model
    ierr = c_demand_run(inputs.shape[0],
            config.shape[0], \
            inputs.shape[1], \
            states.shape[0], \
            outputs.shape[1], \
            start, end,
            <double*> np.PyArray_DATA(config), \
            <double*> np.PyArray_DATA(inputs), \
            <double*> np.PyArray_DATA(states), \
            <double*> np.PyArray_DATA(outputs))

    return ierr

