import numpy as np
cimport numpy as np

np.import_array()

# -- HEADERS --
cdef extern from 'c_sac15.h':
    int c_sac15_run(int nval, int nparams,
            int nuh,
            int ninputs,
            int nstates, int noutputs,
            int start, int end,
            double * params,
            double * uh,
            double * inputs,
            double * statesuh,
            double * states,
            double * outputs)

def sac15_run(int nuh,
        int start, int end,
        np.ndarray[double, ndim=1, mode='c'] params not None,
        np.ndarray[double, ndim=1, mode='c'] uh not None,
        np.ndarray[double, ndim=2, mode='c'] inputs not None,
        np.ndarray[double, ndim=1, mode='c'] statesuh not None,
        np.ndarray[double, ndim=1, mode='c'] states not None,
        np.ndarray[double, ndim=2, mode='c'] outputs not None):

    cdef int ierr

    # check dimensions
    if params.shape[0] != 15:
        raise ValueError('params.shape[0] != 15')

    if states.shape[0] < 6:
        raise ValueError('states.shape[0] < 6')

    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError('inputs.shape[0] != outputs.shape[0]')

    if inputs.shape[1] != 2:
        raise ValueError('inputs.shape[1] != 2')

    if uh.shape[0] < nuh:
        raise ValueError('uh.shape[0] < nuh')

    # Run model
    ierr = c_sac15_run(inputs.shape[0],
            params.shape[0], \
            nuh,
            inputs.shape[1], \
            states.shape[0], \
            outputs.shape[1], \
            start, end,
            <double*> np.PyArray_DATA(params), \
            <double*> np.PyArray_DATA(uh), \
            <double*> np.PyArray_DATA(inputs), \
            <double*> np.PyArray_DATA(statesuh), \
            <double*> np.PyArray_DATA(states), \
            <double*> np.PyArray_DATA(outputs))

    return ierr


