import numpy as np
cimport numpy as np

np.import_array()

# -- HEADERS --

cdef extern from 'c_gr2m.h':
    int c_gr2m_run(int nval, int nconfig, int nparams, int ninputs,
            int nstates, int noutputs,
            int start, int end,
            double * config,
            double * params,
            double * inputs,
            double * statesini,
            double * outputs)

cdef extern from 'c_gr4j.h':
    int c_compute_PmEm(int nval,double * rain, double * evap, double* PmEm)

    int c_gr4j_X1_initial(double Pm, double Em, double X1, double * solution)

    int c_gr4j_run(int nval, int nparams,
            int nuh1, int nuh2,
            int ninputs,
            int nstates, int noutputs,
            int start, int end,
            double * params,
            double * uh1,
            double * uh2,
    	    double * inputs,
            double * statesuh1,
            double * statesuh2,
    	    double * states,
            double * outputs)

cdef extern from 'c_gr6j.h':
    int c_gr6j_run(int version, int nval, int nparams,
            int nuh1, int nuh2,
            int ninputs,
            int nstates, int noutputs,
            int start, int end,
    	    double * params,
            double * uh1,
            double * uh2,
    	    double * inputs,
            double * statesuh1,
            double * statesuh2,
    	    double * states,
            double * outputs)

cdef extern from 'c_lagroute.h':
    int c_lagroute_run(int nval,
            int nparams,
            int nuh,
            int nconfig,
            int ninputs,
            int nstates,
            int noutputs,
            int start, int end,
    	    double * config,
    	    double * params,
            double * uh,
    	    double * inputs,
            double * statesuh,
    	    double * states,
            double * outputs)

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

cdef extern from 'c_wapaba.h':
    int c_wapaba_run(int nval, int nparams,
            int ninputs,
            int nstates, int noutputs,
            int start, int end,
    	    double * params,
    	    double * inputs,
    	    double * statesini,
            double * outputs)


cdef extern from 'c_ihacres.h':
    int c_ihacres_run(int nval,
        int nconfig,
        int nparams,
        int ninputs,
        int nstates,
        int noutputs,
        int start, int end,
        double * config,
        double * params,
        double * inputs,
        double * statesini,
        double * outputs)

cdef extern from 'c_hbv.h':
    int c_hbv_get_maxuh()

    int c_hbv_run(int nval,
        int nparams,
        int ninputs,
        int nstates,
        int noutputs,
        int start, int end,
        double * params,
        double * dquh,
        double * inputs,
        double * statesini,
        double * outputs)


def __cinit__(self):
    pass


def gr2m_run(int start, int end,
        np.ndarray[double, ndim=1, mode='c'] config not None,
        np.ndarray[double, ndim=1, mode='c'] params not None,
        np.ndarray[double, ndim=2, mode='c'] inputs not None,
        np.ndarray[double, ndim=1, mode='c'] statesini not None,
        np.ndarray[double, ndim=2, mode='c'] outputs not None):

    cdef int ierr

    ierr = c_gr2m_run(inputs.shape[0],
            config.shape[0],
            params.shape[0],
            inputs.shape[1],
            statesini.shape[0],
            outputs.shape[1],
            start, end,
            <double*> np.PyArray_DATA(config),
            <double*> np.PyArray_DATA(params),
            <double*> np.PyArray_DATA(inputs),
            <double*> np.PyArray_DATA(statesini),
            <double*> np.PyArray_DATA(outputs))

    return ierr


def compute_PmEm(np.ndarray[double, ndim=1, mode='c'] rain not None,
        np.ndarray[double, ndim=1, mode='c'] evap not None,
        np.ndarray[double, ndim=1, mode='c'] PmEm not None):

    if rain.shape[0] != evap.shape[0]:
        raise ValueError('rain.shape[0] != evap.shape[0]')

    if PmEm.shape[0] != 2:
        raise ValueError('PmEm.shape[0] != 2')

    return c_compute_PmEm(rain.shape[0],
            <double*> np.PyArray_DATA(rain),
            <double*> np.PyArray_DATA(evap),
            <double*> np.PyArray_DATA(PmEm))


def gr4j_X1_initial(double Pm, double Em, double X1,
        np.ndarray[double, ndim=1, mode='c'] solution not None):

    return c_gr4j_X1_initial(Pm, Em, X1,
                <double*> np.PyArray_DATA(solution))


def gr4j_run(int nuh1, int nuh2, int start, int end,
             np.ndarray[double, ndim=1, mode='c'] params not None,
             np.ndarray[double, ndim=1, mode='c'] uh1 not None,
             np.ndarray[double, ndim=1, mode='c'] uh2 not None,
             np.ndarray[double, ndim=2, mode='c'] inputs not None,
             np.ndarray[double, ndim=1, mode='c'] statesuh1 not None,
             np.ndarray[double, ndim=1, mode='c'] statesuh2 not None,
             np.ndarray[double, ndim=1, mode='c'] states not None,
             np.ndarray[double, ndim=2, mode='c'] outputs not None):

    cdef int ierr

    # check dimensions
    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError('inputs.shape[0] != outputs.shape[0]')

    if inputs.shape[1] != 2:
        raise ValueError('inputs.shape[1] != 2')

    if uh1.shape[0] < nuh1:
        raise ValueError('uh1.shape[0] < nuh1')

    if uh2.shape[0] < nuh2:
        raise ValueError('uh2.shape[0] < nuh2')

    # Run model
    ierr = c_gr4j_run(inputs.shape[0],
                      params.shape[0],
                      nuh1,
                      nuh2,
                      inputs.shape[1],
                      states.shape[0],
                      outputs.shape[1],
                      start, end,
                      <double*> np.PyArray_DATA(params),
                      <double*> np.PyArray_DATA(uh1),
                      <double*> np.PyArray_DATA(uh2),
                      <double*> np.PyArray_DATA(inputs),
                      <double*> np.PyArray_DATA(statesuh1),
                      <double*> np.PyArray_DATA(statesuh2),
                      <double*> np.PyArray_DATA(states),
                      <double*> np.PyArray_DATA(outputs))
    return ierr


def gr6j_run(int version, int nuh1,
             int nuh2,
             int start, int end,
             np.ndarray[double, ndim=1, mode='c'] params not None,
             np.ndarray[double, ndim=1, mode='c'] uh1 not None,
             np.ndarray[double, ndim=1, mode='c'] uh2 not None,
             np.ndarray[double, ndim=2, mode='c'] inputs not None,
             np.ndarray[double, ndim=1, mode='c'] statesuh1 not None,
             np.ndarray[double, ndim=1, mode='c'] statesuh2 not None,
             np.ndarray[double, ndim=1, mode='c'] states not None,
             np.ndarray[double, ndim=2, mode='c'] outputs not None):

    cdef int ierr

    # check dimensions
    if params.shape[0] != 6:
        raise ValueError('params.shape[0] != 6')

    if states.shape[0] < 3:
        raise ValueError('states.shape[0] < 3')

    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError('inputs.shape[0] != outputs.shape[0]')

    if inputs.shape[1] != 2:
        raise ValueError('inputs.shape[1] != 2')

    if uh1.shape[0] < nuh1:
        raise ValueError('uh1.shape[0] < nuh1')

    if uh2.shape[0] < nuh2:
        raise ValueError('uh2.shape[0] < nuh2')

    # Run model
    ierr = c_gr6j_run(version, inputs.shape[0],
                      params.shape[0],
                      nuh1,
                      nuh2,
                      inputs.shape[1],
                      states.shape[0],
                      outputs.shape[1],
                      start, end,
                      <double*> np.PyArray_DATA(params),
                      <double*> np.PyArray_DATA(uh1),
                      <double*> np.PyArray_DATA(uh2),
                      <double*> np.PyArray_DATA(inputs),
                      <double*> np.PyArray_DATA(statesuh1),
                      <double*> np.PyArray_DATA(statesuh2),
                      <double*> np.PyArray_DATA(states),
                      <double*> np.PyArray_DATA(outputs))
    return ierr


def lagroute_run(int nuh, int start, int end,
                 np.ndarray[double, ndim=1, mode='c'] config not None,
                 np.ndarray[double, ndim=1, mode='c'] params not None,
                 np.ndarray[double, ndim=1, mode='c'] uh not None,
                 np.ndarray[double, ndim=2, mode='c'] inputs not None,
                 np.ndarray[double, ndim=1, mode='c'] statesuh not None,
                 np.ndarray[double, ndim=1, mode='c'] states not None,
                 np.ndarray[double, ndim=2, mode='c'] outputs not None):

    cdef int ierr

    # check dimensions
    if params.shape[0] != 2:
        raise ValueError('params.shape[0] != 2')

    if states.shape[0] < 1:
        raise ValueError('states.shape[0] < 1')

    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError('inputs.shape[0] != outputs.shape[0]')

    if inputs.shape[1] != 1:
        raise ValueError('inputs.shape[1] != 1')

    if uh.shape[0] < nuh:
        raise ValueError('uh.shape[0] < nuh')

    # Run model
    ierr = c_lagroute_run(inputs.shape[0],
                          params.shape[0],
                          nuh,
                          inputs.shape[1],
                          config.shape[1],
                          states.shape[0],
                          outputs.shape[1],
                          start, end,
                          <double*> np.PyArray_DATA(config),
                          <double*> np.PyArray_DATA(params),
                          <double*> np.PyArray_DATA(uh),
                          <double*> np.PyArray_DATA(inputs),
                          <double*> np.PyArray_DATA(statesuh),
                          <double*> np.PyArray_DATA(states),
                          <double*> np.PyArray_DATA(outputs))
    return ierr


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
                       params.shape[0],
                       nuh,
                       inputs.shape[1],
                       states.shape[0],
                       outputs.shape[1],
                       start, end,
                       <double*> np.PyArray_DATA(params),
                       <double*> np.PyArray_DATA(uh),
                       <double*> np.PyArray_DATA(inputs),
                       <double*> np.PyArray_DATA(statesuh),
                       <double*> np.PyArray_DATA(states),
                       <double*> np.PyArray_DATA(outputs))
    return ierr


def wapaba_run(int start, int end,
               np.ndarray[double, ndim=1, mode='c'] params not None,
               np.ndarray[double, ndim=2, mode='c'] inputs not None,
               np.ndarray[double, ndim=1, mode='c'] statesini not None,
               np.ndarray[double, ndim=2, mode='c'] outputs not None):

    cdef int ierr

    # check dimensions
    if params.shape[0] > 10:
        raise ValueError('params.shape[0] > 10')

    if statesini.shape[0] != 2:
        raise ValueError('statesini.shape[0] < 2')

    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError('inputs.shape[0] != outputs.shape[0]')

    if inputs.shape[1] != 2:
        raise ValueError('inputs.shape[1] != 2')

    ierr = c_wapaba_run(inputs.shape[0],
                        params.shape[0],
                        inputs.shape[1],
                        statesini.shape[0],
                        outputs.shape[1],
                        start, end,
                        <double*> np.PyArray_DATA(params),
                        <double*> np.PyArray_DATA(inputs),
                        <double*> np.PyArray_DATA(statesini),
                        <double*> np.PyArray_DATA(outputs))
    return ierr


def ihacres_run(int start, int end,
                np.ndarray[double, ndim=1, mode='c'] config not None,
                np.ndarray[double, ndim=1, mode='c'] params not None,
                np.ndarray[double, ndim=2, mode='c'] inputs not None,
                np.ndarray[double, ndim=1, mode='c'] statesini not None,
                np.ndarray[double, ndim=2, mode='c'] outputs not None):

    cdef int ierr

    # check dimensions
    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError('inputs.shape[0] != outputs.shape[0]')

    ierr = c_ihacres_run(inputs.shape[0],
                         config.shape[0],
                         params.shape[0],
                         inputs.shape[1],
                         statesini.shape[0],
                         outputs.shape[1],
                         start, end,
                         <double*> np.PyArray_DATA(config),
                         <double*> np.PyArray_DATA(params),
                         <double*> np.PyArray_DATA(inputs),
                         <double*> np.PyArray_DATA(statesini),
                         <double*> np.PyArray_DATA(outputs))
    return ierr


def hbv_get_maxuh():
        return c_hbv_get_maxuh()

def hbv_run(int start, int end,
            np.ndarray[double, ndim=1, mode='c'] params not None,
            np.ndarray[double, ndim=1, mode='c'] dquh not None,
            np.ndarray[double, ndim=2, mode='c'] inputs not None,
            np.ndarray[double, ndim=1, mode='c'] statesini not None,
            np.ndarray[double, ndim=2, mode='c'] outputs not None):

    cdef int ierr
    cdef int nuh = c_hbv_get_maxuh()

    # check dimensions
    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError('inputs.shape[0] != outputs.shape[0]')

    if dquh.shape[0] < nuh:
        raise ValueError('dquh.shape[0] < nuh')

    # Initialise data
    dquh.fill(0.)
    outputs[:, 0].fill(0.)

    # Run model
    ierr = c_hbv_run(inputs.shape[0],
                     params.shape[0],
                     inputs.shape[1],
                     statesini.shape[0],
                     outputs.shape[1],
                     start, end,
                     <double*> np.PyArray_DATA(params),
                     <double*> np.PyArray_DATA(dquh),
                     <double*> np.PyArray_DATA(inputs),
                     <double*> np.PyArray_DATA(statesini),
                     <double*> np.PyArray_DATA(outputs))
    return ierr

