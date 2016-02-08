
import math
import random
import time

import numpy as np

from scipy.optimize import fmin_powell

from pygme.model import Vector, Matrix

def set_seed(seed=333):
    np.random.seed(seed)

def sse(obs, sim, errparams):
    err = obs-sim
    return np.sum(err*err)


def sseabs_bias(obs, sim, errparams):
    err = np.abs(obs-sim)
    E = np.sum(err*err)
    B = np.mean(obs-sim)
    return E*(1+abs(B))


def ssqe_bias(obs, sim, errparams):
    err = np.sqrt(obs)-np.sqrt(sim)
    E = np.sum(err*err)
    B = np.mean(obs-sim)
    return E*(1+abs(B))


def sls_llikelihood(obs, sim, errparams):
    err = obs-sim
    sigma = errparams[0]
    nval = len(obs)

    ll = np.sum(err*err)/(2*sigma*sigma) + nval * math.log(sigma)
    return ll



class Calibration(object):

    def __init__(self, model,
            ncalparams,
            nens_params = 1,
            errfun=None,
            minimize=True,
            optimizer=fmin_powell,
            initialise_model=True,
            timeit=False,
            nrepeat_fit=2):

        self._model = model
        self._minimize = minimize
        self._timeit = timeit
        self._ieval = 0
        self._iprint = 0
        self._runtime = np.nan
        self._initialise_model = initialise_model
        self._status = 'intialised'
        self._dx_sensitivity = 1e-3
        self._nrepeat_fit = nrepeat_fit

        self._obsdata = None
        self._idx_cal = None

        self._calparams = Vector('calparams', ncalparams, nens_params)

        self.errfun = sse

        # Wrapper around optimizer to
        # send the current calibration object
        def _optimizer(objfun, start, calib, disp, *args, **kwargs):
            kwargs['disp'] = disp
            final = optimizer(objfun, start, *args, **kwargs)
            return final

        self._optimizer = _optimizer

    def __str__(self):
        str = 'Calibration instance for model {0}\n'.format(self._model.name)
        str += '  status     : {0}\n'.format(self._status)
        str += '  nrepeat_fit: {0}\n'.format(self._nrepeat_fit)
        str += '  ncalparams : {0}\n'.format(self._calparams.nval)
        str += '  ieval      : {0}\n'.format(self._ieval)

        return str

    @property
    def ieval(self):
        return self._ieval


    @property
    def runtime(self):
        return self._runtime


    @property
    def calparams(self):
        return self._calparams.data


    @property
    def model(self):
        return self._model


    @property
    def obsdata(self):
        return self._obsdata.data


    @property
    def errfun(self):
        return self._errfun

    @errfun.setter
    def errfun(self, value):
        self._errfun = value

        def objfun(calparams):

            if self._timeit:
                t0 = time.time()

            # Set model parameters
            params = self.cal2true(calparams)
            self._model.params = params

            # Exit objectif function if parameters hit bounds
            if self._model._params.hitbounds and self._status:
                return np.inf

            # Run model initialisation if needed
            if self._initialise_model:
                self._model.initialise()

            self._model.idx_start = self._idx_cal[0]
            self._model.idx_end = self._idx_cal[-1]
            self._model.run()
            self._ieval += 1

            if self._timeit:
                t1 = time.time()
                self._runtime = (t1-t0)*1000

            # Get error model parameters if they exist
            # example standard deviation of normal gaussian error model
            errparams = self.cal2err(calparams)

            # Compute objectif function
            ofun = self._errfun(self.obsdata[self._idx_cal, :], \
                    self._model.outputs[self._idx_cal, :], errparams)

            if not self._minimize:
                ofun *= -1

            if self._iprint>0:
                if self._ieval % self._iprint == 0:
                    self._calparams.data = calparams
                    print('{4} {0:3d} : {1:3.3e} {2} ~ {3:.3f} ms'.format( \
                        self._ieval, ofun, self._calparams.data, \
                        self._runtime, self._status))

            return ofun

        self._objfun = objfun


    @property
    def idx_cal(self):
        return self._idx_cal

    @idx_cal.setter
    def idx_cal(self, value):
        if value.dtype == np.dtype('bool'):
            _idx_cal = np.where(value)[0]
        else:
            _idx_cal = value

        if self._obsdata is None:
            raise ValueError('No obsdata data. Please allocate')

        if np.max(_idx_cal) >= self._obsdata.nval:
            raise ValueError(('Max value in idx_cal({0})' +
                ' exceeds number of observations ({1})').format(np.max(idx_cal),
                                                        self._obsdata.nval))

        self._idx_cal = _idx_cal


    def check(self):
        ''' Performs check on calibrated model to ensure that all variables are
        properly allocated '''
        # Check idx_cal is allocated
        if self._idx_cal is None:
            raise ValueError('No idx_cal data. Please allocate')

        # Check obsdata are allocated
        if self._obsdata is None:
            raise ValueError('No obsdata data. Please allocate')

        # Check inputs are allocated
        if self._model._inputs is None:
            raise ValueError(('No inputs data for model {0}.' + \
                ' Please allocate').format(self._model.name))

        # Check outputs are allocated
        if self._model._outputs is None:
            raise ValueError(('No outputs data for model {0}.' + \
                ' Please allocate').format(self._model.name))

        # Check inputs and obsdata have the right dimension
        n1 = self._model._inputs.nval
        n2 = self._obsdata.nval
        if n1 != n2:
            raise ValueError(('model inputs nval({0}) !=' + \
                ' obsdata nval({1})').format(n1, n2))

        n1 = self._model._outputs.nvar
        n2 = self._obsdata.nvar
        if n1 != n2:
            raise ValueError(('model outputs nvar({0}) !=' + \
                ' obsdata nvar({1})').format(n1, n2))

        # Check params size
        ncalparams = self._calparams.nval
        calparams = np.zeros(ncalparams)
        params = self.cal2true(calparams)
        if len(params.shape) != 1:
            raise ValueError('cal2true does not return a 1D array')

        errparams = self.cal2err(calparams)
        if not errparams is None:
            if len(errparams.shape) != 1:
                raise ValueError('cal2err does not return a 1D array')



    def cal2true(self, calparams):
        ''' Convert calibrated parameters to true values '''
        return calparams


    def cal2err(self, calparams):
        ''' Get error model parameters from the list of calibrated parameters '''
        return None

    def setup(self, obsdata, inputs):

        if inputs.nval != obsdata.nval:
            raise ValueError(('Number of value in inputs({0}) different' +
                ' from obsdata ({1})').format(inputs.nval, obdata.nval))

        if obsdata.nvar > self._model.noutputs_max:
            raise ValueError(('Number of variables in outputs({0}) greater' +
                ' than model can produce ({1})').format(obsdata.nvar,
                                                self._model.noutputs_max))


        self._model.allocate(inputs.nval, obsdata.nvar,
                            inputs.nens)
        self._model._inputs = inputs

        self._obsdata = obsdata


    def explore(self, nsamples = None, iprint=0,
            distribution = 'normal'):
        ''' Systematic exploration of parameter space and
        identification of best parameter set
        '''

        self.check()
        self._iprint = iprint
        self._ieval = 0
        self._status = 'explore'
        ncalparams = self._calparams.nval

        # Set the number of samples
        if nsamples is None:
            nsamples = int(200 * math.sqrt(ncalparams))

        # Get random samples from parameter
        calparams_explore = self._calparams.clone(nsamples)
        calparams_explore.random()

        # Setup objective function
        ofun_explore = np.zeros(nsamples) * np.nan
        ofun_min = np.inf

        # Systematic exploration
        calparams_best = None

        for i in range(nsamples):
            # Get parameter sample
            calparams = calparams_explore._data[i, :]

            ofun = self._objfun(calparams)
            ofun_explore[i] = ofun

            if self._iprint>0:
                if self._ieval % self._iprint == 0:
                    self._calparams.data = calparams

            if ofun < ofun_min:
                ofun_min = ofun
                calparams_best = calparams

        if calparams_best is None:
            raise ValueError('Could not identify a suitable' + \
                '  parameter by exploration')

        self._calparams.data = calparams_best

        return calparams_best, calparams_explore, ofun_explore


    def fit(self, calparams_ini, iprint=0, *args, **kwargs):

        self.check()
        self._iprint = iprint
        self._status = 'fit'
        nrepeat_fit = self._nrepeat_fit

        if np.any(np.isnan(calparams_ini)):
            raise ValueError('calparams_ini contains NaN')

        if self._iprint>0:
            ofun_ini = self._objfun(calparams_ini)

            self._calparams.data = calparams_ini
            print('\n>> Fit start: {0:3.3e} {1} ~ {2:.2f} ms <<\n'.format( \
                    ofun_ini, calparams_ini, self._runtime))

        # Apply the optimizer several times to ensure convergence
        for k in range(nrepeat_fit):
            final = self._optimizer(self._objfun, \
                    calparams_ini, self, disp=self._iprint>0, \
                    *args, **kwargs)

            calparams_ini = final

        ofun_final = self._objfun(final)
        outputs_final = self.model.outputs.data
        self._calparams.data = final

        if self._iprint>0:
            print('\n>> Fit final: {0:3.3e} {1} ~ {2:.2f} ms <<\n'.format( \
                    ofun_final, self._calparams.data, self._runtime))

        self._status = 'fit completed'

        return final, outputs_final, ofun_final


    def fullfit(self, obsdata, inputs, \
            idx_cal=None, \
            iprint=0, \
            nsamples=None,
            *args, **kwargs):

        self.setup(obsdata, inputs)

        if idx_cal is None:
            idx_cal = np.arange(self._obsdata.nval)
        self.idx_cal = idx_cal

        try:
            start, explo, explo_ofun = self.explore(iprint=iprint, \
                nsamples=nsamples)
        except ValueError:
            start = self.model._params.default

        kwargs['iprint'] = iprint
        final, out, out_ofun = self.fit(start, *args, **kwargs)

        return final, out, out_ofun


    def sensitivity(self, calparams=None, dx=1e-3):

        if calparams is None:
            calparams = self._calparams.data

        ncalparams = self._calparams.nval
        sensitivity = np.zeros(ncalparams)

        ofun0 = self._objfun(calparams)

        for k in range(ncalparams):
            cp = calparams.copy()
            cp[k] += dx
            ofun1 = self._objfun(cp)
            sensitivity[k] = (ofun1-ofun0)/dx

        return sensitivity




class CrossValidation(object):

    def __init__(self, calib,
            scheme='split', \
            warmup=0, \
            leaveout=0, \
            explore=True):

        self._calib = calib
        self._scheme = scheme
        self._warmup = warmup
        self._explore = explore

        self._calperiods = None
        self._calparams_ini = None

    def __str__(self):
        calib = self._calib
        str = 'Cross-validation instance for model {0}\n'.format(calib._model.name)
        str += '  scheme     : {0}\n'.format(self._scheme)
        str += '  explore    : {0}\n'.format(self._explore)
        str += '  ncalparams : {0}\n'.format(calib.calparams_means.nval)
        str += '  nperiods   : {0}\n'.format(len(self._calperiods))

        return str


    @property
    def calparams_ini(self):
        return self._calparams_ini

    @calparams_ini.setter
    def calparams_ini(self, value):
        self._calparams_ini = Vector('calparams_ini', \
                self._calib._calparams.nval)

        self._calparams_ini.data = value


    def setup(self):
        self._calib.check()
        nval = self._calib.model.inputs.nval
        warmup = self._warmup

        if self._scheme == 'split':
            idx_1 = np.arange(warmup, (nval+warmup)/2)
            idx_2 = np.arange((nval+warmup)/2, nval)

            self._calperiods = [ \
                    {'id':'PER1', 'idx_cal':idx_1}, \
                    {'id':'PER2', 'idx_cal':idx_2} \
            ]


    def run(self):

        if self._idx_cals is None:
            raise ValueError(('No idx_cal for cross-validation of model {0}.' + \
                ' Please setup').format(self._calib._model.name))

        nper = len(self._calperiods)

        for i in range(nper):
            per = self._calperiods[i]
            self._calib.idx_cal = per['idx_cal']

            # Define starting point
            if self._explore:
                calparams_ini, _, _ = self._calib.explore()
            else:
                calparams_ini = self._calparams_ini.value

            # Perform fit for calibration period
            final, out, ofun, sensit = self._calib.fit(calparams_ini)

            per['calparams'] = final.copy()
            per['params'] = self._calib.model.params.data.copy()
            per['simulation'] = out
            per['objfun'] = ofun




