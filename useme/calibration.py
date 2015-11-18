
import math
import random
import time

import numpy as np

from scipy.optimize import fmin_powell as fmin

from useme.model import Vector, Matrix


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

    def __init__(self, model, \
            ncalparams, \
            errfun=None, \
            minimize=True, \
            optimizer=fmin, \
            initialise_model=True, \
            timeit=False, \
            nfit=2):

        self._model = model
        self._minimize = minimize
        self._timeit = timeit
        self._ieval = 0
        self._iprint = 0
        self._runtime = np.nan
        self._initialise_model = initialise_model
        self._is_fitting = False
        self._dx_sensitivity = 1e-3
        self._nfit = nfit

        self._observations = None
        self._idx_cal = None

        self._calparams = Vector('calparams', ncalparams)
        self._calparams_means = Vector('calparams_means', ncalparams)
        self._calparams_stdevs = Vector('calparams_stdevs', \
                ncalparams*ncalparams)

        self.errfun = sse

        # Wrapper around optimizer to do 
        # send the current calibration object
        def _optimizer(objfun, start, calib, disp, *args, **kwargs):
            kwargs['disp'] = disp
            final = optimizer(objfun, start, *args, **kwargs)
            return final
            
        self._optimizer = _optimizer

    def __str__(self):
        str = 'Calibration instance for model {0}\n'.format(self._model.name)
        str += '  nfit       : {0}\n'.format(self._nfit)
        str += '  ncalparams : {0}\n'.format(self.calparams_means.nval)
        str += '  ieval      : {0}\n'.format(self.ieval)
        str += '  runtime    : {0}\n'.format(self._runtime)
        str += '  {0}\n'.format(self.calparams_means)
        str += '  {0}\n'.format(self.calparams)
        str += '  {0}\n'.format(self._model.params)

        return str

    @property
    def ieval(self):
        return self._ieval

    @ieval.setter
    def ieval(self, value):
        self._ieval = value


    @property
    def runtime(self):
        return self._runtime


    @property
    def calparams(self):
        return self._calparams


    @property
    def calparams_means(self):
        return self._calparams_means


    @property
    def calparams_stdevs(self):
        return self._calparams_stdevs


    @property
    def model(self):
        return self._model


    @property
    def observations(self):
        return self._observations


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
            self._model._params.data = params

            # Exit objectif function if parameters hit bounds
            if self._model._params.hitbounds and self._is_fitting:
                return np.inf

            # Run model initialisation if needed
            if self._initialise_model:
                self._model.initialise()

            idx_start = self._idx_cal[0]
            idx_end = self._idx_cal[-1]
            self._model.run(idx_start, idx_end)
            self._ieval += 1

            if self._timeit:
                t1 = time.time()
                self._runtime = (t1-t0)*1000

            # Get error model parameters if they exist
            errparams = self.cal2err(calparams)

            # Compute objectif function
            ofun = self._errfun(self._observations.data[self._idx_cal, :], \
                    self._model._outputs.data[self._idx_cal, :], errparams)

            if not self._minimize:
                ofun *= -1

            if self._iprint>0:
                if self._ieval % self._iprint == 0:
                    self._calparams.data = calparams
                    print('Fit {0:3d} : {1:3.3e} {2} ~ {3:.3f} ms'.format( \
                        self._ieval, ofun, self._calparams, \
                        self._runtime))

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

        if self._observations is None:
            raise ValueError('No observations data. Please allocate')

        if np.max(_idx_cal) >= self._observations.nval:
            raise ValueError('Wrong values in idx_cal')

        self._idx_cal = _idx_cal


    def check(self):
        ''' Performs check on calibrated model to ensure that all variables are
        properly allocated '''
        # Check idx_cal is allocated
        if self._idx_cal is None:
            raise ValueError('No idx_cal data. Please allocate')

        # Check observations are allocated
        if self._observations is None:
            raise ValueError('No observations data. Please allocate')

        # Check inputs are allocated
        if self._model.inputs is None:
            raise ValueError(('No inputs data for model {0}.' + \
                ' Please allocate').format(self._model.name))

        # Check inputs are initialised
        if np.all(np.isnan(self._model._inputs.data)):
            raise ValueError(('All inputs data are NaN for model {0}.' + \
                ' Please initialise').format(self._model.name))

        # Check outputs are allocated
        if self._model.outputs is None:
            raise ValueError(('No outputs data for model {0}.' + \
                ' Please allocate').format(self._model.name))

        # Check inputs and observations have the right dimension
        n1 = self._model._inputs.nval
        n2 = self._observations.nval
        if n1 != n2:
            raise ValueError(('model.inputs.nval({0}) !=' + \
                ' observations.nval({1})').format(n1, n2))

        n1 = self._model._outputs.nvar
        n2 = self._observations.nvar
        if n1 != n2:
            raise ValueError(('model.outputs.nvar({0}) !=' + \
                ' observations.nvar({1})').format(n1, n2))

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
        return calparams


    def cal2err(self, calparams):
        return None


    def setup(self, observations, inputs):

       self._observations = Matrix.fromdata('observations', observations)

       self._model.allocate(self._observations.nval, self._observations.nvar)
       self._model.inputs.data = inputs


    def sample(self, nsamples, seed=333):

        # Save random state
        random_state = random.getstate()

        # Set seed
        np.random.seed(seed)

        ncalparams = self.calparams_means.nval

        # sample parameters
        samples = np.random.multivariate_normal(\
                self.calparams_means.data, \
                self.calparams_stdevs.data.reshape( \
                    ncalparams, ncalparams), \
                nsamples)

        samples = np.atleast_2d(samples)
        if samples.shape[0] == 1:
            samples = samples.T

        # Reset random state
        random.setstate(random_state)

        return samples


    def explore(self, \
            calparams_explore=None, \
            nsamples = None, \
            iprint=0, \
            seed=333):

        self.check()
        self._iprint = iprint
        self._ieval = 0

        if nsamples is None:
            ncalparams = self._calparams_means.nval
            nsamples = int(200 * math.sqrt(ncalparams))

        if calparams_explore is None:
            calparams_explore = self.sample(nsamples, seed)
        else:
            calparams_explore = np.atleast_2d(calparams_explore)
            if calparams_explore.shape[0] == 1:
                calparams_explore = calparams_explore.T

            nsamples = calparams_explore.shape[0]

        ofun_explore = np.zeros(nsamples) * np.nan
        ofun_min = np.inf

        # Systematic exploration
        calparams_best = None

        for i in range(nsamples):
            calparams = calparams_explore[i,:]
            ofun = self._objfun(calparams)
            ofun_explore[i] = ofun

            if self._iprint>0:
                if self._ieval % self._iprint == 0:
                    self._calparams.data = calparams
                    print(('Exploration {0}/{1} : ' + \
                        '{2:3.3e} {3} ~ {4:.2f} ms').format( \
                        self._ieval, nsamples, ofun, self._calparams, \
                        self._runtime))

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
        self._is_fitting = True
        nfit = self._nfit

        if self._iprint>0:
            ofun_ini = self._objfun(calparams_ini)

            self._calparams.data = calparams_ini
            print('\nFit start: {0:3.3e} {1} ~ {2:.2f} ms\n'.format( \
                    ofun_ini, self._calparams, self._runtime))

        # Apply the optimizer several times to ensure convergence
        for k in range(nfit):
            final = self._optimizer(self._objfun, \
                    calparams_ini, self, disp=self._iprint>0, \
                    *args, **kwargs)
            calparams_ini = final

        ofun_final = self._objfun(final)
        outputs_final = self.model.outputs.data
        self._calparams.data = final

        if self._iprint>0:
            print('\nFit final: {0:3.3e} {1} ~ {2:.2f} ms\n'.format( \
                    ofun_final, self._calparams, self._runtime))

        self._is_fitting = False

        return final, outputs_final, ofun_final


    def fullfit(self, observations, inputs, \
            idx_cal=None, \
            iprint=0, \
            nsamples=None, \
            nfit=2):

        self.setup(observations, inputs)

        if idx_cal is None:
            idx_cal = np.arange(self._observations.nval)
        self.idx_cal = idx_cal

        try:
            start, explo, explo_ofun = self.explore(iprint=iprint, \
                nsamples=nsamples)
        except ValueError:
            start = self.model.params.default

        final, out, out_ofun = self.fit(start, iprint=iprint)

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




