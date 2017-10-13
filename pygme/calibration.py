import math, time

import numpy as np

from scipy.optimize import fmin_powell

from hydrodiy.stat import transform
from hydrodiy.data.containers import Vector

BC = transform.BoxCox()

class ObjFun(object):
    ''' Generic class to describe objective functions '''

    def __init__(self, name, orientation=1, \
                    constants=None):

        self.name = name

        if not orientation in [-1, 1]:
            raise ValueError(('Expected orientation in [-1, 1],'+\
                ' got {0}').format(orientation))

        self.orientation = orientation

    def __str__(self):
        str = 'Objective function {0}\n'.format(self.name)
        str += '  orientation : {0}\n'.format(self.orientation)

        return str


    def compute(self, obs, sim):
        raise ValueError('Need to override this function')



class ObjFunSSE(ObjFun):
    ''' Sum of squared error objective function '''

    def __init__(self):
        ObjFun.__init__(self,'SSE')


    def compute(self, obs, sim):
        err = obs-sim
        return np.nansum(err*err)


class ObjFunBCSSE(ObjFun):
    ''' Sum of squared error objective function '''

    def __init__(self, lam=0.2):
        ObjFun.__init__(self,'BCSSE')

        # Set Transform
        BC['lambda'] = lam
        self.trans = BC

    def compute(self, obs, sim):
        tobs = self.trans.forward(obs)
        tsim = self.trans.forward(sim)
        err = tobs-tsim
        return np.nansum(err*err)



class Calibration(object):

    def __init__(self, model, calparams, \
            warmup=0, \
            objfun=ObjFunSSE, \
            minimize=True, \
            optimizer=fmin_powell, \
            initialise_model=True, \
            timeit=False, \
            nrepeat_fit=2):

        self._model = model
        self._calparams = calparams
        self._warmup = warmup
        self._minimize = minimize
        self._timeit = timeit
        self._ieval = 0
        self._iprint = 0
        self._runtime = np.nan
        self._initialise_model = initialise_model
        self._status = 'intialised'
        self._nrepeat_fit = nrepeat_fit

        self._obs = None
        self._index_cal = None

        self._objfun = objfun

        # Wrapper around optimizer to
        # send the current calibration object
        #def _optimizer(objfun, start, calib, disp, *args, **kwargs):
        #    kwargs['disp'] = disp
        #    final = optimizer(objfun, start, *args, **kwargs)
        #    return final

        self._optimizer = optimizer


    def __str__(self):
        str = ('Calibration instance ' +
                'for model {0}\n').format(self._model.name)
        str += '  objfun     : {0}\n'.format(self.objfun.name)
        str += '  status     : {0}\n'.format(self._status)
        str += '  warmup     : {0}\n'.format(self._warmup)
        str += '  nrepeat_fit: {0}\n'.format(self._nrepeat_fit)
        str += '  ncalparams : {0}\n'.format(self.alparams)
        str += '  ieval      : {0}\n'.format(self._ieval)

        return str


    @property
    def ieval(self):
        return self._ieval


    @property
    def warmup(self):
        return self._warmup


    @property
    def runtime(self):
        return self._runtime


    @property
    def calparams(self):
        return self._calparams


    @property
    def model(self):
        return self._model


    @property
    def obs(self):
        return self._obs


    @property
    def index_cal(self):
        return self._index_cal

    @index_cal.setter
    def index_cal(self, value):

        if self._obs is None:
            raise ValueError('No obs data. Please allocate')

        index = self._obs.index

        # Set to all indexes if None
        if value is None:
            value = index

        if value.dtype == np.dtype('bool'):
            if value.shape[0] != index.shape[0]:
                raise ValueError('Trying to set index_cal, got {0} values,' +
                        ' expected {1}'.format(value.shape[0], index.shape[0]))

            _index_cal = index[np.where(value)[0]]

        else:
            _index_cal = value

        # check value is within obs indexes
        if np.any(~np.in1d(_index_cal, index)):
            raise ValueError(('Certain values in index_cal are not within '
                'obs index'))

        # Check value leaves enough data for warmup
        istart = np.where(_index_cal[0] == index)[0]
        if istart < self.warmup:
            raise ValueError(('index_cal starts at {0}, which leaves {1} values '
                'for warmup, whereas {2} are expected').format(_index_cal[0],
                    istart, self.warmup))

        self._index_cal = _index_cal


    def _fitfun(self, calparams_values):

        model = self._model

        # Set model parameters
        calparams_values = np.atleast_1d(calparams_values)
        params_values = self.cal2true(calparams_values)
        model.params.values = params_values

        # Exit objectif function if parameters hit bounds
        if model._params.hitbounds and self._status:
            return np.inf

        # Set start/end of model
        istart = self._index_cal[0] - self.warmup
        if istart < 0:
            raise ValueError('Tried to set model start index before '
                'the first index')

        index = self._obs.index
        istart = np.where(index == istart)[0]
        model.istart = index[istart]
        model.iend = self._index_cal[-1]

        # Initialise model if needed
        if self._initialise_model:
            model.initialise()

        # Run model with runtime assessment
        if self._timeit:
            t0 = time.time()

        model.run()
        self._ieval += 1

        if self._timeit:
            t1 = time.time()
            self._runtime = (t1-t0)*1000

        # Locate indexes in the calibration period
        kk = np.in1d(index, self._index_cal)

        # Compute objectif function during calibration period
        ofun = self._objfun.compute(self.obs[kk, :], \
                    self._model.outputs[kk, :])

        if not self._minimize:
            ofun *= -1

        if np.isnan(ofun):
            raise ValueError('Objective function returns nan')

        # Store data
        self._calparams.data = calparams

        # Print output if needed
        if self._iprint>0:
            if self._ieval % self._iprint == 0:
                print('{4} {0:3d} : {1:3.3e} {2} ~ {3:.3f} ms'.format( \
                    self._ieval, ofun, self._calparams.data, \
                    self._runtime, self._status))

        return ofun


    def check(self):
        ''' Performs check on calibrated model to ensure that all variables are
        properly allocated '''
        # Check index_cal is allocated
        if self._index_cal is None:
            raise ValueError('No index_cal data. Please allocate')

        # Check obs are allocated
        if self._obs is None:
            raise ValueError('No obs data. Please allocate')

        # Check inputs are allocated
        if self.model._inputs is None:
            raise ValueError(('No inputs data for model {0}.' + \
                ' Please allocate').format(self._model.name))

        # Check outputs are allocated
        if self.model._outputs is None:
            raise ValueError(('No outputs data for model {0}.' + \
                ' Please allocate').format(self._model.name))

        # Check inputs and obs have the right dimension
        nval, nvar, _, _ = self.model.get_dims('outputs')
        nval2 = self._obs.nval
        if nval != nval2:
            raise ValueError(('model inputs nval({0}) !=' + \
                ' obs nval({1})').format(nval, nval2))

        nvar2 = self._obs.nvar
        if nvar != nvar2:
            raise ValueError(('model outputs nvar({0}) !=' + \
                ' obs nvar({1})').format(nvar, nvar2))

        # Check params size
        calparams = np.zeros(self.ncalparams)
        params = self.cal2true(calparams)
        try:
            check = (params.ndim == 1)
        except Exception:
            check = False
        if not check:
            raise ValueError('cal2true does not return a 1D Numpy array')


    def cal2true(self, calparams_values):
        ''' Convert calibrated parameters to true values '''
        return calparams_values


    def allocate(self, obs, inputs):

        obs = np.atleast_2d(obs)
        inputs = np.atleast_2d(inputs)

        nval, noutputs = obs.shape

        # Check inputs and outputs size
        if inputs.shape[0] != nval:
            raise ValueError(('Expected same number of timestep '+\
                'in inputs({0}) and outputs({1})').format(\
                    inputs.shape[0], nval))

        # Allocate model
        self.model.allocate(inputs, noutputs)

        # Set obs data
        self._obs = obs

        # By default calibrate on everything excluding warmup
        index_cal = np.arange(nval) >= self._warmup
        self._index_cal = index_cal


    def explore(self, nsamples = None, iprint=0,
            distribution = 'normal'):
        ''' Systematic exploration of parameter space and
        identification of best parameter set
        '''

        self.check()
        self._iprint = iprint
        self._ieval = 0
        self._status = 'explore'
        ncalparams = self.ncalparams

        # Set the number of samples
        if nsamples is None:
            nsamples = int(200 * math.sqrt(ncalparams))

        # Get random samples from parameter
        calparams_explore = self._calparams.clone(nsamples)
        calparams_explore.random(distribution=distribution)

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
                    self.calparams = calparams

            if ofun < ofun_min:
                ofun_min = ofun
                calparams_best = calparams

        if calparams_best is None:
            raise ValueError('Could not identify a suitable' + \
                '  parameter by exploration')

        self.calparams = calparams_best

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
        outputs_final = self.model.outputs
        self.calparams = final

        if self._iprint>0:
            print('\n>> Fit final: {0:3.3e} {1} ~ {2:.2f} ms <<\n'.format( \
                    ofun_final, self._calparams.data, self._runtime))

        self._status = 'fit completed'

        return final, outputs_final, ofun_final


    def run(self, obs, inputs, \
            index_cal=None, \
            iprint=0, \
            nsamples=None,
            *args, **kwargs):

        self.setup(obs, inputs)

        if index_cal is None:
            index_cal = obs.index
        self.index_cal = index_cal

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



