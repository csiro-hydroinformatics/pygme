import math, time
from datetime import datetime

import numpy as np

from scipy.optimize import fmin_powell

from pygme.data import Vector, Matrix

now = datetime.now


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
            warmup=0,
            nens_params = 1,
            errfun=None,
            minimize=True,
            optimizer=fmin_powell,
            initialise_model=True,
            timeit=False,
            nrepeat_fit=2):

        self._model = model
        self._warmup = warmup
        self._minimize = minimize
        self._timeit = timeit
        self._ieval = 0
        self._iprint = 0
        self._runtime = np.nan
        self._initialise_model = initialise_model
        self._status = 'intialised'
        self._nrepeat_fit = nrepeat_fit

        self._obsdata = None
        self._index_cal = None

        # Create vector of calibrated parameters
        # (can be different from model parameters)
        self._calparams = Vector('calparams', ncalparams, nens_params)
        self._calparams.default = np.zeros(ncalparams, dtype=np.float64)
        self._calparams.means = np.zeros(ncalparams, dtype=np.float64)
        self._calparams.covar = np.eye(ncalparams, dtype=np.float64)

        self.errfun = sse

        # Wrapper around optimizer to
        # send the current calibration object
        def _optimizer(objfun, start, calib, disp, *args, **kwargs):
            kwargs['disp'] = disp
            final = optimizer(objfun, start, *args, **kwargs)
            return final

        self._optimizer = _optimizer


    def __str__(self):
        str = ('Calibration instance ' +
                'for model {0}\n').format(self._model.name)
        str += '  status     : {0}\n'.format(self._status)
        str += '  warmup     : {0}\n'.format(self._warmup)
        str += '  nrepeat_fit: {0}\n'.format(self._nrepeat_fit)
        str += '  ncalparams : {0}\n'.format(self.ncalparams)
        str += '  ieval      : {0}\n'.format(self._ieval)

        return str


    @property
    def ieval(self):
        return self._ieval


    @property
    def warmup(self):
        return self._warmup

    @warmup.setter
    def warmup(self, value):

        if self._obsdata is None:
            raise ValueError('No obsdata data. Please allocate')

        nval = self._obsdata.nval
        if value > nval:
            raise ValueError('Tried setting warmup, got a value greater ({0}) ' +
                'than the number of observations ({1})'.format(value, nval))
        self._warmup = value

    @property
    def runtime(self):
        return self._runtime


    @property
    def ncalparams(self):
        return self._calparams.nval


    @property
    def calparams(self):
        return self._calparams.data


    @calparams.setter
    def calparams(self, value):
        self._calparams.data = value


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

            model = self._model

            if self._timeit:
                t0 = time.time()

            # Set model parameters
            params = self.cal2true(calparams)
            model.params = params

            # Exit objectif function if parameters hit bounds
            if model._params.hitbounds and self._status:
                return np.inf

            # Run model initialisation if needed
            if self._initialise_model:
                model.initialise()

            # run model
            model.run()
            self._ieval += 1

            if self._timeit:
                t1 = time.time()
                self._runtime = (t1-t0)*1000

            # Get error model parameters if they exist
            # example standard deviation of normal gaussian error model
            errparams = self.cal2err(calparams)

            # Locate indexes in the calibration period
            index = self._obsdata.index
            kk = np.in1d(index, self._index_cal)

            # Set start/end of model
            istart = self._index_cal[0] - self.warmup
            if istart < 0:
                raise ValueError('Tried to set model start index before '
                    'the first index')
            istart = np.where(index == istart)[0]
            model.index_start = index[istart]
            model.index_end = self._index_cal[-1]

            # Compute objectif function
            ofun = self._errfun(self.obsdata[kk, :], \
                    self._model.outputs[kk, :], errparams)

            if not self._minimize:
                ofun *= -1

            # Print output if needed
            if self._iprint>0:
                if self._ieval % self._iprint == 0:
                    self._calparams.data = calparams
                    print('{4} {0:3d} : {1:3.3e} {2} ~ {3:.3f} ms'.format( \
                        self._ieval, ofun, self._calparams.data, \
                        self._runtime, self._status))

            return ofun

        self._objfun = objfun


    @property
    def index_cal(self):
        return self._index_cal

    @index_cal.setter
    def index_cal(self, value):

        if self._obsdata is None:
            raise ValueError('No obsdata data. Please allocate')

        index = self._obsdata.index

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

        # check value is within obsdata indexes
        if np.any(~np.in1d(_index_cal, index)):
            raise ValueError(('Certain values in index_cal are not within '
                'obsdata index'))

        # Check value leaves enough data for warmup
        istart = np.where(_index_cal[0] == index)[0]
        if istart < self.warmup:
            raise ValueError(('index_cal starts at {0}, which leaves {1} values '
                'for warmup, whereas {2} are expected').format(_index_cal[0],
                    istart, self.warmup))

        self._index_cal = _index_cal


    def check(self):
        ''' Performs check on calibrated model to ensure that all variables are
        properly allocated '''
        # Check index_cal is allocated
        if self._index_cal is None:
            raise ValueError('No index_cal data. Please allocate')

        # Check obsdata are allocated
        if self._obsdata is None:
            raise ValueError('No obsdata data. Please allocate')

        # Check inputs are allocated
        if self.model._inputs is None:
            raise ValueError(('No inputs data for model {0}.' + \
                ' Please allocate').format(self._model.name))

        # Check outputs are allocated
        if self.model._outputs is None:
            raise ValueError(('No outputs data for model {0}.' + \
                ' Please allocate').format(self._model.name))

        # Check inputs and obsdata have the right dimension
        nval, nvar, _, _ = self.model.get_dims('outputs')
        nval2 = self._obsdata.nval
        if nval != nval2:
            raise ValueError(('model inputs nval({0}) !=' + \
                ' obsdata nval({1})').format(nval, nval2))

        nvar2 = self._obsdata.nvar
        if nvar != nvar2:
            raise ValueError(('model outputs nvar({0}) !=' + \
                ' obsdata nvar({1})').format(nvar, nvar2))

        # Check params size
        calparams = np.zeros(self.ncalparams)
        params = self.cal2true(calparams)
        try:
            check = (params.ndim == 1)
        except Exception:
            check = False
        if not check:
            raise ValueError('cal2true does not return a 1D Numpy array')

        errparams = self.cal2err(calparams)
        if not errparams is None:
            try:
                check = (errparams.ndim == 1)
            except Exception:
                check = False
            if not check:
                raise ValueError('cal2err does not return a Numpy 1D Numpy array')


    def cal2true(self, calparams):
        ''' Convert calibrated parameters to true values '''
        return calparams


    def cal2err(self, calparams):
        ''' Get error model parameters from the list of calibrated parameters '''
        return np.zeros(1)


    def setup(self, obsdata, inputs):

        if inputs.nval != obsdata.nval:
            raise ValueError(('Number of value in inputs({0}) different' +
                ' from obsdata ({1})').format(inputs.nval, obdata.nval))

        if not np.allclose(inputs.index, obsdata.index):
            raise ValueError('Different indexes in obsdata and inputs')

        if obsdata.nvar > self._model.noutputs_max:
            raise ValueError(('Number of variables in outputs({0}) greater' +
                ' than model can produce ({1})').format(obsdata.nvar,
                                                self._model.noutputs_max))

        self.model.allocate(inputs, obsdata.nvar)

        self._obsdata = obsdata

        # By default calibrate on everything excluding warmup
        index_cal = inputs.index[np.arange(obsdata.nval) >= self._warmup]
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


    def run(self, obsdata, inputs, \
            index_cal=None, \
            iprint=0, \
            nsamples=None,
            *args, **kwargs):

        self.setup(obsdata, inputs)

        if index_cal is None:
            index_cal = obsdata.index
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




class CrossValidation(object):

    def __init__(self, calib, explore=True):

        self._calib = calib
        self._explore = explore

        self._scheme = None
        self._calparams_ini = None
        self._calperiods = []

        self._obsdata = calib._obsdata.clone()
        nval = self._obsdata.nval
        self._mask = np.ones(nval, dtype=bool)


    def __str__(self):
        calib = self._calib
        str = ('Cross-validation instance for' +
                ' model {0}\n').format(calib._model.name)
        str += '  scheme     : {0}\n'.format(self._scheme)
        str += '  explore    : {0}\n'.format(self._explore)
        str += '  ncalparams : {0}\n'.format(calib.ncalparams)
        str += '  nperiods   : {0}\n'.format(len(self._calperiods))

        return str


    @property
    def calib(self):
        return self._calib


    @property
    def calperiods(self):
        return self._calperiods


    @property
    def calparams_ini(self):
        return self._calparams_ini

    @calparams_ini.setter
    def calparams_ini(self, value):
        self._calparams_ini = Vector('calparams_ini', \
                self._calib._calparams.nval)

        self._calparams_ini.data = value


    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        _value = np.atleast_1d(value, dtype=bool)

        nval = calib._obsdata.nval
        if _value.shape[0] != nval:
            raise ValueError(('Length of mask({0}) does not match ' +
                'nval of obsdata').format(_value.shape[0], nval))

        self._mask = mask


    def get_period_indexes(self, iperiod, is_cal=True):
        if iperiod >= len(self._calperiods):
            raise ValueError(('iperiod ({0}) is greater than the' +
                ' number of periods ({1})').format(iperiod, len(self._calperiods)))

        per = self._calperiods[iperiod]

        # Get total period
        label = ['val', 'cal'][int(is_cal)]
        ipos = np.arange(per['ipos_{0}_start'.format(label)],
                    per['ipos_{0}_end'.format(label)]+1)

        # Remove leave out if any
        item = 'ipos_{0}_startleaveout'.format(label)
        if not per[item] is None:
            ipos_leave = np.arange(per[item],
                    per['ipos_{0}_endleaveout'.format(label)]+1)
            ipos = ipos[~np.in1d(ipos, ipos_leave)]

        # Extract matrix index
        index = self._calib._obsdata.index[ipos]

        return index, ipos



    def set_periods(self, scheme='split', nperiods=2, lengthleaveout=None):
        ''' Define calibration periods '''

        # Get core data
        warmup = self._calib.warmup
        self._calib.check()
        nval, _, _, _ = self._calib.model.get_dims('inputs')
        lengthper = int((nval-warmup)/nperiods)
        index = self._calib.model._inputs.index

        if nval < warmup:
            raise ValueError('number of values ({0}) < warmup ({1})'.format(
                nval, warmup))

        # Exclude nan from calibration
        check = np.sum(np.isnan(self._obsdata.data), axis=1) == 0
        self.mask[np.where(check)[0]] = False

        # Set default lengthleaveout
        if lengthleaveout is None:
            lengthleaveout = lengthper

        if lengthleaveout > lengthper:
            raise ValueError(('Length of leaveout period ({0}) >' +
                ' period duration warmup ({1})').format(
                    lengthleaveout, lengthper))


        # Build calibration sub-periods
        self._calperiods = []

        if scheme in ['split', 'leaveout']:

            for i in range(nperiods):

                if scheme == 'split':
                    ipos_cal_start = warmup + i*lengthper
                    ipos_cal_end = ipos_cal_start + lengthper - 1

                    # Break loop when reaching the end of the period
                    if ipos_cal_end >= nval:
                        break

                    # No leave out here
                    ipos_cal_startleaveout = None
                    ipos_cal_endleaveout = None

                    # Validation on the entire period with cal period left out
                    ipos_val_start = warmup
                    ipos_val_end = nval-1
                    ipos_val_startleaveout = ipos_cal_start
                    ipos_val_endleaveout = ipos_cal_end

                else:
                    # Calibration on the entire period
                    ipos_cal_start = warmup
                    ipos_cal_end = nval-1
                    ipos_cal_startleaveout = i*lengthper + warmup
                    ipos_cal_endleaveout = ipos_cal_startleaveout + lengthper - 1

                    ipos_val_start = ipos_cal_startleaveout
                    ipos_val_end = ipos_cal_startleaveout + lengthleaveout - 1
                    ipos_val_startleaveout = None
                    ipos_val_endleaveout = None

                    if ipos_val_end >= nval:
                        break

                per = {
                    'scheme':scheme,
                    'id':'CALPER{0}'.format(i+1),
                    'ipos_cal_start': ipos_cal_start,
                    'ipos_cal_end': ipos_cal_end,
                    'ipos_cal_startleaveout': ipos_cal_startleaveout,
                    'ipos_cal_endleaveout': ipos_cal_endleaveout,
                    'ipos_val_start': ipos_val_start,
                    'ipos_val_end': ipos_val_end,
                    'ipos_val_startleaveout': ipos_val_startleaveout,
                    'ipos_val_endleaveout': ipos_val_endleaveout,
                    'warmup':warmup,
                    'log':{},
                    'completed':False
                }
                self._calperiods.append(per)
        else:
            raise ValueError('Cross validation scheme {0} ' +
                    'not recognised'.format(self._scheme))

        if len(self._calperiods) == 0:
            raise ValueError('No calibration period generated,' +
                    ' modify inputs')


    def run(self, *args, **kwargs):

        nper = len(self._calperiods)
        index = self._obsdata.index

        if nper == 0:
            raise ValueError('No calibration periods defined, please setup')

        for i in range(nper):
            # Set calibration range
            per = self._calperiods[i]
            per['log'][now()] = 'Calibration of subperiod {0} started'.format(i)

            i1 = per['ipos_cal_start']
            i2 = per['ipos_cal_end']
            self.calib.index_cal = index[np.arange(i1, i2)]

            # Set leave out period
            if len(per['index_cal_leaveout'])>0:
                self.calib._obsdata = self._obsdata.clone()
                i1 = per['ipos_cal_startleaveout']
                i2 = per['ipos_cal_endleaveout']
                self.calib.obsdata[np.arange(i1, i2)] = np.nan

            # Define starting point
            if self._explore:
                try:
                    calparams_ini, _, _ = self._calib.explore()
                    per['log'][now()] = 'Explore completed'
                except ValueError, e:
                    per['log'][now()] = 'Explore failed, {0}'.format(e)
                    calparams_ini = self.calib._calparams.default
            else:
                calparams_ini = self.calib._calparams.default

            # Perform fit for calibration period
            per['log'][now()] = 'Fit started'

            try:
                final, out, ofun = self._calib.fit(calparams_ini,
                                            *args, **kwargs)
                per['log'][now()] = 'Fit completed'

                per['completed'] = True
                per['calparams'] = final.copy()
                per['params'] = self._calib.model.params.copy()
                per['simulation'] = out
                per['objfun'] = ofun

            except Exception, e:
                per['log'][now()] = 'Fit failed, {0}'.format(e)




