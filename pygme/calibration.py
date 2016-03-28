import math, time
from datetime import datetime

import numpy as np

from scipy.optimize import fmin_powell

from pygme.data import Vector, Matrix

now = datetime.now

class ErrorFunction(object):

    def __init__(self,
            name,
            nerrparams=0,
            nconstants=0):

        self.name = name
        self._errparams = Vector('errparams', nerrparams)
        self._constants = Vector('constants', nconstants)

    def __str__(self):
        str = 'Objective function {0}\n'.format(self.name)
        str += '  nerrparams : {0}\n'.format(self.nerrparams)
        str += '  nconstant  : {0}\n'.format(self.nconstants)

        return str

    @property
    def nerrparams(self):
        return self._errparams.nval

    @property
    def errparams(self):
        return self._errparams.data

    @errparams.setter
    def errparams(self, value):
        self._errparams.data = value


    @property
    def nconstants(self):
        return self._constants.nval

    @property
    def constants(self):
        return self._constants.data

    @constants.setter
    def constants(self, value):
        self._constants.data = value


    def run(self, obs, sim):
        raise ValueError('Need to override this function')


def powertrans(x, alpha):
    if np.allclose(alpha, 1.):
        return x
    elif np.allclose(alpha, -1.):
        return 1./x
    else:
        xa = np.abs(x)
        xs = np.sign(x)

        if np.allclose(alpha, 0.5):
            return np.sqrt(xa) * xs
        elif np.allclose(alpha, -0.5):
            return xs/np.sqrt(xa)
        else:
            return xa**alpha * xs


class ErrorFunctionSseBias(ErrorFunction):

    def __init__(self):
        ErrorFunction.__init__(self, 'ssebias',
            nerrparams=0,
            nconstants=3)

        self._constants.names = ['varexp', 'errexp', 'biasfactor']
        self._constants.data = [1., 2., 0.]
        self._constants.min = [0., 0., 0.]

    def run(self, obs, sim):

        # Transform variables
        varexp = self._constants['varexp']
        vo = powertrans(obs, varexp)
        vs = powertrans(sim, varexp)

        # Transform error
        err = np.abs(vo-vs)
        errexp = self._constants['errexp']
        if np.allclose(errexp, 1.):
            objfun = np.nanmean(err)
        else:
            objfun = np.nanmean(err**errexp)

        # Bias constraint
        biasfactor = self._constants['biasfactor']
        if not np.allclose(biasfactor, 0.):
            bias = np.mean(obs-sim)/(1+abs(np.mean(obs)))
            objfun = objfun*(1+biasfactor*bias*bias)

        return objfun


class ErrorFunctionSls(ErrorFunction):

    def __init__(self):
        ErrorFunction.__init__(self, 'slslikelihood',
            nerrparams=1,
            nconstants=0)

        self._errparams.names = ['logsigma']
        self._errparams.min = [0.]

    def run(self, obs, sim):
        err = obs-sim
        logsigma = self._errparams.data[0]
        sigma = np.exp(logsigma)
        nval = len(obs)

        ll = np.nansum(err*err)/(2*sigma*sigma) + nval * logsigma

        return ll


class ErrorFunctionQuantileReg(ErrorFunction):

    def __init__(self):
        ErrorFunction.__init__(self, 'quantileregression',
            nerrparams=0,
            nconstants=1)

        self._constants.name = ['quantile']
        self._constants.data = [0.5]
        self._constants.min = [0.]
        self._constants.max = [1.]

    def run(self, obs, sim):

        alpha = self._constants.data[0]
        idx = obs >= sim
        qq1 = alpha * np.nansum(obs[idx]-sim[idx])
        qq2 = (alpha-1) * np.nansum(obs[~idx]-sim[~idx])

        return qq1+qq2



class Calibration(object):

    def __init__(self, model,
            nparams=None,
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

        # Objective function
        if errfun is None:
            self._errfun = ErrorFunctionSseBias()
        else:
            self._errfun = errfun

        # Create vector of calibrated parameters
        # Can be smaller than model parameters and
        # includes error model parameters
        if nparams is None:
            nparams, _ = model.get_dims('params')

        ncalparams = nparams + self.errfun.nerrparams
        self._nparams = nparams
        self._calparams = Vector('calparams', ncalparams, nens_params)
        self._calparams.names = ['X{0}'.format(i) for i in range(nparams)] \
                        + ['XE{0}'.format(i) for i in range(self._errfun.nerrparams)]
        self._calparams.default = np.zeros(ncalparams, dtype=np.float64)
        self._calparams.means = np.zeros(ncalparams, dtype=np.float64)
        self._calparams.covar = np.eye(ncalparams, dtype=np.float64)

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
        str += '  errfun     : {0}\n'.format(self.errfun.name)
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


    def _objfun(self, calparams):

        model = self._model

        # Set model parameters
        calparams = np.atleast_1d(calparams)
        params = self.cal2true(calparams[:self._nparams])
        model.params = params

        # Exit objectif function if parameters hit bounds
        if model._params.hitbounds and self._status:
            return np.inf

        # Set error model parameters
        # (example standard deviation of normal gaussian error model)
        if self._errfun.nerrparams > 0:
            self.errfun.errparams = calparams[self._nparams:]

        # Set start/end of model
        istart = self._index_cal[0] - self.warmup
        if istart < 0:
            raise ValueError('Tried to set model start index before '
                'the first index')

        index = self._obsdata.index
        istart = np.where(index == istart)[0]
        model.index_start = index[istart]
        model.index_end = self._index_cal[-1]

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
        ofun = self._errfun.run(self.obsdata[kk, :], \
                    self._model.outputs[kk, :])

        if not self._minimize:
            ofun *= -1

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


    def cal2true(self, calparams):
        ''' Convert calibrated parameters to true values '''
        return calparams


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


    def get_period_indexes(self, iperiod, type='cal'):

        if iperiod >= len(self._calperiods):
            raise ValueError(('iperiod ({0}) is greater or equal to the' +
                ' number of periods ({1})').format(iperiod, len(self._calperiods)))

        if not type in ['cal', 'val']:
            raise ValueError(('type {0} is not valid. ' +
                'Only cal or val'.format(type)))

        per = self._calperiods[iperiod]

        # Get total period
        i1, i2 = per['ipos_{0}'.format(type)]
        ipos = np.arange(i1, i2+1)

        # Remove leave out if any
        item = 'ipos_{0}_leaveout'.format(type)
        if not per[item] is None:
            i1, i2 = per[item]
            ipos_leave = np.arange(i1, i2+1)
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
                    i0 = warmup + i*lengthper
                    ipos_cal = [i0, i0 + lengthper - 1]

                    # Break loop when reaching the end of the period
                    if ipos_cal[1] >= nval:
                        break

                    # No leave out here
                    ipos_cal_leaveout = None

                    # Validation on the entire period with cal period left out
                    ipos_val = [warmup, nval-1]
                    ipos_val_leaveout = ipos_cal

                else:
                    # Calibration on the entire period
                    ipos_cal = [warmup, nval-1]

                    i0 = i*lengthper + warmup
                    ipos_cal_leaveout = [i0, i0 + lengthper - 1]

                    ipos_val = [i0, i0 + lengthleaveout - 1]
                    ipos_val_leaveout = None

                    if ipos_val[1] >= nval:
                        break

                per = {
                    'scheme':scheme,
                    'id':'CALPER{0}'.format(i+1),
                    'ipos_cal': ipos_cal,
                    'ipos_cal_leaveout': ipos_cal_leaveout,
                    'ipos_val': ipos_val,
                    'ipos_val_leaveout': ipos_val_leaveout,
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

            i1, i2 = per['ipos_cal']
            self.calib.index_cal = index[np.arange(i1, i2+1)]

            # Set leave out period
            if not per['ipos_cal_leaveout'] is None:
                self.calib._obsdata = self._obsdata.clone()
                i1, i2 = per['ipos_cal_leaveout']
                self.calib.obsdata[np.arange(i1, i2+1)] = np.nan

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




